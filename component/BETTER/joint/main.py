import pickle
import sys, os
import argparse
from collections import OrderedDict, defaultdict
from itertools import combinations
import numpy as np
import random
import math
import time
import copy
import torch
from dataset import *
from neural_model import BertClassifier
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
from util import *
from transformers import BertTokenizer, BertModel, BertConfig
from seqeval.metrics import f1_score, accuracy_score, classification_report
from generate_data.uw_json_to_pkl_ace import EVENT_TYPES, ARG_ROLES
from eval import *
import pdb

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def main(args):
    if args.use_bert:
        params = {'batch_size': args.batch,
                  'shuffle': False,
                  'collate_fn': pad_collate_bert}
    elif args.finetune_bert:
        params = {'batch_size': args.batch,
                  'shuffle': False,
                  'collate_fn': pad_collate_bert_finetune}
    else:
        params = {'batch_size': args.batch,
                  'shuffle': False,
                  'collate_fn': pad_collate_bert}

    if args.regen_vocfile:
        data_train = pickle.load(open(args.train_pkl, 'rb'))
        data_dev = pickle.load(open(args.dev_pkl, 'rb'))
        data_test = pickle.load(open(args.test_pkl, 'rb'))

        all_data = data_train + data_dev + data_test
        all_pos = np.concatenate([d['pos_tag'] for d in all_data])
        all_pos = sorted(list(set(all_pos)))  # make pos2idx generation deterministic
        pos_list = all_pos
        pos2idx = OrderedDict(zip(pos_list, range(len(pos_list))))
        pos_emb = np.zeros((len(pos2idx), len(pos2idx)))
        for i in range(pos_emb.shape[0]):
            pos_emb[i, i] = 1.0
        with open('{}/ACE_pos2idx.pickle'.format(args.data_dir), 'wb') as f:
            pickle.dump(pos2idx, f)
        np.save(open('{}/ACE_pos_emb.npy'.format(args.data_dir), 'wb'), pos_emb)

        if args.use_glove:
            ######## quick try
            data_train = pickle.load(open(args.train_pkl, 'rb'))
            data_dev = pickle.load(open(args.dev_pkl, 'rb'))
            data_test = pickle.load(open(args.test_pkl, 'rb'))

            all_data = data_train + data_dev + data_test
            all_tokens = np.concatenate([d['tokens'] for d in all_data])
            all_tokens = list(set(all_tokens))
            if args.lower:
                all_tokens = [i.lower() for i in all_tokens]
                all_tokens = list(set(all_tokens))
            word_list = ['<PAD>', '<UNK>'] + all_tokens
            word2idx = OrderedDict(zip(word_list, range(len(word_list))))
            if args.lower:
                with open('{}/ACE_vocab_lower.pickle'.format(args.data_dir), 'wb') as f:
                    pickle.dump(word2idx, f)
            else:
                with open('{}/ACE_vocab_case.pickle'.format(args.data_dir), 'wb') as f:
                    pickle.dump(word2idx, f)
            print('Loading glove dict from glove file at {}'.format(args.glove_dir))
            glove_dict = read_glove_dict(args.glove_dir)
            word_emb = read_glove_emb(word2idx, glove_dict)
            if args.lower:
                np.save(open('{}/glove_emb_300_lower.npy'.format(args.data_dir), 'wb'), word_emb)
            else:
                np.save(open('{}/glove_emb_300_case.npy'.format(args.data_dir), 'wb'), word_emb)

    if not args.finetune_bert:
        # use embeddings (bert/glove/bert+glove)
        if args.use_glove:
            print ("Loading vocab and word embeddings...")
            if args.lower:
                with open('{}/ACE_vocab_lower.pickle'.format(args.data_dir), 'rb') as f:
                    word2idx = pickle.load(f)
                word_emb = np.load('{}/glove_emb_300_lower.npy'.format(args.data_dir))
            else:
                with open('{}/ACE_vocab_case.pickle'.format(args.data_dir), 'rb') as f:
                    word2idx = pickle.load(f)
                word_emb = np.load('{}/glove_emb_300_case.npy'.format(args.data_dir))
            args.word2idx = word2idx
            args.word_emb = word_emb

        print ("Loading pos vocab...")
        with open('{}/ACE_pos2idx.pickle'.format(args.data_dir), 'rb') as f:
            pos2idx = pickle.load(f)
        args.pos2idx = pos2idx
        print ("Loading pos embeddings...")
        pos_emb = np.load('{}/ACE_pos_emb.npy'.format(args.data_dir))
        args.pos_emb = pos_emb
    elif args.finetune_bert:
        assert args.bert_model_type
        MODELS = [(BertConfig, BertModel, BertTokenizer, args.bert_model_type)]
        for config_class, model_class, tokenizer_class, pretrained_weights in MODELS:
            config = config_class.from_pretrained(pretrained_weights, output_hidden_states=True)
            args.bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            # args.bert_encoder = model_class.from_pretrained(pretrained_weights, config=config)
            # device = torch.device("cuda" if args.cuda else "cpu")
            # args.bert_encoder.to(device)


    if args.trigger_type:
        #### for ACE
        label_t_list = ['<PAD>', 'O']
        for i in EVENT_TYPES:
            label_t_list.append('B-{}'.format(i))
            if not args.use_single_token_tri:
                label_t_list.append('I-{}'.format(i))
        args._label_to_id_t = OrderedDict([(label_t, i) for i, label_t in enumerate(label_t_list)])

    else:
        args._label_to_id_t = OrderedDict([('O', 1), ('B-ANCHOR', 2), ('I-ANCHOR', 3), ('<PAD>', 0)])
    args._id_to_label_t = OrderedDict([(v, k) for k, v in args._label_to_id_t.items()])

    # construct the B2I_trigger dict for use in `train.py`
    B2I_trigger = {}
    B2I_trigger_string = {}
    for tri_label, tri_id in args._label_to_id_t.items():
        if tri_label.startswith('B-'):
            if not args.use_single_token_tri:
                B2I_trigger[args._label_to_id_t[tri_label]] = args._label_to_id_t['I-{}'.format(tri_label[2:])]
                B2I_trigger_string[tri_label] = 'I-{}'.format(tri_label[2:])
            else:
                B2I_trigger[args._label_to_id_t[tri_label]] = args._label_to_id_t['B-{}'.format(tri_label[2:])]
                B2I_trigger_string[tri_label] = 'I-{}'.format(tri_label[2:])
    args.B2I_trigger = B2I_trigger
    args.B2I_trigger_string = B2I_trigger_string

    # args._label_to_id_e = OrderedDict([('O', 1), ('B-ENT', 2), ('I-ENT', 3), ('<PAD>', 0)])
    args._label_to_id_r = OrderedDict([('None', 0), ('agent', 1), ('patient', 2), ('agent--patient', 3)])

    if args.split_arg_role_label:
        # args._label_to_id_e = OrderedDict([('<PAD>', 0), ('O', 1), \
        #                                    ('B-AGENT', 2), ('I-AGENT', 3), \
        #                                    ('B-PATIENT', 4), ('I-PATIENT', 5)])
        #### for ACE
        label_e_list = ['<PAD>', 'O']
        for i in ARG_ROLES:
            label_e_list.append('B-{}'.format(i))
            label_e_list.append('I-{}'.format(i))
        args._label_to_id_e = OrderedDict([(label_e, i) for i, label_e in enumerate(label_e_list)])
        args._id_to_label_e = OrderedDict([(v, k) for k, v in args._label_to_id_e.items()])
    else:
        args._label_to_id_e = OrderedDict([('<PAD>', 0), ('O', 1), \
                                           ('B-AGENT', 2), ('I-AGENT', 3), \
                                           ('B-PATIENT', 2), ('I-PATIENT', 3)])
        args._id_to_label_e = OrderedDict([(0, '<PAD>'), (1, 'O'), (2, 'B-ENT'), (3, 'I-ENT')])

    # construct the B2I_arg dict for use in `train.py`
    B2I_arg = {}
    B2I_arg_string = {}
    for arg_label, arg_id in args._label_to_id_e.items():
        if arg_label.startswith('B-'):
            B2I_arg[args._label_to_id_e[arg_label]] = args._label_to_id_e['I-{}'.format(arg_label[2:])]
            B2I_arg_string[arg_label] = 'I-{}'.format(arg_label[2:])
    args.B2I_arg = B2I_arg
    args.B2I_arg_string = B2I_arg_string

    # if args.decode_w_ents_mask:
    args._label_to_id_e_sent = OrderedDict(
             [('<PAD>', 0),
             ('O', 1),
             ('B-ORG', 2), ('I-ORG', 3),
             ('B-WEA', 4), ('I-WEA', 5),
             ('B-VEH', 6), ('I-VEH', 7),
             ('B-GPE', 8), ('I-GPE', 9),
             ('B-LOC', 10), ('I-LOC', 11),
             ('B-FAC', 12), ('I-FAC', 13),
             ('B-PER', 14), ('I-PER', 15)]
            )
    # else:
        # args._label_to_id_e_sent = OrderedDict([('<PAD>', 0), ('O', 1), ('B-ENT', 2), ('I-ENT', 3)])
    args._id_to_label_e_sent = OrderedDict([(v, k) for k, v in args._label_to_id_e_sent.items()])


    combs = pickle.load(open('all_ace/all_uw.comb.pkl', 'rb'))
    # if args.decode_w_arg_role_mask:
    # tri--arg_role combs
    _t_id_to_e_id = defaultdict(list)
    for tri, argus in combs['tri_arg_comb'].items():
        t_id = args._label_to_id_t['B-{}'.format(tri)]
        for argu in argus:
            _t_id_to_e_id[t_id].append(args._label_to_id_e['B-{}'.format(argu)])
            _t_id_to_e_id[t_id].append(args._label_to_id_e['I-{}'.format(argu)])
    args._t_id_to_e_id = _t_id_to_e_id
    # ent--arg_role combs
    _ner_id_to_e_id = defaultdict(list)
    for ent, argus in combs['ent_arg_comb'].items():
        ner_id = args._label_to_id_e_sent['B-{}'.format(ent)]
        for argu in argus:
            _ner_id_to_e_id[ner_id].append(args._label_to_id_e['B-{}'.format(argu)])
            _ner_id_to_e_id[ner_id].append(args._label_to_id_e['I-{}'.format(argu)])
            # +1 accounts for I-<ent_type>
            _ner_id_to_e_id[ner_id+1].append(args._label_to_id_e['B-{}'.format(argu)])
            _ner_id_to_e_id[ner_id+1].append(args._label_to_id_e['I-{}'.format(argu)])
    args._ner_id_to_e_id = _ner_id_to_e_id

    print(args.B2I_trigger)
    print(args.B2I_trigger_string)
    print(args._id_to_label_t)
    print(args.B2I_arg)
    print(args.B2I_arg_string)
    print(args._id_to_label_e)
    print(len(args._id_to_label_t))
    print(len(args._id_to_label_e))

    train_data = EventDataset(args.train_pkl, args)
    train_generator = data.DataLoader(train_data, **params)

    dev_data = EventDataset(args.dev_pkl, args)
    dev_generator = data.DataLoader(dev_data, **params)

    test_data = EventDataset(args.test_pkl, args)
    test_generator = data.DataLoader(test_data, **params)

    if args.model_type == 'local':
        if args.argument_weight > 0 and args.trigger_weight > 0 and args.pipe_epochs >= 1000:
            task = 'multitask_pipe'
        elif args.argument_weight > 0 and args.trigger_weight > 0 and args.pipe_epochs < 1000:
            task = 'multitask_joint'
        elif args.argument_weight > 0 and args.trigger_weight <= 0:
            task = 'singletask_arg'
        elif args.argument_weight <= 0 and args.trigger_weight > 0:
            task = 'singletask_trigger'
        elif args.ner_weight> 0 and args.trigger_weight<=0 and args.argument_weight <=0:
            task = 'singletask_ner'
    elif args.model_type == 'global':
        task = 'ssvm'
    else:
        print('Error! Have to indicate either argument_weight > 0 or trigger_weight > 0 or both')
        assert False

    args.log_dir=\
            '{}{}_bert{}_batch{}_epoch{}_lr{}_opt{}_hid{}_dp{}_act{}_crfa{}_att{}_attmthd{}_attfunc{}_attdrp{}_attlinearout{}_norm{}_tw{}_aw{}_tb{}_ab{}_finetunebert{}_lrt{}_lra{}_warm{}_bertenc{}_{}'.\
            format(args.tensorboard_dir, task, args.use_bert, args.batch, args.epochs,
                   args.lr, args.opt, args.hid, args.dropout,
                   args.activation, args.use_crf_a, args.use_att, args.att_mthd,
                   args.att_func, args.att_dropout, args.use_att_linear_out, args.norm,
                   args.trigger_weight, args.argument_weight, args.bias_t, args.bias_a,
                   args.finetune_bert, args.lr_other_t, args.lr_other_a, args.num_warmup_steps, args.bert_encode_mthd, args.bert_model_type.split('-')[1])

    args.save_dir=\
            '{}{}_bert{}_batch{}_epoch{}_lr{}_opt{}_hid{}_dp{}_act{}_crfa{}_att{}_attmthd{}_attfunc{}_attdrp{}_attlinearout{}_norm{}_tw{}_aw{}_tb{}_ab{}_finetunebert{}_lrt{}_lra{}_warm{}_bertenc{}_{}'.\
            format(args.model_dir, task, args.use_bert, args.batch, args.epochs,
                   args.lr, args.opt, args.hid, args.dropout,
                   args.activation, args.use_crf_a, args.use_att, args.att_mthd,
                   args.att_func, args.att_dropout, args.use_att_linear_out, args.norm,
                   args.trigger_weight, args.argument_weight, args.bias_t, args.bias_a,
                   args.finetune_bert, args.lr_other_t, args.lr_other_a, args.num_warmup_steps, args.bert_encode_mthd, args.bert_model_type.split('-')[1])
    args.task = task
    print('Task: {}'.format(args.task))
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    model = BertClassifier(args)
    if torch.cuda.device_count() > 1 and args.multigpu:
        print('Using {} GPUs'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    if args.model_type == 'local':
        from train import NNClassifier
    elif args.model_type == 'global':
        from train_ssvm import NNClassifier
    else:
        print('Unsupported model')
        assert False
    trainer = NNClassifier(args, model)
    if args.load_model:
        if args.load_model_single:
            # specify trigger model attributes
            args.use_crf_ner=False
            args.use_att=False
            args.bert_model_type='bert-large-uncased'
            args.hid_lastmlp=600
            args.ner_weight=0.0
            model_t = BertClassifier(args)
            # specify argument model attributes
            args.use_crf_ner=False
            args.use_att=True
            args.bert_model_type='bert-base-uncased'
            args.hid_lastmlp=384
            args.ner_weight=0.0
            model = BertClassifier(args)
            # specify NER model attributes
            args.use_crf_ner=True
            args.use_att=True
            args.bert_model_type='bert-large-cased'
            args.hid_lastmlp=512
            args.ner_weight=1.0
            model_ner = BertClassifier(args)
            trainer = NNClassifier(args, model, model_t, model_ner)
            trainer.load(filename=args.load_model_path, filename_t=args.load_model_path_t, filename_ner=args.load_model_path_ner)
        else:
            trainer.load(filename=args.load_model_path)
    if args.do_train:
        # # generate tmp gold pkl for dev, for  internal eval during training
        # y_trues_t, y_preds_t, y_trues_e, y_preds_e, y_trues_e_sent, y_preds_e_sent, sent_ids_out, y_trues_ner, y_preds_ner =\
        #     trainer.predict(dev_generator, task, test=False, use_gold=True, e2e_eval=True, generate_eval_file=True)
        # write_pkl(sent_ids_out, y_trues_t, y_preds_t, y_trues_e, y_preds_e, y_trues_e_sent, y_preds_e_sent, 'tmp/', args.split_arg_role_label, 'gold_dev_internal_tritypeTrue_ft_uw')
        # # generate tmp gold pkl for test, for internal eval during training
        # y_trues_t, y_preds_t, y_trues_e, y_preds_e, y_trues_e_sent, y_preds_e_sent, sent_ids_out, y_trues_ner, y_preds_ner =\
        #     trainer.predict(test_generator, task, test=False, use_gold=True, e2e_eval=True, generate_eval_file=True)
        # write_pkl(sent_ids_out, y_trues_t, y_preds_t, y_trues_e, y_preds_e, y_trues_e_sent, y_preds_e_sent, 'tmp/', args.split_arg_role_label, 'gold_test_internal_tritypeTrue_ft_uw')
        best_f1, best_epoch, test_f1_e, test_f1_t = trainer.train_epoch(train_generator, dev_generator, test_generator, task)
        # write_result('results.pkl', args, {'dev_f1':best_f1, 'test_f1_t':test_f1_t, 'test_f1_e':test_f1_e})
    if args.do_test:
        # test for NER
        y_trues_t, y_preds_t, y_trues_e, y_preds_e, y_trues_e_sent, y_preds_e_sent, _, y_trues_ner, y_preds_ner = \
                trainer.predict(test_generator, task, test=True, use_gold=True, e2e_eval=False)
        # if epoch == 3:
        #     pdb.set_trace()
        if len(y_trues_ner) > 0:
            f1_ner = f1_score(y_trues_ner, y_preds_ner)
            acc_ner = accuracy_score(y_trues_ner, y_preds_ner)
            print('ner f1: {:.4f}, ner acc: {:.4f}'.format(f1_ner, acc_ner))
        # # test for trigger + arguments
        # y_trues_t, y_preds_t, y_trues_e, y_preds_e, y_trues_e_sent, y_preds_e_sent, sent_ids_out, y_trues_ner, y_preds_ner =\
        #     trainer.predict(test_generator, task, test=True, use_gold=args.eval_on_gold_tri, e2e_eval=args.e2e_eval)
    if args.write_pkl:
        out_pkl_dir = args.out_pkl_dir
        write_pkl(sent_ids_out, y_trues_t, y_preds_t, y_trues_e, y_preds_e, y_trues_e_sent, y_preds_e_sent, out_pkl_dir, args.split_arg_role_label, args.out_pkl_name)
        eval_ace('tmp/gold_test_internal_tritypeTrue_ft_uw_cls.pkl', os.path.join(out_pkl_dir, args.out_pkl_name), args.B2I_trigger_string, args.B2I_arg_string)

def get_uniq_id_list(sent_ids):
    '''
    remove duplicated ids in the list of sent_ids, with preserved order
    need this b/c the sent_ids output might be extended when doing event-level argument
    we reduce them to unique ids for alignment to sent-level trigger/arg eval
    '''
    output = list(OrderedDict.fromkeys(sent_ids))
    return output

def write_pkl(sent_ids, y_trues_t, y_preds_t, y_trues_e, y_preds_e, y_trues_e_sent, y_preds_e_sent, out_pkl_dir, split_arg_role, out_pkl_name, y_trues_ner=None, y_preds_ner=None):
    #ensure dir
    if not os.path.exists(out_pkl_dir):
        os.makedirs(out_pkl_dir)
    if y_trues_ner is not None and y_preds_ner is not None:
        # meaning this is the output of a single task NER
        output_ner = []
        sent_ids_uniq = get_uniq_id_list(sent_ids)
        for i in range(len(sent_ids_uniq)):
            output_ner.append({'sent_id': sent_ids_uniq[i], 'gold': y_trues_ner[i], 'pred': y_preds_ner[i]})
        with open('{}/ner.pkl'.format(out_pkl_dir), 'wb') as f:
            pickle.dump(output_ner, f)
        print('pkl saved at {}/ner.pkl'.format(out_pkl_dir))
        # for NER case, after prediction just exit to avoid accidently writing out other meaning less pickles
        sys.exit()


    if len(y_trues_e) > 0 and len(y_preds_e) == 0 and len(y_trues_t) > 0 and len(y_preds_t) == 0:
        #to generate gold event-level trigger and arg seqs
        output_t_e_event = []
        for i in range(len(sent_ids)):
            output_t_e_event.append({'sent_id': sent_ids[i], 'gold_trigger': y_trues_t[i], 'gold_arg': y_trues_e[i]})
        if split_arg_role is True:
            with open('{}/{}_cls.pkl'.format(out_pkl_dir, out_pkl_name), 'wb') as f:
                pickle.dump(output_t_e_event, f)
            print('pkl saved at {}/{}_cls.pkl'.format(out_pkl_dir, out_pkl_name))
        elif split_arg_role is False:
            with open('{}/{}_id.pkl'.format(out_pkl_dir, out_pkl_name), 'wb') as f:
                pickle.dump(output_t_e_event, f)
            print('pkl saved at {}/{}_id.pkl'.format(out_pkl_dir, out_pkl_name))

    if len(y_trues_t) > 0 and len(y_preds_t) > 0:
        # meaning either multitask or singletask_trigger, so that trigger can be evaluated
        output_t = []
        sent_ids_uniq = get_uniq_id_list(sent_ids)
        for i in range(len(sent_ids_uniq)):
            output_t.append({'sent_id': sent_ids_uniq[i], 'gold': y_trues_t[i], 'pred': y_preds_t[i]})
        with open('{}/sent_level_trigger.pkl'.format(out_pkl_dir), 'wb') as f:
            pickle.dump(output_t, f)
        print('pkl saved at {}/sent_level_trigger.pkl'.format(out_pkl_dir))
    if len(y_trues_e) > 0 and len(y_preds_e) > 0:
        # meaning either multitask or singletask_argument, so that event-level argument id/cls can be evaluated
        output_e_event = []
        for i in range(len(sent_ids)):
            output_e_event.append({'sent_id': sent_ids[i], 'gold': y_trues_e[i], 'pred': y_preds_e[i]})
        if split_arg_role is True:
            with open('{}/event_level_arg_cls.pkl'.format(out_pkl_dir), 'wb') as f:
                pickle.dump(output_e_event, f)
            print('pkl saved at {}/event_level_arg_cls.pkl'.format(out_pkl_dir))
        elif split_arg_role is False:
            with open('{}/event_level_arg_id.pkl'.format(out_pkl_dir), 'wb') as f:
                pickle.dump(output_e_event, f)
            print('pkl saved at {}/event_level_arg_id.pkl'.format(out_pkl_dir))
    if len(y_trues_e_sent) > 0 and len(y_preds_e_sent) > 0:
        # meaning the sent level aggregation is enabled, so that argument
        output_e_sent = []
        sent_ids_uniq = get_uniq_id_list(sent_ids)
        for i in range(len(sent_ids_uniq)):
            output_e_sent.append({'sent_id': sent_ids_uniq[i], 'gold': y_trues_e_sent[i], 'pred': y_preds_e_sent[i]})
        with open('{}/sent_level_arg_id.pkl'.format(out_pkl_dir), 'wb') as f:
            pickle.dump(output_e_sent, f)
        print('pkl saved at {}/sent_level_arg_id.pkl'.format(out_pkl_dir))
    if len(y_trues_e) == 0 and len(y_preds_e) > 0:
        # meaning this is the final_test case, i.e. end2end for event-level arg cls
        output_t_e_event = []
        # output_e_event = []
        for i in range(len(sent_ids)):
            output_t_e_event.append({'sent_id': sent_ids[i], 'pred_trigger': y_preds_t[i], 'pred_arg': y_preds_e[i]})
        if args.split_arg_role_label is True:
            with open('{}/{}'.format(out_pkl_dir, out_pkl_name), 'wb') as f:
                pickle.dump(output_t_e_event, f)
            print('pkl saved at {}/{}'.format(out_pkl_dir, out_pkl_name))
    # if len(y_trues_e) == 0 and len(y_preds_e) > 0:
    #     # meaning this is the final_test case, i.e. end2end for event-level arg cls
    #     output_t_e_event = []
    #     # output_e_event = []
    #     for i in range(len(sent_ids)):
    #         output_t_e_event.append({'sent_id': sent_ids[i], 'pred_trigger': y_preds_t[i], 'pred_arg': y_preds_e[i]})
    #     if split_arg_role is True:
    #         with open('{}/end2end_event_level_arg_cls.pkl'.format(out_pkl_dir), 'wb') as f:
    #             pickle.dump(output_t_e_event, f)
    #         print('pkl saved at {}/end2end_event_level_arg_cls.pkl'.format(out_pkl_dir))
    #     elif split_arg_role is False:
    #         with open('{}/end2end_event_level_arg_id.pkl'.format(out_pkl_dir), 'wb') as f:
    #             pickle.dump(output_t_e_event, f)
    #         print('pkl saved at {}/end2end_event_level_arg_id.pkl'.format(out_pkl_dir))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # data argument
    p.add_argument('-data_dir', type=str, default='./all_liz')
    p.add_argument('-train_pkl', type=str, default='./all_liz/train_w-pairs_xlmroberta.pkl')
    p.add_argument('-dev_pkl', type=str, default='./all_liz/dev_w-pairs_xlmroberta.pkl')
    p.add_argument('-test_pkl', type=str, default='./all_liz/test_w-pairs_xlmroberta.pkl')
    p.add_argument('-glove_dir', type=str, default='./glove.6B/glove.6B.300d.txt')
    p.add_argument('-out_pkl_dir', type=str, default='./out_pkl')
    p.add_argument('-out_pkl_name', type=str, default='test.pkl')

    # training argument
    p.add_argument('-batch', type=int, default=16)
    p.add_argument('-iter_size', type=int, default=2, help='how many batches to accumulate')
    p.add_argument('-epochs', type=int, default=40)
    p.add_argument('-pipe_epochs', type=int, default=1000)
    p.add_argument('-tri_start_epochs', type=int, default=50)
    p.add_argument('-lr', type=float, default=0.001)
    p.add_argument('-lr_other_ner', type=float, default=0.001)
    p.add_argument('-lr_other_t', type=float, default=0.001)
    p.add_argument('-lr_other_a', type=float, default=0.001)
    p.add_argument('-num_warmup_steps', type=int, default=300)
    p.add_argument('-opt', choices=['adam', 'sgd', 'bertadam'], default='adam')
    p.add_argument('-momentum', type=float, default=0.9)
    p.add_argument('-cuda', type=str2bool, default=True)
    p.add_argument('-multigpu', type=str2bool, default=False)
    p.add_argument('-params', type=dict, default={})
    p.add_argument('-patience', type=int, default=10000)  # disable early stopping
    p.add_argument('-do_train', type=str2bool, default=True)
    p.add_argument('-do_test', type=str2bool, default=False, help='use with load_model. Useful when want to skip training and directly test the loaded model on test')
    p.add_argument('-write_pkl', type=str2bool, default=False, help='whether write out the pkl files for eval')
    p.add_argument('-eval_on_gold_tri', type=str2bool, default=True)
    p.add_argument('-e2e_eval', type=str2bool, default=True, help='when eval_on_gold_tri is True, e2e_eval should be False; when eval_on_gold_tri is False, e2e_eval should be True')
    p.add_argument('-train_on_e2e_data', type=str2bool, default=True, help='when True, train on all predicted trigger pairs, otherwise train on gold+predicted triggers data')
    p.add_argument('-model_type', type=str, choices=['local', 'global'])
    p.add_argument('-do_ssvm_train', type=str2bool, default=False)
    p.add_argument('-tri_partial_match', type=str2bool, default=True)
    p.add_argument('-gold_ent', type=str2bool, default=False, help='when True, will do the joint tri-arg model')
    p.add_argument('-use_single_token_tri', type=str2bool, default=True)

    # arguments for RNN model
    p.add_argument('-hid', type=int, default=150)
    p.add_argument('-hid_lastmlp', type=int, default=600)
    p.add_argument('-num_layers', type=int, default=1)
    p.add_argument('-dropout', type=float, default=0.4)
    p.add_argument('-activation', type=str, default='relu',
                   choices=[None, 'tanh', 'relu', 'leakyrelu', 'prelu', 'None'])

    # multi-task argument
    p.add_argument('-ner_weight', type=float, default=0.0)
    p.add_argument('-argument_weight', type=float, default=5.0)
    p.add_argument('-trigger_weight', type=float, default=1.0)

    # hyper-parameter
    p.add_argument('-finetune_bert', type=str2bool, default=False)
    p.add_argument('-bert_model_type', type=str, default='bert-base-uncased')
    p.add_argument('-bert_encode_mthd', type=str, choices=['average', 'max', 'head'], default='average')
    p.add_argument('-use_bert', type=str2bool, default=True)
    p.add_argument('-use_glove', type=str2bool, default=True)
    p.add_argument('-bert_dim', type=int, default=1024)
    p.add_argument('-use_pos', type=str2bool, default=True)
    p.add_argument('-regen_vocfile', type=str2bool, default=False)
    p.add_argument('-trainable_emb', type=str2bool, default=True)
    p.add_argument('-trainable_pos_emb', type=str2bool, default=False)
    p.add_argument('-random_seed', type=int, default=123)
    p.add_argument('-lower', type=str2bool, default=False)
    p.add_argument('-use_crf_ner', type=str2bool, default=True)
    p.add_argument('-use_crf_t', type=str2bool, default=True)
    p.add_argument('-use_crf_a', type=str2bool, default=True)
    p.add_argument('-use_att', type=str2bool, default=True)
    p.add_argument('-att_func', type=str, choices=['general', 'dot'], default='general', help='function to compute attention weights')
    p.add_argument('-att_dropout', type=float, default=0.0,
                   help='dropout applied to the multi-head attn internal matrix, when att_func is general. If set to 0.0, then no dropout')
    p.add_argument('-use_att_linear_out', type=str2bool, default=True, help='whether to use WO when computing multi-head attention')
    p.add_argument('-norm', type=str2bool, default=True, help='whether apply the norm layer for the representation after attn')
    p.add_argument('-att_pool', type=str, choices=['max'], default='max')
    p.add_argument('-att_mthd', type=str, default='cat',
                   choices=[None, 'cat', 'gate', 'att_cat', 'att_mul_cat', 'att_sum', 'att_sub', 'att_mul_sum', 'att_mul_sub', 'att_mul_replace',\
                           'cat_self_att_sub_elem_prod', 'cat_self_att_sub', 'cat_self_att_elem_prod'])
    p.add_argument('-split_arg_role_label', type=str2bool, default=True, help='split B-ENT to B-agent and B-patient for argument role')
    p.add_argument('-eval_sent_arg_role', type=str2bool, default=False, help='should set to True when using attn and when split_arg_role_label is False')
    p.add_argument('-trigger_type', type=str2bool, default=False, help='whether to label trigger label with event types')
    p.add_argument('-k_ner', type=int, default=1, help='beam size for ner')
    p.add_argument('-k_tri', type=int, default=1, help='beam size for trigger')
    p.add_argument('-k_arg', type=int, default=1, help='beam size for argument')
    p.add_argument('-margin', type=str, default='e2e_eval', choices=['const', 'e2e_eval'])
    p.add_argument('-bias_t', type=float, default=5.0, help='bias weight for triggers that are NOT O. For O trigger, weight is 1.0')
    p.add_argument('-bias_a', type=float, default=5.0, help='bias weight for args that are NOT O. For O args, weight is 1.0')
    p.add_argument('-decode_w_ents_mask', type=str2bool, default=False)
    p.add_argument('-decode_w_arg_role_mask_by_tri', type=str2bool, default=False)
    p.add_argument('-decode_w_trigger_mask', type=str2bool, default=False)
    p.add_argument('-decode_w_arg_role_mask_by_ent', type=str2bool, default=False)

    # save arugment
    p.add_argument('-tensorboard_dir', type=str, default='./logs/')
    p.add_argument('-model_dir', type=str, default='./saved_models/')
    p.add_argument('-load_model', type=str2bool, default=False)
    p.add_argument('-load_model_path', type=str, default='./saved_models/multitask_joint_bertTrue_batch16_epoch60_lr0.001_optadam_hid150_rnnlay1_dp0.5_acttanh_crfTrue_attTrue_attmthdcat_attfuncgeneral_attdrp0.0_attlinearoutTrue_normTrue_tw1.0_aw5.0_splitargTrue_30/best_model.pt')
    p.add_argument('-load_model_path_t', type=str, default='./saved_models/multitask_joint_quadFalse/best_model.pt')
    p.add_argument('-load_model_path_ner', type=str, default='./saved_models/multitask_joint_quadFalse/best_model.pt')
    p.add_argument('-load_model_single', type=str2bool, default=False)
    args = p.parse_args()
    if args.activation=='None':
        args.activation=None
    torch.manual_seed(args.random_seed)
    main(args)


