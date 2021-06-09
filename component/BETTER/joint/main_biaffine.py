import pickle
import sys, os
import argparse
from collections import OrderedDict
from itertools import combinations
import numpy as np
import random
import math
import time
import copy
import torch
from dataset import *
from neural_model import BertClassifier, BiaffineClassifier
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
from generate_data.uw_json_to_pkl_ace import EVENT_TYPES, ARG_ROLES
from util import *
import pdb
from train_biaffine import NNClassifier

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
        MODELS = [(BertModel, BertTokenizer, args.bert_model_type)]
        for model_class, tokenizer_class, pretrained_weights in MODELS:
            args.bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            args.bert_encoder = model_class.from_pretrained(pretrained_weights)
            device = torch.device("cuda" if args.cuda else "cpu")
            args.bert_encoder.to(device)


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

    #### for ACE
    label_e_list = ['<PAD>', 'O']
    for i in ARG_ROLES:
        label_e_list.append('B-{}'.format(i))
        label_e_list.append('I-{}'.format(i))
    args._label_to_id_e = OrderedDict([(label_e, i) for i, label_e in enumerate(label_e_list)])
    args._id_to_label_e = OrderedDict([(v, k) for k, v in args._label_to_id_e.items()])

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


    text = 'biaffine_bert{}_batch{}_ep{}_pipeep{}_triep{}_lr{}_opt{}_hid{}_rnn{}_dp{}_act{}_crf{}_tw{}_aw{}_slw{}_slloss{}_wbytri{}_predonly{}_partmatch{}_trigtype{}_argubias{}_poolprob{}_mergeBIO{}_entlossj{}'.\
            format(args.use_bert, args.batch, args.epochs, args.pipe_epochs, args.tri_start_epochs,
                   args.lr, args.opt, args.hid, args.num_layers,
                   args.dropout, args.activation, args.use_crf,
                   args.trigger_weight, args.argument_weight, args.sent_level_weight,
                   args.sent_level_loss, args.weighted_bytrig, args.train_on_e2e_data,
                   args.tri_partial_match, args.trigger_type,
                   args.argument_bias, args.argupooled_byprob,
                   args.merge_arguBIO, args.entloss_in_joint)

    args.log_dir='{}{}'.format(args.tensorboard_dir, text)

    save_dir = '{}{}'.format(args.model_dir, text)
    args.text = text
    args.save_dir = save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model = BiaffineClassifier(args)
    trainer = NNClassifier(args, model)
    if args.load_model:
        trainer.load(filename=args.load_model_path)
    if args.do_train:
        if args.regen_goldpkl:
            # generate tmp gold pkl for dev, for internal eval during training
            y_trues_t, _, y_trues_e, _, y_trig_input, sent_ids_out =\
                trainer.predict(dev_generator, test=False, blind_test=False, use_gold_trig=True)
            write_pkl(sent_ids_out, y_trig_input, [], y_trues_e, [], True, 'tmp', 'gold_dev_biaffine_internal')
            # generate tmp gold pkl for test, for internal eval during training
            y_trues_t, _, y_trues_e, _, y_trig_input, sent_ids_out =\
                trainer.predict(test_generator, test=False, blind_test=False, use_gold_trig=True)
            write_pkl(sent_ids_out, y_trig_input, [], y_trues_e, [], True, 'tmp', 'gold_test_biaffine_internal')
        best_f1, best_epoch, test_f1_t, test_f1_e, test_f1_e2e = trainer.train_epoch(train_generator, dev_generator, test_generator, None)
        # write_result_biaffine('results_biaffine.pkl', args, {'dev_f1':best_f1, 'test_f1_t':test_f1_t, 'test_f1_e':test_f1_e, 'test_f1_e2e':test_f1_e2e})
    if args.do_test:
        y_trues_t, y_preds_t, y_trues_e, y_preds_e, y_trig_input, sent_ids_out =\
            trainer.predict(test_generator, test=True, blind_test=args.blind_test, use_gold_trig=args.eval_on_gold_tri)
    if args.write_pkl:
        assert args.do_test
        out_pkl_dir = args.out_pkl_dir
        write_pkl(sent_ids_out, [], y_trig_input, y_trues_e, y_preds_e,
                  True, out_pkl_dir, args.out_pkl_name)

def get_uniq_id_list(sent_ids):
    '''
    remove duplicated ids in the list of sent_ids, with preserved order
    need this b/c the sent_ids output might be extended when doing event-level argument
    we reduce them to unique ids for alignment to sent-level trigger/arg eval
    '''
    output = list(OrderedDict.fromkeys(sent_ids))
    return output

def write_pkl(sent_ids, y_trues_t, y_preds_t, y_trues_e, y_preds_e, split_arg_role, out_pkl_dir, out_pkl_name):
    #ensure dir
    if not os.path.exists(out_pkl_dir):
        os.makedirs(out_pkl_dir)
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
        # # meaning either multitask or singletask_argument, so that event-level argument id/cls can be evaluated
        # output_e_event = []
        # for i in range(len(sent_ids)):
        #     output_e_event.append({'sent_id': sent_ids[i], 'gold': y_trues_e[i], 'pred': y_preds_e[i]})
        # if split_arg_role is True:
        #     with open('{}/event_level_arg_cls.pkl'.format(out_pkl_dir), 'wb') as f:
        #         pickle.dump(output_e_event, f)
        #     print('pkl saved at {}/event_level_arg_cls.pkl'.format(out_pkl_dir))
        # elif split_arg_role is False:
        #     with open('{}/event_level_arg_id.pkl'.format(out_pkl_dir), 'wb') as f:
        #         pickle.dump(output_e_event, f)
        #     print('pkl saved at {}/event_level_arg_id.pkl'.format(out_pkl_dir))

        ## meaning this is the prediction given gold trigger case
        output_t_e_event = []
        for i in range(len(sent_ids)):
            output_t_e_event.append({'sent_id': sent_ids[i], 'pred_trigger': y_preds_t[i], 'pred_arg': y_preds_e[i]})
        if split_arg_role is True:
            with open('{}/{}'.format(out_pkl_dir, out_pkl_name), 'wb') as f:
                pickle.dump(output_t_e_event, f)
            print('pkl saved at {}/{}'.format(out_pkl_dir, out_pkl_name))

    if len(y_trues_e) == 0 and len(y_preds_e) > 0:
        # meaning this is the final_test case, i.e. end2end for event-level arg cls
        output_t_e_event = []
        for i in range(len(sent_ids)):
            output_t_e_event.append({'sent_id': sent_ids[i], 'pred_trigger': y_preds_t[i], 'pred_arg': y_preds_e[i]})
        if split_arg_role is True:
            with open('{}/{}'.format(out_pkl_dir, out_pkl_name), 'wb') as f:
                pickle.dump(output_t_e_event, f)
            print('pkl saved at {}/{}'.format(out_pkl_dir, out_pkl_name))

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
    p.add_argument('-glove_dir', type=str, default='./glove/glove.6B.300d.txt')
    p.add_argument('-out_pkl_dir', type=str, default='./out_pkl')
    p.add_argument('-out_pkl_name', type=str, default='test.pkl')

    # training argument
    p.add_argument('-batch', type=int, default=16)
    p.add_argument('-iter_size', type=int, default=16)
    p.add_argument('-epochs', type=int, default=200)
    p.add_argument('-tri_start_epochs', type=int, default=50)
    p.add_argument('-pipe_epochs', type=int, default=70)
    p.add_argument('-lr', type=float, default=0.002)
    p.add_argument('-lr_other_t', type=float, default=0.002)
    p.add_argument('-lr_other_a', type=float, default=0.002)
    p.add_argument('-num_warmup_steps', type=int, default=300)
    p.add_argument('-opt', choices=['adam', 'sgd', 'bertadam'], default='adam')
    p.add_argument('-momentum', type=float, default=0.9)
    p.add_argument('-cuda', type=str2bool, default=True)
    p.add_argument('-params', type=dict, default={})
    p.add_argument('-patience', type=int, default=10)  # disable early stopping
    p.add_argument('-do_train', type=str2bool, default=True)
    p.add_argument('-do_test', type=str2bool, default=False, help='use with load_model. Useful when want to skip training and directly test the loaded model on test')
    p.add_argument('-write_pkl', type=str2bool, default=False, help='whether write out the pkl files for eval')
    p.add_argument('-train_on_e2e_data', type=str2bool, default=False, help='when True, train on all predicted trigger pairs, otherwise train on gold+predicted triggers data') # True works also good though (test best)
    p.add_argument('-tri_partial_match', type=str2bool, default=True)
    p.add_argument('-regen_goldpkl', type=str2bool, default=False)
    p.add_argument('-blind_test', type=str2bool, default=True)
    p.add_argument('-eval_on_gold_tri', type=str2bool, default=False)
    p.add_argument('-gold_ent', type=str2bool, default=False, help='when True, will do the joint tri-arg model')
    p.add_argument('-use_single_token_tri', type=str2bool, default=True)

    # arguments for RNN model
    p.add_argument('-hid', type=int, default=300)
    p.add_argument('-hid_lastmlp', type=int, default=600)
    p.add_argument('-num_layers', type=int, default=2)
    p.add_argument('-dropout', type=float, default=0.5)
    p.add_argument('-activation', type=str, default='leakyrelu',
                   choices=[None, 'tanh', 'relu', 'leakyrelu', 'prelu', 'None'])

    # multi-task argument
    p.add_argument('-argument_weight', type=float, default=1.0)
    p.add_argument('-trigger_weight', type=float, default=0.02)
    p.add_argument('-column_weight', type=float, default=0.02)
    p.add_argument('-sent_level_weight', type=float, default=0.2)

    # bias factor for training
    p.add_argument('-argument_bias', type=float, default=2.5,
                   help='the factor will multiply to the argument token that are not O in training')
    # decode tricks
    p.add_argument('-argupooled_byprob', type=str2bool, default=True)
    p.add_argument('-merge_arguBIO', type=str2bool, default=False)
    p.add_argument('-sent_level_loss', type=str2bool, default=False)
    p.add_argument('-weighted_bytrig', type=str2bool, default=False)
    p.add_argument('-entloss_in_joint', type=str2bool, default=False)

    # hyper-parameter
    p.add_argument('-finetune_bert', type=str2bool, default=False)
    p.add_argument('-bert_model_type', type=str, default='bert-base-uncased')
    p.add_argument('-bert_encode_mthd', type=str, choices=['average', 'max', 'head'], default='average')
    p.add_argument('-use_bert', type=str2bool, default=True)
    p.add_argument('-use_glove', type=str2bool, default=True)
    p.add_argument('-bert_dim', type=int, default=1024)
    p.add_argument('-use_pos', type=str2bool, default=True)
    p.add_argument('-regen_vocfile', type=str2bool, default=False)
    p.add_argument('-trainable_emb', type=str2bool, default=False)
    p.add_argument('-trainable_pos_emb', type=str2bool, default=False)
    p.add_argument('-random_seed', type=int, default=1234)
    p.add_argument('-lower', type=str2bool, default=False)
    p.add_argument('-use_crf', type=str2bool, default=True)
    p.add_argument('-use_crf_a', type=str2bool, default=False)
    p.add_argument('-trigger_type', type=str2bool, default=False, help='whether to label trigger label with event types')
    p.add_argument('-k_tri', type=int, default=1, help='beam size for trigger')
    p.add_argument('-decode_w_ents_mask', type=str2bool, default=False)
    p.add_argument('-decode_w_arg_role_mask_by_tri', type=str2bool, default=False)
    p.add_argument('-decode_w_trigger_mask', type=str2bool, default=False)
    p.add_argument('-decode_w_arg_role_mask_by_ent', type=str2bool, default=False)

    # save arugment
    p.add_argument('-model_dir', type=str, default='./saved_models/')
    p.add_argument('-tensorboard_dir', type=str, default='./logs/')
    p.add_argument('-load_model', type=str2bool, default=False)
    p.add_argument('-load_model_path', type=str, default='./saved_models/biaffine_model/best_model.pt')
    args = p.parse_args()
    if args.activation=='None':
        args.activation=None
    torch.manual_seed(args.random_seed)
    main(args)
