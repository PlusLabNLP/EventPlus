import pickle
import json
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
from neural_model import BertClassifier
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
from seqeval.metrics import f1_score, accuracy_score, classification_report
from generate_data.uw_json_to_pkl_ace import EVENT_TYPES, ARG_ROLES
from eval import *
import pdb
from EventPipeline import EventPipeline
from JsonBuilder import JsonBuilder

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_args(args, path):
    '''
    args: a argparser Namespace
    '''
    with open(path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def load_args(path):
    '''
    return: a argparser Namespace whose values are loaded from
            the file at path
    '''
    with open(path, 'r') as f:
        args_dict = json.load(f)
    args = argparse.Namespace()
    for key, value in args_dict.items():
        setattr(args, key, value)
    return args

class BETTER_API:
    def __init__(self, base_dir = '.', args_path = "saved_args.json"):
        args = load_args(os.path.join(base_dir, args_path))
        args.base_dir = base_dir
        print('BETTER_API.__init__: args loaded')
        print('BETTER API: Number of cuda devices: ', torch.cuda.device_count())

        if torch.cuda.device_count() > 1:
            args.cuda = True

        #### for ACE
        label_t_list = ['<PAD>', 'O']
        for i in EVENT_TYPES:
            label_t_list.append('B-{}'.format(i))
        args._label_to_id_t = OrderedDict([(label_t, i) for i, label_t in enumerate(label_t_list)])

        args._id_to_label_t = OrderedDict([(v, k) for k, v in args._label_to_id_t.items()])

        # construct the B2I_trigger dict for use in `train.py`
        B2I_trigger = {}
        B2I_trigger_string = {}
        for tri_label, tri_id in args._label_to_id_t.items():
            if tri_label.startswith('B-'):
                B2I_trigger[args._label_to_id_t[tri_label]] = args._label_to_id_t['B-{}'.format(tri_label[2:])]
                B2I_trigger_string[tri_label] = 'I-{}'.format(tri_label[2:])
        args.B2I_trigger = B2I_trigger
        args.B2I_trigger_string = B2I_trigger_string


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
        args._id_to_label_e_sent = OrderedDict([(v, k) for k, v in args._label_to_id_e_sent.items()])

        # construct the B2I_arg dict for use in `train.py`
        B2I_ner_string = {}
        for ner_label, ner_id in args._label_to_id_e_sent.items():
            if ner_label.startswith('B-'):
                B2I_ner_string[ner_label] = 'I-{}'.format(ner_label[2:])
        args.B2I_ner_string = B2I_ner_string


        combs = pickle.load(open(os.path.join(args.base_dir, 'generate_data/all_uw.comb.pkl'), 'rb'))
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
        self.system = EventPipeline(args, model, model_t, model_ner)
        self.system.load(filename=os.path.join(args.base_dir, args.load_model_path), filename_t=os.path.join(args.base_dir, args.load_model_path_t), filename_ner=os.path.join(args.base_dir, args.load_model_path_ner))

        self.args = args

    def pred(self, input_sent):
        json_builder = JsonBuilder(self.args.B2I_trigger_string, self.args.B2I_arg_string, self.args.B2I_ner_string)

        y_trues_t, y_preds_t, y_trues_e, y_preds_e, sent_ids_out, y_trues_ner, y_preds_ner =\
        self.system.predict_pipeline(input_sent)
        json_out = json_builder.from_preds(input_sent, y_preds_t, y_preds_e, y_preds_ner)
        return json_out
def main(args):
    api = BETTER_API()
    # input_sent = ['Yesterday', 'New', 'York', 'governor', 'George', 'Pataki', 'toured', 'five', 'counties', 'that', 'have', 'been', 'declared', 'under', 'a', 'state', 'of', 'emergency']
    # input_sent = ["We", "'re", "talking", "about", "possibilities", "of", "full", "scale", "war", "with", "former", "Congressman", "Tom", "Andrews", ",", "Democrat", "of", "Maine", "."]
    # input_sent = ["Orders", "went", "out", "today", "to", "deploy", "17,000", "U.S.", "Army", "soldiers", "in", "the", "Persian", "Gulf", "region", "."]
    # input_sent = ['Yesterday', 'New', 'York', 'governor', ' ',  ' ', 'George', 'Pataki', 'toured', 'five', 'counties', 'that', 'have', 'been', 'declared', 'under', 'a', 'state', 'of', 'emergency']
    # input_sent = ['Brooklyn', 'Beckham', 'asked', 'Nicola', 'Peltz', 'to', 'marry', 'him', ',', 'and', 'she', 'said', 'yes', ',', 'the', 'cameraman', 'and', 'model', 'announced', 'on', 'Saturday', '.', ' ', ' ', 'Beckham', ',', 'whose', 'parents', 'are', 'retired', 'soccer', 'star', 'David', 'Beckham', 'and', 'fashion', 'designer', 'Victoria', 'Beckham', ',', 'popped', 'the', 'big', 'question', 'two', 'weeks', 'ago', ',', 'but', 'they', "’re", 'just', 'letting', 'the', 'world', 'know', 'now', '.']
    # raw_text = 'I like pizza.\nI ate a pizza.      \n I gained weights after one month.'
    raw_text = 'Brooklyn Beckham asked Nicola Peltz to marry him , and she said yes , the cameraman and model announced on Saturday \n.\n \n\n     Beckham , whose parents are retired soccer star David Beckham and fashion designer Victoria Beckham , popped the big question two weeks ago , but they ’re just letting the world know now .'
    # raw_text = ' \n '
    import spacy

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(raw_text)
    input_sent = [x.text for x in doc]
    pdb.set_trace()
    print(api.pred(input_sent))

if __name__ == '__main__':
    p = argparse.ArgumentParser()

    # training argument
    p.add_argument('-batch', type=int, default=2)
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
    p.add_argument('-cuda', type=str2bool, default=False)
    p.add_argument('-multigpu', type=str2bool, default=False)
    p.add_argument('-params', type=dict, default={})
    p.add_argument('-patience', type=int, default=10000)  # disable early stopping
    p.add_argument('-do_train', type=str2bool, default=True)
    p.add_argument('-do_test', type=str2bool, default=False, help='use with load_model. Useful when want to skip training and directly test the loaded model on test')
    p.add_argument('-write_pkl', type=str2bool, default=False, help='whether write out the pkl files for eval')
    p.add_argument('-eval_on_gold_tri', type=str2bool, default=True)
    p.add_argument('-e2e_eval', type=str2bool, default=True, help='when eval_on_gold_tri is True, e2e_eval should be False; when eval_on_gold_tri is False, e2e_eval should be True')
    p.add_argument('-train_on_e2e_data', type=str2bool, default=True, help='when True, train on all predicted trigger pairs, otherwise train on gold+predicted triggers data')
    p.add_argument('-tri_partial_match', type=str2bool, default=True)
    p.add_argument('-use_single_token_tri', type=str2bool, default=True)
    p.add_argument('-gold_ent', type=str2bool, default=False)

    # arguments for RNN model
    p.add_argument('-hid', type=int, default=150)
    p.add_argument('-hid_lastmlp', type=int, default=600)
    p.add_argument('-num_layers', type=int, default=1)
    p.add_argument('-dropout', type=float, default=0.4)
    p.add_argument('-activation', type=str, default='relu',
                   choices=[None, 'tanh', 'relu', 'leakyrelu', 'prelu', 'None'])

    # multi-task argument
    p.add_argument('-ner_weight', type=float, default=1.0)
    p.add_argument('-argument_weight', type=float, default=5.0)
    p.add_argument('-trigger_weight', type=float, default=1.0)

    # hyper-parameter
    p.add_argument('-finetune_bert', type=str2bool, default=True)
    p.add_argument('-bert_model_type', type=str, default='bert-large-uncased')
    p.add_argument('-bert_encode_mthd', type=str, choices=['average', 'max', 'head'], default='head')
    p.add_argument('-use_bert', type=str2bool, default=False)
    p.add_argument('-use_glove', type=str2bool, default=False)
    p.add_argument('-bert_dim', type=int, default=1024)
    p.add_argument('-use_pos', type=str2bool, default=False)
    p.add_argument('-regen_vocfile', type=str2bool, default=False)
    p.add_argument('-trainable_emb', type=str2bool, default=False)
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
    p.add_argument('-k_ner', type=int, default=1, help='beam size for ner')
    p.add_argument('-k_tri', type=int, default=1, help='beam size for trigger')
    p.add_argument('-k_arg', type=int, default=1, help='beam size for argument')
    p.add_argument('-bias_t', type=float, default=1.0, help='bias weight for triggers that are NOT O. For O trigger, weight is 1.0')
    p.add_argument('-bias_a', type=float, default=1.0, help='bias weight for args that are NOT O. For O args, weight is 1.0')
    p.add_argument('-decode_w_ents_mask', type=str2bool, default=True)
    p.add_argument('-decode_w_arg_role_mask_by_tri', type=str2bool, default=True)
    p.add_argument('-decode_w_trigger_mask', type=str2bool, default=True)
    p.add_argument('-decode_w_arg_role_mask_by_ent', type=str2bool, default=False)

    # save arugment
    p.add_argument('-load_model', type=str2bool, default=True)
    p.add_argument('-load_model_path', type=str, default='worked_model_ace/baseline_repro.pt')
    p.add_argument('-load_model_path_t', type=str, default='worked_model_ace/singletrigger_bertlarge2.pt')
    p.add_argument('-load_model_path_ner', type=str, default='worked_model_ace/ner_bertlarge.pt')
    p.add_argument('-load_model_single', type=str2bool, default=True)
    args = p.parse_args()

    # save args to tuple so API class can load args
    save_args(args, 'saved_args.json')

    if args.activation=='None':
        args.activation=None
    torch.manual_seed(args.random_seed)
    main(args)


