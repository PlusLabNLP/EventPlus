import numpy as np
from collections import OrderedDict
from tensorboardX import SummaryWriter
from seqeval.metrics import f1_score, accuracy_score, classification_report
from datetime import datetime
import os
import pickle
from eval import eval_e2e_event_level_arg_id_cls
import torch

def read_glove_dict(glove_dir):
    glove_emb = open(glove_dir, 'r+', encoding="utf-8")
    emb_dict = OrderedDict([(x.strip().split(' ')[0], [float(xx) for xx in x.strip().split(' ')[1:]]) for x in glove_emb])
    return emb_dict

def read_glove_emb(word2idx, glove_dict):
    word_emb = []
    for word in word2idx:
        if word in glove_dict:
            word_emb.append(glove_dict[word])
        elif word == '<PAD>':
            word_emb.append(np.zeros(300))
        else:
            word_emb.append(glove_dict['unk'])

    return np.array(word_emb)

class Logger(object):
    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)
    
    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

def write_result(filename, args, scores):
    if os.path.exists(filename):
        result = pickle.load(open(filename, 'rb'))
    else:
        result = list()
    result.append({
        'score_time': datetime.today().strftime('%Y-%m-%d-%H:%M:%S'),
        'save_dir': args.save_dir,
        'task': args.task,
        'use_bert': args.use_bert,
        'batch_size': args.batch,
        'epoch': args.epochs,
        'lr': args.lr,
        'opt': args.opt,
        'hid': args.hid,
        'n_layers': args.num_layers,
        'dp': args.dropout,
        'act': args.activation,
        'use_crf': args.use_crf,
        'use_att': args.use_att,
        'att_mthd': args.att_mthd,
        'trigger_weight': args.trigger_weight,
        'argument_weight': args.argument_weight,
        'dev_f1': scores['dev_f1'],
        'test_f1_tri': scores['test_f1_t'],
        'test_f1_argu': scores['test_f1_e']
    })
    with open(filename, 'wb') as f:
        pickle.dump(result, f)

def write_result_struct(filename, args, scores):
    if os.path.exists(filename):
        result = pickle.load(open(filename, 'rb'))
    else:
        result = list()
    result.append({
        'score_time': datetime.today().strftime('%Y-%m-%d-%H:%M:%S'),
        'save_dir': args.save_dir,
        'method': args.method,
        'use_bert': args.use_bert,
        'batch_size': args.batch,
        'epoch': args.epochs,
        'lr': args.lr,
        'opt': args.opt,
        'hid': args.hid,
        'n_layers': args.num_layers,
        'dp': args.dropout,
        'act': args.activation,
        'use_crf': args.use_crf,
        'eval_on_gold_tri': args.eval_on_gold_tri,
        'trigger_weight': args.trigger_weight,
        'argument_weight': args.argument_weight,
        'soft_attn': args.soft_attn,
        'query_mthd': args.query_mthd,
        'attn_mthd': args.attn_mthd,
        'att_heads': args.att_heads,
        'att_dropout': args.att_dropout,
        'att_func': args.att_func,
        'use_att_linear_out': args.use_att_linear_out,
        'dev_f1': scores['dev_f1'],
        'test_f1': scores['test_f1'],
    })
    with open(filename, 'wb') as f:
        pickle.dump(result, f)

def get_loss_mlp(lengths, label, pred_logit, criterion):
    # retrieve and flatten prediction for loss calculation
    tri_pred, tri_label = [], []
    for i,l in enumerate(lengths):
        # flatten prediction
        tri_pred.append(pred_logit[i, :l])
        # flatten entity label
        tri_label.append(label[i, :l])
    tri_pred = torch.cat(tri_pred, 0)
    tri_label = torch.cat(tri_label, 0)
    assert tri_pred.size(0) == tri_label.size(0)
    return(criterion(tri_pred, tri_label))

def get_output_rel(pred_logit, input_ref):
    '''
    input_ref: a list of integer, each integer indicate how many output in each batch
    pred_logit: a tensor (# of total events in a batch, num_class)
    
    output: a list of list of prediction(integer)
    '''
    output = list()
    cnt = 0
    for n in input_ref:
        if n != 0:
            output.append(torch.argmax(pred_logit[cnt:cnt+n], dim=1, keepdim=False).tolist())
        else:
            output.append([])
        cnt += n
    return output

def get_loss_rel(gold_rel, pred_logit, criterion):
    '''
    gold_rel = a list of list of prediction(integer)
    pred_logit: a tensor (# of total events in a batch, num_class)
    '''
    # flatten gold_rel
    flatten = pred_logit.new_tensor([x for i in gold_rel for x in i], dtype=torch.long)
    return (criterion(pred_logit, flatten))

def eval_struct_score(y_trues_t, y_preds_t, y_trues_e, y_preds_e, y_pred_paired, sent_ids, test=True):
    # trigger id score:
    assert len(y_trues_t) == len(y_preds_t)
    f1_tri = f1_score(y_trues_t, y_preds_t)
    acc_tri = accuracy_score(y_trues_t, y_preds_t)
    report = classification_report(y_trues_t, y_preds_t)

    # sent-level argument id score:
    assert len(y_trues_e) == len(y_preds_e)
    f1_arg = f1_score(y_trues_e, y_preds_e)
    acc_arg = accuracy_score(y_trues_e, y_preds_e)
    report = classification_report(y_trues_e, y_preds_e)

    # end2end eval
    output_event = []
    for i in range(len(sent_ids)):
        for event in y_pred_paired[i]:
            output_event.append({'sent_id': sent_ids[i], 'pred_trigger': event[0], 'pred_arg': event[1]})
    with open('temp/end2end_event_level_arg_cls.pkl', 'wb') as f:
        pickle.dump(output_event, f)
    print('temp pkl saved, start evaluation...')
    B2I_trigger = {'B-ANCHOR': 'I-ANCHOR'}
    B2I_arg = {'B-AGENT': 'I-AGENT', 'B-PATIENT': 'I-PATIENT'}
    if test:
        prec, recall, f1 = eval_e2e_event_level_arg_id_cls('out_pkl/gold_event_level_tri_arg_cls.pkl',
                                                           'temp/end2end_event_level_arg_cls.pkl',
                                                           B2I_trigger, B2I_arg)
    else:
        prec, recall, f1 = eval_e2e_event_level_arg_id_cls('out_pkl/dev_gold_event_level_tri_arg_cls.pkl',
                                                           'temp/end2end_event_level_arg_cls.pkl',
                                                           B2I_trigger, B2I_arg)

    scores = {
        'f1_tri': f1_tri,
        'acc_tri': acc_tri,
        'f1_arg': f1_arg,
        'acc_arg': acc_arg,
        'precision_e2e': prec,
        'recall_e2e': recall,
        'f1_e2e': f1
    }
    return scores
