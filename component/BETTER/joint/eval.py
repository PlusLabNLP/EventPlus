import pickle
from seqeval.metrics import f1_score, accuracy_score, classification_report
import os
import argparse
import pdb
from collections import OrderedDict

def get_uniq_id_list(sent_ids):
    '''
    remove duplicated ids in the list of sent_ids, with preserved order
    need this b/c the sent_ids output might be extended when doing event-level argument
    we reduce them to unique ids for alignment to sent-level trigger/arg eval
    '''
    output = list(OrderedDict.fromkeys(sent_ids))
    return output
def eval_seqs(pkl_path):
    '''
    support all the evaluation except for end2end eval
    expect sequences like (sent-level) "O O O B-ANCHOR I-ANCHOR O O ..." or
                                       "O O O B-ENT I-ENT O O ..."
    expect sequences like (event-level) "O O O B-ENT I-ENT O O ..."
    expect sequences like (event-level) "O O O B-AGENT I-AGENT O O ..."
    '''
    data = pickle.load(open(pkl_path, 'rb'))
    # for entry in data:
    #     sent_id = entry['sent_id']
    #     gold_seq = entry['gold']
    #     pred_seq = entry['pred']
    y_trues = [i['gold'] for i in data]
    y_preds = [i['pred'] for i in data]
    assert len(y_trues) == len(y_preds)
    # pdb.set_trace()
    f1 = f1_score(y_trues, y_preds)
    acc = accuracy_score(y_trues, y_preds)
    report = classification_report(y_trues, y_preds)

    return f1, acc, report


def iob_to_obj(y, B2I):
    '''
    B2I : {'B-AGENT': 'I-AGENT', 'B-PATIENT': 'I-PATIENT'}
    '''
    obj = []
    in_obj = False
    curr_obj = []
    curr_I = None
    for i in range(len(y)):
        # end of obj
        if in_obj:
            if y[i] != curr_I:
                obj.append(curr_obj + [i-1])
                curr_obj = []
                curr_I = None
                in_obj = False
            else:
                if i == len(y) - 1:
                    obj.append(curr_obj + [i])
        # beginning of obj
        if y[i] in B2I:
            curr_obj = [y[i][2:], i]
            curr_I = B2I[y[i]]
            in_obj = True
            if i == len(y) - 1:
                obj.append(curr_obj + [i])

    return obj


def cal_prec_recall_f1(n_corr, n_pred, n_true):
    prec = (1.0 * n_corr / n_pred) if n_pred!=0 else 0.0
    recall = (1.0 * n_corr / n_true) if n_true!=0 else 0.0
    f1 = 2.0 * prec * recall / (prec+ recall) if (prec + recall != 0) else 0.0

    return prec, recall, f1

def score_match(gold_obj_list, pred_obj_list):
    '''
    e.g.
    gold_obj_list: [('PATIENT', 4, 4), ('PATIENT', 17, 17), ('PATIENT', 26, 26), ('PATIENT', 31, 31), ('AGENT', 36, 38)]
    pred_obj_list: [('PATIENT', 4, 4), ('PATIENT', 17, 17), ('AGENT', 36, 38)]
    the purpose is to perform a sort of soft partial matching of the args for an event when a subset of ALL arguments is correctly identified.
    '''
    n_corr = 0
    # n_pred = len(pred_obj_list)
    # n_true = len(gold_obj_list)
    for pred_obj in pred_obj_list:
        for gold_obj in gold_obj_list:  # strict match
            if pred_obj == gold_obj:
                n_corr += 1
                # found match, go on to next predicted obj
                continue
    return n_corr#, n_pred, n_true

def eval_e2e_event_level_arg_id_cls(gold_pkl_path, pred_pkl_path, B2I_trigger, B2I_arg):
    '''
    expect two pkl files, one is the fixed gold, one is the model prediction. See the format in the pkl files.
    '''
    data_gold = pickle.load(open(gold_pkl_path, 'rb'))
    data_pred = pickle.load(open(pred_pkl_path, 'rb'))
    sent_ids_uniq = list(OrderedDict.fromkeys([i['sent_id'] for i in data_gold]))


    n_corr, n_pred, n_true = 0., 0., 0.
    for sent_id in sent_ids_uniq:
        golds = [i for i in data_gold if i['sent_id'] == sent_id]
        preds = [i for i in data_pred if i['sent_id'] == sent_id]

        gold_triggers = [i['gold_trigger'] for i in golds]
        gold_args = [i['gold_arg'] for i in golds]
        gold_trigger_objs = [iob_to_obj(i, B2I_trigger) for i in gold_triggers]
        gold_arg_objs = [iob_to_obj(i, B2I_arg) for i in gold_args]
        # sort the order of args of each event(trigger), according to the appearance order of text
        gold_arg_objs = [sorted(i, key=lambda x: x[1]) for i in gold_arg_objs]
        n_true += len([x for y in gold_arg_objs for x in y])

        pred_triggers = [i['pred_trigger'] for i in preds]
        pred_args = [i['pred_arg'] for i in preds]
        pred_trigger_objs = [iob_to_obj(i, B2I_trigger) for i in pred_triggers]
        pred_arg_objs = [iob_to_obj(i, B2I_arg) for i in pred_args]
        # sort the order of args of each event(trigger), according to the appearance order of text
        pred_arg_objs = [sorted(i, key=lambda x: x[1]) for i in pred_arg_objs]
        n_pred += len([x for y in pred_arg_objs for x in y])

        # print('gold trigger {}'.format(gold_trigger_objs))
        # print('pred trigger {}'.format(pred_trigger_objs))
        # print('gold arg {}'.format(gold_arg_objs))
        # print('pred arg {}'.format(pred_arg_objs))
        n_match = 0
        for p_idx, p_tri in enumerate(pred_trigger_objs):
            # first, trigger should be correct
            if p_tri == []:
                continue
            if (p_tri[0][1], p_tri[0][2]) in [(i[0][1], i[0][2]) for i in gold_trigger_objs if i != []]:  # strict match for only span, not type
                g_idx = [(i[0][1], i[0][2]) for i in gold_trigger_objs if i != []].index((p_tri[0][1], p_tri[0][2]))
                # g_idx = gold_trigger_objs.index(p_tri)
                gold_args = gold_arg_objs[g_idx]
                pred_args = pred_arg_objs[p_idx]
                # second, retrieve how many argument matches are there given the correctly predicted trigger
                n_match += score_match(gold_args, pred_args)
        n_corr += n_match

    prec, recall, f1 = cal_prec_recall_f1(n_corr, n_pred, n_true)
    return prec, recall, f1

def eval_ace(gold_pkl_path, pred_pkl_path, B2I_trigger, B2I_arg, eval_argu=True):
    '''
    NOTE, the raw_gold_pkl_path is the raw `train_pkl`, while the
          gold_pkl_path is the tmp gold file, pred_pkl_path is the model output prediction
    '''
    # raw_gold = pickle.load(open(raw_gold_pkl_path, 'rb'))
    data_gold = pickle.load(open(gold_pkl_path, 'rb'))
    data_pred = pickle.load(open(pred_pkl_path, 'rb'))
    sent_ids_uniq = list(OrderedDict.fromkeys([i['sent_id'] for i in data_gold]))

    n_corr_t_id, n_corr_t_cls, n_pred_t, n_true_t = 0., 0., 0., 0.
    n_corr_a_id, n_corr_a_cls, n_pred_a, n_true_a = 0., 0., 0., 0.
    for sent_id in sent_ids_uniq:
        golds = [i for i in data_gold if i['sent_id'] == sent_id]
        preds = [i for i in data_pred if i['sent_id'] == sent_id]
        # head_to_full = [i for i in raw_gold if i['sent_id'] == sent_id][0]['head_to_full']

        gold_objs_t, gold_objs_a = [], []
        for gold in golds:
            gold_t = gold['gold_trigger']
            gold_obj_t = iob_to_obj(gold_t, B2I_trigger)  # each item is [t_type, t_l, t_r]
            assert len(gold_obj_t) <= 1
            if len(gold_obj_t) == 0:
                # if no trigger, skip
                continue
            gold_objs_t.extend(gold_obj_t)
            if eval_argu:
                gold_a = gold['gold_arg']
                gold_obj_a = iob_to_obj(gold_a, B2I_arg)
                if len(gold_obj_a) == 0:
                    # if no arguments, do not count, skip
                    continue
                # add trigger info to arg obj, for eval
                gold_obj_a = [x + [gold_obj_t[0][0]] \
                        for x in gold_obj_a]     # each item is [a_type, a_l, a_r, t_type]
                gold_objs_a.extend(gold_obj_a)


        pred_objs_t, pred_objs_a = [], []
        for pred in preds:
            pred_t = pred['pred_trigger']
            pred_obj_t = iob_to_obj(pred_t, B2I_trigger)  # each item is [t_type, t_l, t_r]
            assert len(pred_obj_t) <= 1
            if len(pred_obj_t) == 0:
                continue
            pred_objs_t.extend(pred_obj_t)
            if eval_argu:
                pred_a = pred['pred_arg']
                pred_obj_a = iob_to_obj(pred_a, B2I_arg)
                if len(pred_obj_a) == 0:
                    continue
                # add trigger info to arg obj, for eval
                pred_obj_a = [x + [pred_obj_t[0][0]] \
                        for x in pred_obj_a]     # each item is [a_type, a_l, a_r, t_type]
                pred_objs_a.extend(pred_obj_a)

        n_true_t += len(gold_objs_t)
        n_pred_t += len(pred_objs_t)
        n_true_a += len(gold_objs_a)
        n_pred_a += len(pred_objs_a)
        # pdb.set_trace()
        for x in pred_objs_t:
            # tri cls
            if x in gold_objs_t:
                n_corr_t_cls += 1
            # tri id
            if (x[1], x[2]) in [(y[1], y[2]) for y in gold_objs_t]:
                n_corr_t_id += 1
        if eval_argu:
            for x in pred_objs_a:
                if x in gold_objs_a:
                    # arg cls
                    n_corr_a_cls += 1
                if (x[1], x[2], x[3]) in [(y[1], y[2], y[3]) for y in gold_objs_a]:
                    # arg id
                    n_corr_a_id += 1

    prec_t_id, recall_t_id, f1_t_id = cal_prec_recall_f1(n_corr_t_id, n_pred_t, n_true_t)
    prec_t_cls, recall_t_cls, f1_t_cls = cal_prec_recall_f1(n_corr_t_cls, n_pred_t, n_true_t)
    prec_a_id, recall_a_id, f1_a_id = cal_prec_recall_f1(n_corr_a_id, n_pred_a, n_true_a)
    prec_a_cls, recall_a_cls, f1_a_cls = cal_prec_recall_f1(n_corr_a_cls, n_pred_a, n_true_a)
    print('Trigger Id {:.4f}, {:.4f}, {:.4f}, Trigger Cls {:.4f}, {:.4f}, {:.4f}'.format(prec_t_id, recall_t_id, f1_t_id, prec_t_cls, recall_t_cls, f1_t_cls))
    print('Arg Id {:.4f}, {:.4f}, {:.4f}, Arg Cls {:.4f}, {:.4f}, {:.4f}'.format(prec_a_id, recall_a_id, f1_a_id, prec_a_cls, recall_a_cls, f1_a_cls))
    return [prec_t_id, prec_t_cls, prec_a_id, prec_a_cls], \
           [recall_t_id, recall_t_cls, recall_a_id, recall_a_cls], \
           [f1_t_id, f1_t_cls, f1_a_id, f1_a_cls]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-pkl_dir', type=str, default='./out_pkl')
    p.add_argument('-model_name', type=str, default='multitask_joint_splitargFalse', help='Model name to select the pkl file to be evaluaeted.')
    p.add_argument('-gold_pkl_file_cls', type=str, default='7d_gold_event_level_tri_arg_cls.pkl')
    p.add_argument('-gold_pkl_file_id', type=str, default='7d_gold_event_level_tri_arg_id.pkl')

    args = p.parse_args()
    print(args.model_name)

    if os.path.isfile('{}/{}'.format(args.pkl_dir, args.gold_pkl_file_cls)) and \
       os.path.isfile('{}/{}_e2eTrue/end2end_event_level_arg_cls.pkl'.format(args.pkl_dir, args.model_name)):
        B2I_trigger = {'B-ANCHOR': 'I-ANCHOR'}
        B2I_arg = {'B-AGENT': 'I-AGENT', 'B-PATIENT': 'I-PATIENT'}
        prec, recall, f1 = eval_e2e_event_level_arg_id_cls('{}/{}'.format(args.pkl_dir, args.gold_pkl_file_cls),
                                        '{}/{}_e2eTrue/end2end_event_level_arg_cls.pkl'.format(args.pkl_dir, args.model_name),
                                        B2I_trigger,
                                        B2I_arg)
        print('End2End eval, sent-level argument classification: F1 {:.4f} Prec {:.4f}, Recall {:.4f}'.format(f1, prec, recall))
    if os.path.isfile('{}/{}'.format(args.pkl_dir, args.gold_pkl_file_id)) and \
       os.path.isfile('{}/{}_e2eTrue/end2end_event_level_arg_id.pkl'.format(args.pkl_dir, args.model_name)):
        B2I_trigger = {'B-ANCHOR': 'I-ANCHOR'}
        B2I_arg = {'B-ENT': 'I-ENT'}
        prec, recall, f1 = eval_e2e_event_level_arg_id_cls('{}/{}'.format(args.pkl_dir, args.gold_pkl_file_id),
                                        '{}/{}_e2eTrue/end2end_event_level_arg_id.pkl'.format(args.pkl_dir, args.model_name),
                                        B2I_trigger,
                                        B2I_arg)
        print('End2End eval, sent-level argument identification: F1 {:.4f} Prec {:.4f}, Recall {:.4f}'.format(f1, prec, recall))

    if os.path.isfile('{}/{}_e2eFalse/sent_level_trigger.pkl'.format(args.pkl_dir, args.model_name)):
        f1, acc, report = eval_seqs('{}/{}_e2eFalse/sent_level_trigger.pkl'.format(args.pkl_dir, args.model_name))
        print('sent level trigger F1 {:.4f}, Acc {:.4f}'.format(f1, acc))
        print(report)

    if os.path.isfile('{}/{}_e2eFalse/sent_level_arg_id.pkl'.format(args.pkl_dir,args.model_name)):
        f1, acc, report = eval_seqs('{}/{}_e2eFalse/sent_level_arg_id.pkl'.format(args.pkl_dir, args.model_name))
        print('sent level argument identification F1 {:.4f}, Acc {:.4f}'.format(f1, acc))
        print(report)

    if os.path.isfile('{}/{}_e2eFalse/event_level_arg_id.pkl'.format(args.pkl_dir, args.model_name)):
        f1, acc, report = eval_seqs('{}/{}_e2eFalse/event_level_arg_id.pkl'.format(args.pkl_dir, args.model_name))
        print('event level argument identification F1 {:.4f}, Acc {:.4f}'.format(f1, acc))
        print(report)

    if os.path.isfile('{}/{}_e2eFalse/event_level_arg_cls.pkl'.format(args.pkl_dir, args.model_name)):
        f1, acc, report = eval_seqs('{}/{}_e2eFalse/event_level_arg_cls.pkl'.format(args.pkl_dir, args.model_name))
        print('event level argument classification: F1 {:.4f}, Acc {:.4f}'.format(f1, acc))
        print(report)
