import pickle
import json
import argparse
from collections import OrderedDict, defaultdict
import os
import pdb
from events.better_core import BetterDocument, BetterEvent

def get_uniq_id_list(sent_ids):
    '''
    remove duplicated ids in the list of sent_ids, with preserved order
    need this b/c the sent_ids output might be extended when doing event-level argument
    we reduce them to unique ids for alignment to sent-level trigger/arg eval
    '''
    output = list(OrderedDict.fromkeys(sent_ids))
    return output

def get_helper_dict_from_gold_json(gold_data):
    '''
    given liz's processed internal json file, return the helper dict with original sentence (tokens) info
    '''
    helper_dict = defaultdict(dict)
    for doc_id, doc in gold_data.items():
        for sent in doc['sentences']:
            sent_id = sent['sent_id']
            helper_dict[doc_id][sent_id] = sent['words']


    return helper_dict


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
                obj.append(tuple(curr_obj + [i-1]))
                curr_obj = []
                curr_I = None
                in_obj = False
            else:
                if i == len(y) - 1:
                    obj.append(tuple(curr_obj + [i]))
        # beginning of obj
        if y[i] in B2I:
            curr_obj = [y[i][2:], i]
            curr_I = B2I[y[i]]
            in_obj = True
            if i == len(y) - 1:
                obj.append(tuple(curr_obj + [i]))

    return obj


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def construct_event_objs_from_pred(preds, helper_dict, trigger_type, data_type):
    '''
    `preds` contain doc-level prediction
    return: Mapping[int, list of BetterEvent objs]
    '''
    sent_ids = [x['sent_id'] for x in preds]
    sent_ids_uniq = get_uniq_id_list(sent_ids)
    events = {}
    for sent_id in sent_ids_uniq:
        sel_preds = [x for x in preds if x['sent_id'] == sent_id]
        out_dicts = convert_out_dicts_to_event_dicts(sel_preds, helper_dict, trigger_type=trigger_type, data_type=data_type)
        events[int(sent_id.split('_')[1])] = [BetterEvent.from_json(x) for x in out_dicts]
    return events


def convert_out_dicts_to_event_dicts(sel_preds, helper_dict, trigger_type=False, data_type='local'):
    '''
    `sel_preds` contain sent-level prediction
    return a list of dicts, which will be used to create the BetterEvent objs
    `data_type`, currently support choose from ['local', 'ssvm']
    '''
    if trigger_type:
        B2I_trigger ={
                'B-material_helpful': 'I-material_helpful',
                'B-material_harmful': 'I-material_harmful',
                'B-material_neutral': 'I-material_neutral',
                'B-material_unk': 'I-material_unk',
                'B-verbal_helpful': 'I-verbal_helpful',
                'B-verbal_harmful': 'I-verbal_harmful',
                'B-verbal_neutral': 'I-verbal_neutral',
                'B-verbal_unk': 'I-verbal_unk',
                'B-both_helpful': 'I-both_helpful',
                'B-both_harmful': 'I-both_harmful',
                'B-both_neutral': 'I-both_neutral',
                'B-both_unk': 'I-both_unk',
                'B-unk_helpful': 'I-unk_helpful',
                'B-unk_harmful': 'I-unk_harmful',
                'B-unk_neutral': 'I-unk_neutral',
                'B-unk_unk': 'I-unk_unk'}
    else:
        B2I_trigger = {'B-ANCHOR': 'I-ANCHOR'}
    B2I_arg = {'B-AGENT': 'I-AGENT', 'B-PATIENT': 'I-PATIENT'}

    out_dicts = []
    cnt = 1
    if data_type == 'local':
        for event in sel_preds:
            out_dict = {}
            doc_id = event['sent_id'].split('_')[0]
            sent_id = event['sent_id'].split('_')[1]

            out_dict['event_id'] = 'event{}'.format(cnt)
            out_dict['event_type'] = 'abstract'

            tri_seq = event['pred_trigger']
            trigger_objs = iob_to_obj(tri_seq, B2I_trigger)
            if len(trigger_objs) == 0:
                # when predicted trigger is all O, generate an empty span list, assign 'fake' quad class. --> perhaps need to modify later
                print('found doc {}, sent {} with no predicted trigger'.format(doc_id, sent_id))
                # continue
                trigger_span_dicts = []
                out_dict['properties'] = {'helpful-harmful': 'harmful',
                                          'material-verbal': 'material'}
            else:
                assert len(trigger_objs) == 1

                if trigger_type is False:
                    out_dict['properties'] = {'helpful-harmful': 'harmful',
                                              'material-verbal': 'material'}
                else:
                    quad = trigger_objs[0][0]
                    out_dict['properties'] = {'helpful-harmful': quad.split('_')[1],
                                              'material-verbal': quad.split('_')[0]}
                trigger_span_dicts = get_span_dicts_from_objs(trigger_objs, doc_id, sent_id, helper_dict)
            out_dict['anchors'] = {'spans': trigger_span_dicts}

            arg_objs = iob_to_obj(event['pred_arg'], B2I_arg)
            out_dict['arguments'] = []
            for target in ['AGENT', 'PATIENT']:
                arg_span_dicts = []
                a_objs = [x for x in arg_objs if x[0] == target]
                if len(a_objs) == 0:
                    # This is the case where an event does not have agent/patient
                    # the output dict should not have this key --> skip
                    continue
                arg_span_dicts = get_span_dicts_from_objs(a_objs, doc_id, sent_id, helper_dict)
                out_dict['arguments'].append({'role': target.lower(),
                                              'span_set': {'spans': arg_span_dicts}
                    })

            cnt += 1
            out_dicts.append(out_dict)
    elif data_type == 'ssvm':
        assert len(sel_preds) == 1
        event = sel_preds[0]
        for (tri_seq, arg_seq) in event['pred_tri_arg_pair']:
            out_dict = {}
            doc_id = event['sent_id'].split('_')[0]
            sent_id = event['sent_id'].split('_')[1]

            out_dict['event_id'] = 'event{}'.format(cnt)
            out_dict['event_type'] = 'abstract'

            trigger_objs = iob_to_obj(tri_seq, B2I_trigger)
            if len(trigger_objs) == 0:
                # when predicted trigger is all O, generate an empty span list, assign 'fake' quad class. --> perhaps need to modify later
                print('found doc {}, sent {} with no predicted trigger'.format(doc_id, sent_id))
                # continue
                trigger_span_dicts = []
                out_dict['properties'] = {'helpful-harmful': 'harmful',
                                          'material-verbal': 'material'}
            else:
                assert len(trigger_objs) == 1, pdb.set_trace()
                if trigger_type is False:
                    out_dict['properties'] = {'helpful-harmful': 'harmful',
                                              'material-verbal': 'material'}
                else:
                    quad = trigger_objs[0][0]
                    out_dict['properties'] = {'helpful-harmful': quad.split('_')[1],
                                              'material-verbal': quad_split('_')[0]}

                trigger_span_dicts = get_span_dicts_from_objs(trigger_objs, doc_id, sent_id, helper_dict)
            out_dict['anchors'] = {'spans': trigger_span_dicts}

            arg_objs = iob_to_obj(arg_seq, B2I_arg)
            out_dict['arguments'] = []
            for target in ['AGENT', 'PATIENT']:
                arg_span_dicts = []
                a_objs = [x for x in arg_objs if x[0] == target]
                if len(a_objs) == 0:
                    continue
                arg_span_dicts = get_span_dicts_from_objs(a_objs, doc_id, sent_id, helper_dict)
                out_dict['arguments'].append({'role': target.lower(),
                                              'span_set': {'spans': arg_span_dicts}
                    })

            cnt += 1
            out_dicts.append(out_dict)
    else:
        print('Unsupported data type!!')
        assert False
    return out_dicts

        # out_dict['anchors'] = {'spans': [{'text': 'review', 'head_text': 'review', 'grounded_span': {'sent_id': 5, 'full_span': {'text': 'the judicial review', 'start_token': 4, 'end_token': 6}, 'head_span': {'text': 'review', 'start_token': 6, 'end_token': 6}, 'entity_type': None}}]}

        # out_dict['arguments'] = [{'role': 'agent', 'span_set': {'spans': [{'text': 'judicial', 'head_text': 'judicial', 'grounded_span': {'sent_id': 5, 'full_span': {'text': 'judicial', 'start_token': 5, 'end_token': 5}, 'head_span': {'text': 'judicial', 'start_token': 5, 'end_token': 5}, 'entity_type': None}}]}}, {'role': 'patient', 'span_set': {'spans': [{'text': 'decision', 'head_text': 'decision', 'grounded_span': {'sent_id': 5, 'full_span': {'text': 'the decision', 'start_token': 9, 'end_token': 10}, 'head_span': {'text': 'decision', 'start_token': 10, 'end_token': 10}, 'entity_type': None}}]}}]
def get_span_dicts_from_objs(objs, doc_id, sent_id, helper_dict):
    span_dicts = []
    for obj in objs:
        l_idx = obj[1]
        r_idx = obj[2]
        text = helper_dict[doc_id][int(sent_id)][l_idx] if r_idx == l_idx \
                else ' '.join(helper_dict[doc_id][int(sent_id)][l_idx:r_idx+1])
        span_dict = {'text': text,
                     'start_token': l_idx,
                     'end_token': r_idx
                }
        tri_dict = {'text': text,    # TODO what's the difference of `text` and `head_text` here?
                    'head_text': text,
                    'grounded_span': {'sent_id': int(sent_id),
                                      'full_span': span_dict,
                                      'head_span': span_dict,   # set `full_span` and `head_span` identical
                                      'entity_type': None
                        }
                }
        span_dicts.append(tri_dict)
    return span_dicts

def main(args):
    if not os.path.exists(args.output_json_dir):
        os.makedirs(args.output_json_dir)

    gold = json.load(open(args.gold_json, 'rb'))
    helper_dict = get_helper_dict_from_gold_json(gold)

    pred = pickle.load(open(args.pred_pkl, 'rb'))
    pred_docs = {}
    for doc_id, doc in gold.items():
        preds = [x for x in pred if x['sent_id'].split('_')[0] == doc_id]
        golds = BetterDocument.from_json(doc)
        events_pred = construct_event_objs_from_pred(preds, helper_dict, trigger_type=args.trigger_type, data_type=args.data_type)

        doc_out = BetterDocument(doc_id=doc_id,
                                 sentences=golds.sentences,  # set the sentences item with gold sentences,
                                 entities=golds.entities,
                                 abstract_events=events_pred,
                                 basic_events=[])  # set basic_event to be [] b/c we dont have this item
        pred_docs[doc_id] = doc_out.to_dict()

    with open('{}/{}'.format(args.output_json_dir, args.output_json), 'w') as f:
        json.dump(pred_docs, f)
    print('Internal JSON saved at {}/{}'.format(args.output_json_dir, args.output_json))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-pred_pkl', type=str, default='out_pkl/test_ssvm.pkl', help='path to the pickle file to be evaluated')
    p.add_argument('-gold_json', type=str, default='../data_liz/abstract-8d-inclusive.analysis.internal.json', help="path to liz's processed *.arguments.json file")
    p.add_argument('-output_json_dir', type=str, default='sys_json/')
    p.add_argument('-output_json', type=str, default='system.json')
    p.add_argument('-trigger_type', type=str2bool, default=False, help='whether to label quad class type')
    p.add_argument('-data_type', type=str, default='ssvm', choices=['local', 'ssvm'])

    args = p.parse_args()

    main(args)


