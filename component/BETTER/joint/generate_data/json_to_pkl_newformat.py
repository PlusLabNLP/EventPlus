import pickle
import json
import pdb
import argparse
import os
from events.better_core import BetterDocument


def get_seq_label_from_idxs(idxs, tokens, label_str='ANCHOR', types=None):

    seq_label = ['O'] * len(tokens)

    if label_str == 'ANCHOR':
        # for anchor case, only consider label to be among {'O', 'B-ANCHOR', 'I-ANCHOR'}
        for i in idxs:
            l_idx = i[0]
            r_idx = i[1]
            if r_idx - l_idx == 0:
                # single-token trigger
                seq_label[l_idx] = 'B-{}'.format(label_str)
            elif r_idx - l_idx > 0:
                seq_label[l_idx] = 'B-{}'.format(label_str)
                seq_label[l_idx + 1: r_idx + 1] = ['I-{}'.format(label_str)] * (r_idx - l_idx)
    elif label_str == 'TYPE':
        assert len(idxs) == len(types), pdb.set_trace()
        # for type case, only consider label to be among {'O', 'B-material--helpful', 'I-material--helpful', ...}
        for i, idx in enumerate(idxs):
            l_idx = idx[0]
            r_idx = idx[1]
            if r_idx - l_idx == 0:
                # single-token trigger
                seq_label[l_idx] = 'B-{}'.format(types[i])
            elif r_idx - l_idx > 0:
                seq_label[l_idx] = 'B-{}'.format(types[i])
                seq_label[l_idx + 1: r_idx + 1] = ['I-{}'.format(types[i])] * (r_idx - l_idx)

    elif label_str == 'ENT':
        # for argument case, consider label to be among {'O', 'B-agent', 'I-agent', 'B-patient', 'I-patient'}
        for i in idxs:
            l_idx = i[0]
            r_idx = i[1]
            if len(i) == 3:
                # when arg_role is fed in, use this as label
                arg_role = i[2].upper()
            else:
                # else this is for sent-level arg label, only consider {'O', 'B-ENT', 'I-ENT'}
                arg_role = 'ENT'
            if r_idx - l_idx == 0:
                seq_label[l_idx] = 'B-{}'.format(arg_role)
            elif r_idx - l_idx > 0:
                seq_label[l_idx] = 'B-{}'.format(arg_role)
                seq_label[l_idx + 1: r_idx + 1] = ['I-{}'.format(arg_role)] * (r_idx - l_idx)

    return seq_label


def get_seq_label_fine_grained(idxs, tokens, label_str='AGENT'):
    assert label_str == 'AGENT' or label_str == 'PATIENT'
    seq_label = ['O'] * len(tokens)
    for i in idxs:
        if label_str == 'AGENT':
            if i[2] != 'agent':
                continue
        elif label_str == 'PATIENT':
            if i[2] != 'patient':
                continue
        l_idx = i[0]
        r_idx = i[1]
        if r_idx - l_idx == 0:
            # single-token trigger
            seq_label[l_idx] = 'B-{}'.format(label_str)
        elif r_idx - l_idx > 0:
            seq_label[l_idx] = 'B-{}'.format(label_str)
            seq_label[l_idx + 1: r_idx + 1] = ['I-{}'.format(label_str)] * (r_idx - l_idx)
    return seq_label


def check_span(gold_start, gold_end, c_start, c_end):
    if gold_start > c_start:
        if gold_end <= c_end:
            # candidate contains gold
            return True
        elif gold_end > c_end:
            return False
    elif gold_start == c_start:
        if gold_end >= c_end:
            # gold contains candidate
            return True
        elif gold_end < c_end:
            # candidate contains gold
            return True
    elif gold_start < c_start:
        if gold_end >= c_end:
            # gold contains candidate
            return True
        elif gold_end < c_end:
            return False


def check_duplicate(all_pairs, current_pair):
    # if return True means there's duplicate in all_pairs already
    for p in all_pairs:
        trigger_flag = check_span(p[0], p[1], current_pair[0], current_pair[1])
        argument_flag = check_span(p[2], p[3], current_pair[2], current_pair[3])
        if trigger_flag and argument_flag:
            return True
    return False


def generate_all_candidate_pairs(all_candidates, all_pairs):
    '''
    all_candidates is a list of tuple: (start_idx, end_idx)
    all_pairs is a list of tuple that contains gold trigger argument pairs:
        [(tri_start, tri_end, arg_start, arg_end, label), (....)]

    output: similar structure like all_pairs, but augument with all_candidates
    '''
    output = list()
    for i in range(len(all_candidates)):
        for j in range(len(all_candidates)):
            if i != j:
                current_pair = (all_candidates[i][0], all_candidates[i][1], all_candidates[j][0], all_candidates[j][1])
                if not check_duplicate(all_pairs, current_pair):
                    output.append((all_candidates[i][0], all_candidates[i][1], all_candidates[j][0], all_candidates[j][1], 'None'))
    return output + all_pairs

def get_data_from_json(json_file):

    with open(json_file, 'rb') as f:
        data = json.load(f)
    documents = {}
    data_outs = []
    for doc_id, doc in data.items():
        documents[doc_id] = BetterDocument.from_json(doc)
        for s in documents[doc_id].sentences:
            sent_id = s.sent_id
            sentence = s.text
            tokens = s.words
            pos_tags = s.pos_tags

            # gather all events in this sentence
            sent_events = documents[doc_id].abstract_events[sent_id]
            sent_tri_idxs, sent_arg_idxs = [], []
            sent_tri_arg_pairs = []
            sent_tri_agent_pairs = []
            sent_tri_patient_pairs = []
            sent_tri_arg_pairs_type = []
            sent_event_types = []
            for event in sent_events:
                tri_idxs = [(x.grounded_span.head_span.start_token, x.grounded_span.head_span.end_token) for x in event.anchors.spans]
                arg_idxs = [(x.grounded_span.head_span.start_token, x.grounded_span.head_span.end_token, y.role)
                             for y in event.arguments for x in y.span_set.spans]
                assert len(tri_idxs) > 0, pdb.set_trace()
                type1 = event.properties['material-verbal']
                type2 = event.properties['helpful-harmful']
                # if type1 not in ['material', 'verbal', 'both', 'unk']:
                #     type1 = 'unk'
                # if type2 not in ['helpful', 'harmful', 'neutral']:
                #     type2 = 'unk'
                assert type1 in ['material', 'verbal', 'both', 'unk'], pdb.set_trace()
                assert type2 in ['helpful', 'harmful', 'neutral'], pdb.set_trace()
                event_type = '{}_{}'.format(type1, type2)
                sent_event_types.extend([event_type] * len(tri_idxs))

                tri_label = get_seq_label_from_idxs(tri_idxs, tokens, 'ANCHOR')
                arg_label = get_seq_label_from_idxs(arg_idxs, tokens, 'ENT')
                sent_tri_arg_pairs.append((tri_label, arg_label))

                tri_label_type = get_seq_label_from_idxs(tri_idxs, tokens, 'TYPE', [event_type] * len(tri_idxs))
                sent_tri_arg_pairs_type.append((tri_label_type, arg_label))
                sent_tri_idxs.extend(tri_idxs)
                sent_arg_idxs.extend(arg_idxs)
            sent_tri_idxs_uniq = list(set(sent_tri_idxs))   # there are cases where the sent_tri_idxs has duplicated event idxs
            sent_arg_idxs = list(set([(i[0], i[1]) for i in sent_arg_idxs]))
            sent_tri_label = get_seq_label_from_idxs(sent_tri_idxs_uniq, tokens, 'ANCHOR')
            sent_arg_label = get_seq_label_from_idxs(sent_arg_idxs, tokens, 'ENT')
            sent_tri_label_type = get_seq_label_from_idxs(sent_tri_idxs, tokens, 'TYPE', sent_event_types)

            data_outs.append({
                'ori_sent': sentence.strip(),
                'sent_id': '{}_{}_0'.format(doc_id, sent_id),
                'tokens': tokens,
                'pos_tag': pos_tags,
                'trigger_label': sent_tri_label,
                'argu_label': sent_arg_label,
                'tri_arg_pairs': sent_tri_arg_pairs,
                'tri_agent_pairs': sent_tri_agent_pairs,
                'tri_patient_pairs': sent_tri_patient_pairs,
                'sent_tri_label_type': sent_tri_label_type,
                'sent_tri_arg_pairs_type': sent_tri_arg_pairs_type
            })
    return data_outs

def save_pkl(data, out_file):
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)
    print('{} saved.'.format(out_file))

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="""Convert internal JSON to pkl.""")
    p.add_argument('input_file', type=str, help="JSON file in internal format, or a dir holding the JSONs")
    p.add_argument('output_file', type=str, help="Where to save the pkl file, or a dir to hold the output Pkls")
    args = p.parse_args()

    if os.path.isdir(args.input_file):
        # ensure output dir
        # directory = os.path.dirname(args.output_file)
        # if directory == '':
            # raise OSError('{} is not a dir. Output argument and Input argument should be both dir or both file'.format(args.output_file))
        # assume the output_file is a dir
        if not os.path.exists(args.output_file):
            os.makedirs(args.output_file)
        assert os.path.isdir(args.output_file)

        print('Read JSON files from dir {}'.format(args.input_file))
        for json_file in os.listdir(args.input_file):
            data = get_data_from_json(os.path.join(args.input_file, json_file))
            base_name = os.path.splitext(json_file)[0]
            out_file = os.path.join(args.output_file, '{}.pkl'.format(base_name))
            save_pkl(data, out_file)

    elif os.path.isfile(args.input_file):
        # assume the output_file is a file
        directory = os.path.split(args.output_file)[0]
        # if directory != '':
        #     raise OSError('{} is not a dir. Output argument and Input argument should be both dir or both file'.format(args.output_file))
        if directory != '':
            if not os.path.exists(directory):
                os.makedirs(directory)

        print('Read JSON file from file {}'.format(args.input_file))
        data = get_data_from_json(args.input_file)
        save_pkl(data, args.output_file)

