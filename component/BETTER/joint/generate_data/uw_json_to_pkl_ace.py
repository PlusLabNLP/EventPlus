import pickle
import json
import pdb
import argparse
import os
from collections import OrderedDict, defaultdict
import sys
#from events.better_core import BetterDocument

# Event types, 33 in total
EVENT_TYPES = ['Business:Declare-Bankruptcy',
               'Business:End-Org',
               'Business:Merge-Org',
               'Business:Start-Org',
               'Conflict:Attack',
               'Conflict:Demonstrate',
               'Contact:Meet',
               'Contact:Phone-Write',
               'Justice:Acquit',
               'Justice:Appeal',
               'Justice:Arrest-Jail',
               'Justice:Charge-Indict',
               'Justice:Convict',
               'Justice:Execute',
               'Justice:Extradite',
               'Justice:Fine',
               'Justice:Pardon',
               'Justice:Release-Parole',
               'Justice:Sentence',
               'Justice:Sue',
               'Justice:Trial-Hearing',
               'Life:Be-Born',
               'Life:Die',
               'Life:Divorce',
               'Life:Injure',
               'Life:Marry',
               'Movement:Transport',
               'Personnel:Elect',
               'Personnel:End-Position',
               'Personnel:Nominate',
               'Personnel:Start-Position',
               'Transaction:Transfer-Money',
               'Transaction:Transfer-Ownership']
# Argument roles: UW data gives 22 roles in total
ARG_ROLES =['Vehicle',
            'Attacker',
            'Prosecutor',
            'Victim',
            'Beneficiary',
            'Entity',
            'Org',
            'Adjudicator',
            'Target',
            'Artifact',
            'Instrument',
            'Giver',
            'Origin',
            'Defendant',
            'Buyer',
            'Agent',
            'Person',
            'Place',
            'Plaintiff',
            'Destination',
            'Seller',
            'Recipient']
# gold Entity types: UW has 7 in total
ENT_TYPES = ['ORG',
             'WEA',
             'VEH',
             'GPE',
             'LOC',
             'FAC',
             'PER']

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
                arg_role = i[2]
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

def construct_full_to_head_dict(gold_ent_mentions):
    '''
    Since ACE annotation still use full ent span as the event arguments, we need a mapping of full span --> head span
    and we want to replace the event arguments with head spans
    -------
    params: `gold_ent_mentions`, the "gold_ent_mentions" item for a sentence
    return: a dict of Mapping[full_span, head_span]
    '''
    full_to_head = {}
    for gold_ent in gold_ent_mentions:
        full_span = (gold_ent['start'], gold_ent['end'] - 1)
        head_span = (gold_ent['head']['start'], gold_ent['head']['end'] - 1)
        if full_span not in full_to_head:
            full_to_head[full_span] = head_span
        else:
            # WARNING seems like quite a lot this case --> many gold entities have more than one type
            # print('Found a duplicated full span for entities.')
            pass


    return full_to_head

def map_arg_idxs_to_head(arg_idxs, full_to_head):

    arg_idxs_head = []
    for arg_idx in arg_idxs:
        full_span = (arg_idx[0], arg_idx[1])
        role = arg_idx[2]
        head_span = full_to_head[full_span] if full_span in full_to_head else full_span
        arg_idxs_head.append((head_span[0], head_span[1], role))

    return arg_idxs_head
def map_time_roles(arg_idxs):
    for i, arg_idx in enumerate(arg_idxs):
        if arg_idx[2].startswith('Time-'):
            arg_idxs[i] = (arg_idxs[i][0], arg_idxs[i][1], 'Time')
    return arg_idxs


def get_data_from_json(json_file, merge_time_roles=True):

    data = []
    with open(json_file, 'rb') as f:
        for line in f:
            json_line = json.loads(line)
            data.append(json_line)

    documents = {}
    data_outs = []
    tri_cnt = 0
    arg_cnt = 0
    multi_tri_cnt = 0
    multi_arg_cnt = 0
    tri_arg_comb = {}   # keep record of valid combinations of trigger_type:arg_role pairs
    ent_arg_comb = defaultdict(list)   # keep record of valid combinations of ent_type:arg_role pairs
    for gold_data in data:
        doc_id = gold_data['doc_key']
        doc_sent_starts = gold_data['sentence_start']
        doc_sents = gold_data['sentences']
        doc_ents = gold_data['ner']
        doc_events = gold_data['events']
        assert len(doc_sent_starts) == len(doc_sents)
        assert len(doc_sent_starts) == len(doc_ents)
        assert len(doc_sent_starts) == len(doc_events)

        data_cnt = 1
        for sent_idx, _ in enumerate(doc_sents):
            sent_id = '{}_{}'.format(doc_id, data_cnt)
            sent_start = doc_sent_starts[sent_idx]
            tokens = doc_sents[sent_idx]
            # pos_tags = gold_data['pos-tags']
            sentence = ' '.join(tokens)
            of_counter = sentence.count('-')
            if float(of_counter)/float(len(sentence)) > 0.7:
                continue
            sent_gold_ents = [(x[0] - sent_start, x[1] - sent_start, x[2]) for x in doc_ents[sent_idx]]
            sent_events = doc_events[sent_idx]


            # gather all events in this sentence
            sent_tri_idxs, sent_arg_idxs = [], []
            sent_tri_arg_pairs = []
            # sent_tri_agent_pairs = []
            # sent_tri_patient_pairs = []
            sent_tri_arg_pairs_type = []
            sent_event_types = []
            sent_ent_to_arg = {(k[0], k[1]):[] for k in sent_gold_ents} if len(sent_gold_ents) > 0 else {}
            for event in sent_events:
                event_type = ':'.join(event[0][-1].split('.'))
                tri_idxs = [(event[0][0] - sent_start, event[0][0] - sent_start)]
                if len(event) > 1:
                    # this event has arguments
                    arg_idxs = [(x[0] - sent_start, x[1] - sent_start, x[2]) for x in event[1:]]
                else:
                    arg_idxs = []
                for argu in arg_idxs:
                    # argu must be ents
                    assert (argu[0], argu[1]) in sent_ent_to_arg, pdb.set_trace()
                    sent_ent_to_arg[(argu[0], argu[1])].append((event[0][0] - sent_start, event[0][0] - sent_start, argu[2]))


                assert len(tri_idxs) == 1   # unlike BETTER, ACE will not have multiple trigger chunks as a event trigger

                # find valid tri--arg_role combinations
                if event_type in tri_arg_comb:
                    for arg in arg_idxs:
                        if arg[2] not in tri_arg_comb[event_type]:
                            tri_arg_comb[event_type].append(arg[2])
                else:
                    tri_arg_comb[event_type] = list(set([arg[2] for arg in arg_idxs]))
                # find valid ent_type--arg_role combinations
                for ent in sent_gold_ents:
                    for argu in arg_idxs:
                        if (argu[0], argu[1]) == (ent[0], ent[1]):
                            ent_arg_comb[ent[2]].append(argu[2])

                tri_cnt += len(tri_idxs)
                arg_cnt += len(arg_idxs)
                multi_tri_cnt += len([x for x in tri_idxs if x[1] - x[0] > 0])
                multi_arg_cnt += len([x for x in arg_idxs if x[1] - x[0] > 0])

                sent_event_types.append(event_type)
                # sent_event_types.extend([event_type] * len(tri_idxs))

                tri_label = get_seq_label_from_idxs(tri_idxs, tokens, 'ANCHOR')
                arg_label = get_seq_label_from_idxs(arg_idxs, tokens, 'ENT')
                sent_tri_arg_pairs.append((tri_label, arg_label))

                tri_label_type = get_seq_label_from_idxs(tri_idxs, tokens, 'TYPE', [event_type])
                sent_tri_arg_pairs_type.append((tri_label_type, arg_label))
                sent_tri_idxs.extend(tri_idxs)
                sent_arg_idxs.extend(arg_idxs)
            sent_tri_idxs_uniq = list(set(sent_tri_idxs))   # there are cases where the sent_tri_idxs has duplicated event idxs
            sent_arg_idxs = list(set([(i[0], i[1]) for i in sent_arg_idxs]))
            sent_tri_label = get_seq_label_from_idxs(sent_tri_idxs_uniq, tokens, 'ANCHOR')
            sent_arg_label = get_seq_label_from_idxs(sent_arg_idxs, tokens, 'ENT')
            sent_ent_label = get_seq_label_from_idxs(sent_gold_ents, tokens, 'ENT')  # record all given entities
            sent_tri_label_type = get_seq_label_from_idxs(sent_tri_idxs, tokens, 'TYPE', sent_event_types)


            data_outs.append({
                'ori_sent': sentence.strip(),
                'sent_id': sent_id,
                'pos_tag': [],
                'tokens': tokens,
                'trigger_label': sent_tri_label,
                'argu_label': sent_arg_label,
                'tri_arg_pairs': sent_tri_arg_pairs,
                'sent_tri_label_type': sent_tri_label_type,
                'sent_tri_arg_pairs_type': sent_tri_arg_pairs_type,
                'raw_events': sent_events,
                'gold_ents': sent_gold_ents,
                'ent_label': sent_ent_label,
                'ent_to_arg': sent_ent_to_arg
            })
            data_cnt += 1

    print('total num of triggers {}, num of multi-token triggers {}'.format(tri_cnt, multi_tri_cnt))
    print('total num of arguments {}, num of multi-token arguments {}'.format(arg_cnt, multi_arg_cnt))

    # de-duplicate
    tri_arg_comb = {k:list(set(v)) for k,v in tri_arg_comb.items()}
    ent_arg_comb = {k:list(set(v)) for k,v in ent_arg_comb.items()}
    # sort the dict by key
    tri_arg_comb = OrderedDict(sorted(tri_arg_comb.items()))
    ent_arg_comb = OrderedDict(sorted(ent_arg_comb.items()))
    # sort each value list of the didct
    tri_arg_comb = {k:sorted(v) for (k,v) in tri_arg_comb.items()}
    ent_arg_comb = {k:sorted(v) for (k,v) in ent_arg_comb.items()}

    combs = {'tri_arg_comb': tri_arg_comb,
             'ent_arg_comb': ent_arg_comb
            }

    return data_outs, combs

def save_pkl(data, out_file):
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)
    print('{} saved.'.format(out_file))

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="""Convert internal JSON to pkl.""")
    p.add_argument('-input_file', type=str, required=False, help="JSON file in internal format, or a dir holding the JSONs")
    p.add_argument('-output_file', type=str, required=False, help="Where to save the pkl file, or a dir to hold the output Pkls")
    p.add_argument('-output_comb_file', type=str, required=False, help="Where to save the tri_arg_comb dict")
    p.add_argument('-merge_combs', action='store_true')
    args = p.parse_args()

    if args.merge_combs:
        train = pickle.load(open('../all_ace/train_uw.comb.pkl', 'rb'))
        dev = pickle.load(open('../all_ace/dev_uw.comb.pkl', 'rb'))
        test = pickle.load(open('../all_ace/test_uw.comb.pkl', 'rb'))
        print('# tri--arg_role combs in train {}'.format(sum([len(v) for k, v in train['tri_arg_comb'].items()])))
        print('# tri--arg_role combs in dev {}'.format(sum([len(v) for k, v in dev['tri_arg_comb'].items()])))
        print('# tri--arg_role combs in test {}'.format(sum([len(v) for k, v in test['tri_arg_comb'].items()])))
        print('# ent--arg_role combs in train {}'.format(sum([len(v) for k, v in train['ent_arg_comb'].items()])))
        print('# ent--arg_role combs in dev {}'.format(sum([len(v) for k, v in dev['ent_arg_comb'].items()])))
        print('# ent--arg_role combs in test {}'.format(sum([len(v) for k, v in test['ent_arg_comb'].items()])))

        # merge all tri--arg_role combs to train
        for tri, argus in dev['tri_arg_comb'].items():
            if tri in train['tri_arg_comb']:
                for arg in argus:
                    if arg not in train['tri_arg_comb'][tri]:
                        train['tri_arg_comb'][tri].append(arg)
            else:
                print('Found non-exist item in dev!!')
        for tri, argus in test['tri_arg_comb'].items():
            if tri in train['tri_arg_comb']:
                for arg in argus:
                    if arg not in train['tri_arg_comb'][tri]:
                        train['tri_arg_comb'][tri].append(arg)
            else:
                print('Found non-exist item in test!!')
        train['tri_arg_comb'] = {k:sorted(v) for (k,v) in train['tri_arg_comb'].items()}
        print('# merged tri--arg_role combs {}'.format(sum([len(v) for k, v in train['tri_arg_comb'].items()])))

        # merge all ent--arg_role combs to train
        for ent, argus in dev['ent_arg_comb'].items():
            if ent in train['ent_arg_comb']:
                for arg in argus:
                    if arg not in train['ent_arg_comb'][ent]:
                        train['ent_arg_comb'][ent].append(arg)
            else:
                print('Found non-exist item in dev!!')
        for ent, argus in test['ent_arg_comb'].items():
            if ent in train['ent_arg_comb']:
                for arg in argus:
                    if arg not in train['ent_arg_comb'][ent]:
                        train['ent_arg_comb'][ent].append(arg)
            else:
                print('Found non-exist item in test!!')
        train['ent_arg_comb'] = {k:sorted(v) for (k,v) in train['ent_arg_comb'].items()}
        print('# merged ent--arg_role combs {}'.format(sum([len(v) for k, v in train['ent_arg_comb'].items()])))

        save_pkl(train, '../all_ace/all_uw.comb.pkl')
        sys.exit()

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
            data = get_data_from_json(os.path.join(args.input_file, json_file), merge_time_roles=True)
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
        data, tri_arg_comb = get_data_from_json(args.input_file, merge_time_roles=True)
        save_pkl(data, args.output_file)
        if args.output_comb_file:
            save_pkl(tri_arg_comb, args.output_comb_file)








