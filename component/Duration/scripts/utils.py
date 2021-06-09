import logging
import json
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import nltk
from itertools import combinations
from predpatt import load_conllu
from predpatt import PredPatt
from predpatt import PredPattOpts

# from factslab.datastructures import ConstituencyTree, DependencyTree
from .src.factslab.factslab.pytorch.temporalmodule import TemporalModel, TemporalTrainer

options = PredPattOpts(resolve_relcl=True, borrow_arg_for_relcl=True, resolve_conj=False, cut=True)

import allennlp
import torch
import torch.nn.functional as F
# import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from allennlp.commands.elmo import ElmoEmbedder
import pickle
from torch.distributions.binomial import Binomial
from torch.nn import MSELoss, L1Loss, SmoothL1Loss, CrossEntropyLoss

import torch
from torch import nn
# from torchviz import make_dot, make_dot_from_trace

from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm_n

options_file = "elmo_files/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "elmo_files/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

from nltk import DependencyGraph
import re


def html_ify(s):
    '''
        Takes care of &quot &lsqb &rsqb &#39
    '''
    html_string = re.sub(r'\)', r'&rcrb;', s)
    html_string = re.sub(r'\(', r'&lcrb;', html_string)
    return html_string


def get_structs(file_path):
    files = [file_path]
    structures = {}
    for file in files:
        with open(file, 'r') as f:
            filename = file.split("/")[-1]
            iden = 0
            a = ""
            words = []
            for line in f:
                if line != "\n":
                    a += line
                    words.append(line.split("\t")[1])
                else:
                    iden += 1
                    a = html_ify(a)
                    structure = DependencyGraph(a, top_relation_label='root')
                    sent = " ".join(words)
                    sent = html_ify(sent)
                    sent_id = filename + " sent_" + str(iden)
                    structures[sent_id] = structure
                    a = ""
                    words = []
    return structures


def extract_struct_dicts(structures):
    '''
    Input: A dictionary of DependencyGraph objects with key as the sentence id:
            Key example: sample_corpus_document.txt.output sent_1'
            
    Output: A dictionary of sentences with key as the sentence id:

    '''
    struct_dict = {}
    for key in structures:
        N = len(structures[key].nodes)
        struct_dict[key] = [structures[key].nodes[i]['word'] for i in range(1, N)]

    return struct_dict


def depth_in_tree(idx, dep_obj):
    '''
    Input: Index of the word in a linear sequence of words
    
    Output: Depth of that word in the dependency tree
    
    '''
    nodes = dep_obj.nodes
    depth = 0
    i = idx + 1
    while nodes[i]['rel'] != 'root':
        i = nodes[i]['head']
        depth += 1

    return depth


def find_pivot_predicate(fname, sentid_num, predp_object, structures):
    '''
    Find the pivot-predicate of a given sentence's id
    
    Heuristic/Algo:  Follow the root predicate until you find a predicate which doesn't have
                any xcomp, ccomp or csubj dependencies.
                
    '''
    # preds = filter_preds([(sentid_num, x) for x in predp_object.instances])
    preds = [(sentid_num, x) for x in predp_object.instances]
    tokens = [y.root.position for x, y in preds]

    if tokens:
        tokens_covered = set()

        struct_id = fname + " sent_" + str(sentid_num)
        dep_object = structures[struct_id]
        pred_heights = sorted([(x, depth_in_tree(x, dep_object)) for x in tokens], key=lambda x: x[1])
        tokens_reverse = [x for x, y in pred_heights][::-1]

        root_idx = tokens.index(pred_heights[0][0])
        root_predicate = preds[root_idx]
        deps = dep_object.nodes[tokens[root_idx] + 1]['deps']

        tokens_covered.add(tokens[root_idx])
        tokens_reverse.pop()

        while ('ccomp' in deps) or ('xcomp' in deps) or ('csubj') in deps:
            variables = ['ccomp', 'xcomp', 'csubj']
            for var in variables:
                if var in deps:
                    tok_idx = deps[var][0] - 1
                    if tok_idx in tokens:
                        root_idx = tokens.index(tok_idx)
                        tokens_covered.add(tokens[root_idx])
                        tokens_reverse.pop()
                    else:
                        if tokens_reverse:
                            root_idx = tokens.index(tokens_reverse[-1])
                            tokens_covered.add(tokens[root_idx])
                            tokens_reverse.pop()
                        else:
                            return root_predicate
                    break

            deps = dep_object.nodes[tokens[root_idx] + 1]['deps']
            root_predicate = preds[root_idx]

        return root_predicate

    return []


def filter_preds(pred_tuples):
    '''
    Input: a list of tuples of (sent_id_num, predicate object)
    
    Output: filter tuples only with specific pos tags predicates
    
    '''
    ans = []
    pos_tags = set(["ADJ", "NOUN", "NUM", "DET", "PROPN", "PRON", "VERB", "AUX"])
    for sent_id, pred_obj in pred_tuples:
        if pred_obj.root.tag not in pos_tags:
            # print(pred_obj.root.tag)
            # print("not in pos tags")
            continue
        elif pred_obj.root.tag not in ["VERB", "AUX"]:
            gov_rels = [tok.gov_rel for tok in pred_obj.tokens]
            if 'cop' in gov_rels:
                ans.append((sent_id, pred_obj))
            elif pred_obj.root.tag == 'ADJ':
                ans.append((sent_id, pred_obj))
        else:
            ans.append((sent_id, pred_obj))
    return ans


def predicate_info(predicate):
    '''
    Input: predicate object
    Output: pred_text, token, root_token
    
    Note: If predicate is copular: pred_text is only upto first 5 words
    '''
    copula_bool = False

    # Extend predicate to start from the copula
    if predicate.root.tag not in ["VERB", "AUX"]:
        all_pred = predicate.tokens
        gov_rels = [tok.gov_rel for tok in all_pred]
        if 'cop' in gov_rels:
            copula_bool = True
            cop_pos = gov_rels.index('cop')
            pred = [x.text for x in all_pred[cop_pos:]]
            pred_token = [x.position for x in all_pred[cop_pos:]]
            def_pred_token = predicate.root.position  # needed for it_happen set
            cop_bool = True
            # print(predicate, idx)

        elif predicate.root.tag == "ADJ":
            pred_token = [predicate.root.position]
            pred = [predicate.root.text]
            def_pred_token = predicate.root.position
        else:  ## Different from protocol as we are considering all predicates
            pred_token = [predicate.root.position]
            pred = [predicate.root.text]
            def_pred_token = predicate.root.position

    # Else keep the root
    else:
        pred_token = [predicate.root.position]
        pred = [predicate.root.text]
        def_pred_token = predicate.root.position

        # Stringify pred and pred_tokens:
    # pred_token = "_".join(map(str, pred_token))

    if len(pred) > 5:
        pred = pred[:5]
        pred = " ".join(pred) + "..."
    else:
        pred = " ".join(pred)

    return pred, pred_token, def_pred_token


def dict_pred_double(pred_comb, raw_sentence, fname, sentid_num, sentid_num_next):
    '''
    Extract turk_parse dict from input predicate combination 
    
    INputs:
    1. pred_all : one list of all predicates in both sentences
    2. raw_sentence: a dict of two sentences, with key: sent_id_num
    3. sentid_num: 1st sentence in adjacent sentence
    4. sentid_num_next: 2nd sentence in adjacent sentence
    
    '''
    token_dict = {}
    pred1_obj, pred2_obj = [y for x, y in pred_comb]
    sent_id1, sent_id2 = [x for x, y in pred_comb]

    pred1_text, pred1_token, pred1_root_token = predicate_info(pred1_obj)
    pred2_text, pred2_token, pred2_root_token = predicate_info(pred2_obj)

    token_dict['pred1_token'] = "_".join(map(str, pred1_token))
    token_dict['pred1_text'] = pred1_text
    token_dict['pred2_token'] = "_".join(map(str, pred2_token))
    token_dict['pred2_text'] = pred2_text
    token_dict['sentence_id_1'] = fname + " " + sent_id1
    token_dict['sentence_id_2'] = fname + " " + sent_id2
    token_dict['pred1_root_token'] = pred1_root_token
    token_dict['pred2_root_token'] = pred2_root_token

    pred_sentence = raw_sentence[sentid_num] + raw_sentence[sentid_num_next]
    token_dict['sentence'] = " ".join(pred_sentence)

    return token_dict, pred1_token, pred2_token


def extract_dataframe(file_path, structures):
    '''
    Input: Input document file path which contains conllu format 
            sentences separated by '\n'
    
    Output: A dataframe after processing the file through PredPatt and exracting
             roots and spans of each predicate. 
             Each row in the dataframe corresponds to an event-pair
    '''

    with open(file_path) as infile:
        data = infile.read()
        parsed = [(PredPatt(ud_parse, opts=options), sent_id) for sent_id, ud_parse in load_conllu(data)]
        print("Number of sentences in the document: {}".format(len(parsed)))

    fname = file_path.split("/")[-1]

    total_preds = 0
    global_tuples = []
    sent_total = 0

    total_sents_doc = len(parsed)
    for i, parse_sen in enumerate(parsed):
        pred_object = parse_sen[0]
        total_preds += len(pred_object.instances)
        sentid_num = parse_sen[1].split("_")[-1]
        # print(sentid_num)

        ## Concatenate adjacent sentences
        if i < total_sents_doc - 1:
            parse_sen_next = parsed[i + 1]
            pred_object_next = parse_sen_next[0]
            sentid_num_next = parse_sen_next[1].split("_")[-1]

            raw_sentence = {sentid_num: [token.text for token in pred_object.tokens],
                            sentid_num_next: [token.text for token in pred_object_next.tokens]}

            preds_curr = [(sentid_num, pred) for pred in pred_object.instances]
            preds_next = [(sentid_num_next, pred) for pred in pred_object_next.instances]

            # Curr_sent combinations (all possible)
            pred_combs_curr = combinations(preds_curr, 2)
            for pred_comb in pred_combs_curr:
                # token dict from all predicates in the antecedent sentence:
                token_dict, pred_token1, pred_token2 = dict_pred_double(pred_comb, raw_sentence,
                                                                        fname, sentid_num,
                                                                        sentid_num_next)
                global_tuples.append((token_dict, pred_token1, pred_token2))
                sent_total += 1

            # Combinations of Pivot predicate of curr_sent with predicates of next_sent:
            pivot_curr_pred = find_pivot_predicate(fname, sentid_num, pred_object, structures)
            # print("Pivot predicate: {}".format(pivot_curr_pred))
            if pivot_curr_pred:
                for tupl in preds_next:
                    pred_comb = [pivot_curr_pred, tupl]
                    token_dict, pred_token1, pred_token2 = dict_pred_double(pred_comb, raw_sentence,
                                                                            fname, sentid_num,
                                                                            sentid_num_next)
                    global_tuples.append((token_dict, pred_token1, pred_token2))
                    sent_total += 1

    ## Create a dataframe from the global tuples dictionary
    dcts = [dct for dct, pred1_span, pred2_span in global_tuples]
    pred1_spans = [pred1_span for dct, pred1_span, pred2_span in global_tuples]
    pred2_spans = [pred2_span for dct, pred1_span, pred2_span in global_tuples]

    df = pd.DataFrame(dcts)
    # df['pred1_span'] = np.array(pred1_spans)
    # df['pred2_span'] = np.array(pred2_spans)
    df['pred1_span'] = pred1_spans
    df['pred2_span'] = pred2_spans

    return df


def correct_pred2_root(row, struct_dict):
    if row.sentence_id_1 == row.sentence_id_2:
        return row.pred2_root_token
    else:
        sent_str, num = row.sentence_id_1.split(" ")
        sent_name = sent_str + " " + "sent_" + num

        return len(struct_dict[sent_name]) + row.pred2_root_token


def correct_pred2_tokens(row, struct_dict):
    if row.sentence_id_1 == row.sentence_id_2:
        return row.pred2_token
    else:
        sent_str, num = row.sentence_id_1.split(" ")
        sent_name = sent_str + " " + "sent_" + num

        curr_posns = [int(x) for x in row.pred2_token.split("_")]
        new_posns = [len(struct_dict[sent_name]) + x for x in curr_posns]

        return "_".join([str(x) for x in new_posns])


def extract_X(data):
    sents = data.sentence.values
    structures = [x.split() for x in sents]
    root_idxs = data[['pred1_root_token', 'pred2_root_token_mod']].values
    span_idxs = data[['pred1_token_span', 'pred2_token_span']].values

    X_data = list(zip(structures, span_idxs, root_idxs))
    print("Number of event pairs considered: {}".format(len(X_data)))

    return X_data


def predict_fine_dur_only(data_x, model, predict_batch_size=80):
    '''
    Predict duration and fine-grained relations
    '''
    # Turn on evaluation mode which disables dropout.
    model.eval()

    with torch.no_grad():
        bidx_i = 0
        bidx_j = predict_batch_size
        total_obs = len(data_x)
        p1_dur_yhat = torch.zeros(total_obs, 11).to(model.device)
        p2_dur_yhat = torch.zeros(total_obs, 11).to(model.device)
        # coarse_yhat = torch.zeros(total_obs, 13).to(model.device)
        # coarser_yhat = torch.zeros(total_obs, 7).to(model.device)
        fine_yhat = torch.zeros(total_obs, 4).to(model.device)
        rel_yhat = torch.zeros(total_obs, 1280).to(model.device)

        while bidx_j < total_obs:
            words = [p for p, q, r in data_x[bidx_i:bidx_j]]
            spans = [q for p, q, r in data_x[bidx_i:bidx_j]]
            roots = [r for p, q, r in data_x[bidx_i:bidx_j]]
            predicts = model(words, spans, roots)
            # print(predicts[0].size())
            # print(p1_dur_yhat[bidx_i:bidx_j].size())
            # print("\n")
            p1_dur_yhat[bidx_i:bidx_j] = predicts[0]
            p2_dur_yhat[bidx_i:bidx_j] = predicts[1]
            # coarse_yhat[bidx_i:bidx_j] = predicts[3]
            # coarser_yhat[bidx_i:bidx_j] = predicts[4]
            fine_yhat[bidx_i:bidx_j] = predicts[2]
            rel_yhat[bidx_i:bidx_j] = predicts[3]

            bidx_i = bidx_j
            bidx_j = bidx_i + predict_batch_size

            if bidx_j >= total_obs:
                words = [p for p, q, r in data_x[bidx_i:bidx_j]]
                spans = [q for p, q, r in data_x[bidx_i:bidx_j]]
                roots = [r for p, q, r in data_x[bidx_i:bidx_j]]
                predicts = model(words, spans, roots)
                p1_dur_yhat[bidx_i:bidx_j] = predicts[0]
                p2_dur_yhat[bidx_i:bidx_j] = predicts[1]
                # coarse_yhat[bidx_i:bidx_j] = predicts[3]
                # coarser_yhat[bidx_i:bidx_j] = predicts[4]
                fine_yhat[bidx_i:bidx_j] = predicts[2]
                rel_yhat[bidx_i:bidx_j] = predicts[3]

        p1_dur_yhat = F.softmax(p1_dur_yhat, dim=1)
        p2_dur_yhat = F.softmax(p2_dur_yhat, dim=1)
        # coarse_yhat = F.softmax(coarse_yhat, dim=1)
        # coarser_yhat = F.softmax(coarser_yhat, dim=1)

        _, p1_dur_yhat = p1_dur_yhat.max(1)
        _, p2_dur_yhat = p2_dur_yhat.max(1)
        # _ , coarse_yhat =  coarse_yhat.max(1)
        # _ , coarser_yhat =  coarser_yhat.max(1)

    return p1_dur_yhat.detach(), p2_dur_yhat.detach(), fine_yhat.detach(), rel_yhat.detach()


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def extract_preds(data):
    '''
    Extracts a dict of predicates for a given docid data
    Key: pred_sent_id
    Value: predicate-index
    '''
    cols = ['sent_pred_id1', 'sent_pred_id2', 'b1', 'e1', 'b2', 'e2',
            'pred1_duration', 'pred2_duration',
            'pred1_text', 'pred2_text']

    local_data = data[cols]
    preds_arr = local_data[['sent_pred_id1', 'sent_pred_id2']].values
    uniq_preds = np.unique(preds_arr.flatten())

    pred_dict = {}
    idx = 0
    for pred in uniq_preds:
        pred_dict[pred] = idx
        idx += 1

    local_data['pred1_dict_idx'] = local_data['sent_pred_id1'].map(lambda x: pred_dict[x])
    local_data['pred2_dict_idx'] = local_data['sent_pred_id2'].map(lambda x: pred_dict[x])

    return pred_dict, idx, local_data


def extract_pred_text(lst, data):
    '''
    Input: A list of sent_pred tokens
    Output: A list of predicate text
    '''
    ans = []
    for sent_pred in lst:
        try:
            pred_text = data[(data.sent_pred_id1 == sent_pred)]['pred1_text'].values[0]
            ans.append(pred_text)
        except:
            pred_text = data[(data.sent_pred_id2 == sent_pred)]['pred2_text'].values[0]
            ans.append(pred_text)

    return ans
