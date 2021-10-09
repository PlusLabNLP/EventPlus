import pickle
import os
import argparse
from collections import Counter, OrderedDict
from itertools import combinations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import copy
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
#from featurize_data import matres_label_map, tbd_label_map
from transformers import BertTokenizer, BertModel
import spacy


tbd_label_map = OrderedDict([('VAGUE', 'VAGUE'),
                             ('BEFORE', 'BEFORE'),
                             ('AFTER', 'AFTER'),
                             ('SIMULTANEOUS', 'SIMULTANEOUS'),
                             ('INCLUDES', 'INCLUDES'),
                             ('IS_INCLUDED', 'IS_INCLUDED'),
                         ])

matres_label_map = OrderedDict([('VAGUE', 'VAGUE'),
                             ('BEFORE', 'BEFORE'),
                             ('AFTER', 'AFTER'),
                             ('SIMULTANEOUS', 'SIMULTANEOUS')
                         ])

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(123)

class Event():
    def __init__(self, id, type, text, tense, polarity, span):
        self.id = id
        self.type = type
        self.text = text
        self.tense = tense
        self.polarity = polarity
        self.span = span

        
def pad_collate(batch):
    """Puts data, and lengths into a packed_padded_sequence then returns                           
    the packed_padded_sequence and the labels. Set use_lengths to True
    to use this collate function.                                                                                                                         
    Args:                                                                                                                                                 
        batch: (list of tuples) [(doc_id, sample_id, pair, label, sent, pos, fts, rev, lidx_start_s, lidx_end_s, ridx_start_s, ridx_end_s, pred_ind)].                                

    Output:                                                                                                                                               
        packed_batch: (PackedSequence for sent and pos), see torch.nn.utils.rnn.pack_padded_sequence                                                                         
        labels: (Tensor)
    
        other arguments remain the same.                                                                             
        """
    if len(batch) >= 1:

        bs  = list(zip(*[ex for ex in sorted(batch, key=lambda x: x[2].shape[0], reverse=True)]))
        
        max_len, n_fts = bs[2][0].shape
        lengths = [x.shape[0] for x in bs[2]]
        
        ### gather sents: idx = 2 in batch_sorted
        sents = [torch.cat((torch.FloatTensor(s), torch.zeros(max_len - s.shape[0], n_fts)), 0) 
                 if s.shape[0] != max_len else torch.FloatTensor(s) for s in bs[2]]
        sents = torch.stack(sents, 0)

        # gather entity labels: idx = 3 in batch_sorted                                                                      
        # we need a unique doc_span key for aggregation later
        all_key_ent = [list(zip(*key_ent)) for key_ent in bs[3]]

        keys = [[(bs[0][i], k) for k in v[0]] for i, v in enumerate(all_key_ent)]

        ents = [v[1] for v in all_key_ent]
        ents = [torch.cat((torch.LongTensor(s).unsqueeze(1), torch.zeros(max_len - len(s), 1, dtype=torch.long)), 0)
                if len(s) != max_len else torch.LongTensor(s).unsqueeze(1) for s in ents]
        ents = torch.stack(ents, 0).squeeze(2)
        
        # gather pos tags: idx = 6 in batch_sorted; treat pad as 0 -- this needs to be fixed !!! 
        #poss = [torch.cat((s.unsqueeze(1), torch.zeros(max_len - s.size(0), 1, dtype=torch.long)), 0) 
        #        if s.size(0) != max_len else s.unsqueeze(1) for s in bs[4]]
        #poss = torch.stack(poss, 0)
        
    return bs[0], bs[1], sents, keys, ents, bs[4], bs[5], lengths


class EventDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_dir, data_split):
        'Initialization'
        # load data
        with open(data_dir + data_split + '.pickle', 'rb') as handle:
            self.data = pickle.load(handle)
            self.data = list(self.data.values())
        handle.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, idx):
        'Generates one sample of data'
        
        sample = self.data[idx]
        
        doc_id = sample['doc_id']
        context_id = sample['context_id']
        context = sample['context']
        rels = sample['rels']

        return doc_id, context_id, context[0], context[1], context[2],  rels

class BertClassifier(nn.Module):
    'Neural Network Architecture'
    def __init__(self, args):
        
        super(BertClassifier, self).__init__()
        
        self.hid_size = args.hid
        self.batch_size = args.batch
        self.num_layers = args.num_layers
        self.num_classes = len(args.label_to_id)
        self.num_ent_classes = 2

        self.dropout = nn.Dropout(p=args.dropout)
        # lstm is shared for both relation and entity
        self.lstm = nn.LSTM(768, self.hid_size, self.num_layers, bias = False, bidirectional=True)

        # MLP classifier for relation
        self.linear1 = nn.Linear(self.hid_size*4+args.n_fts, self.hid_size)
        self.linear2 = nn.Linear(self.hid_size, self.num_classes)

        # MLP classifier for entity
        self.linear1_ent = nn.Linear(self.hid_size*2, int(self.hid_size / 2))
        self.linear2_ent = nn.Linear(int(self.hid_size / 2), self.num_ent_classes)

        self.act = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.softmax_ent = nn.Softmax(dim=2)

    def forward(self, sents, lengths, fts = [], rel_idxs=[], lidx_start=[], lidx_end=[], ridx_start=[], 
                ridx_end=[], pred_ind=True, flip=False, causal=False, token_type_ids=None, task='relation'):

        batch_size = sents.size(0)
        # dropout
        out = self.dropout(sents)
        # pack and lstm layer
        out, _ = self.lstm(pack(out, lengths, batch_first=True))
        # unpack
        out, _ = unpack(out, batch_first = True)

        ### entity prediction - predict each input token 
        if task == 'entity':
            out_ent = self.linear1_ent(self.dropout(out))
            out_ent = self.act(out_ent)
            out_ent = self.linear2_ent(out_ent)
            prob_ent = self.softmax_ent(out_ent)
            return out_ent, prob_ent
        
        ### relaiton prediction - flatten hidden vars into a long vector
        if task == 'relation':
            
            ltar_f = torch.cat([out[b, lidx_start[b][r], :self.hid_size].unsqueeze(0) for b,r in rel_idxs], dim=0)
            ltar_b = torch.cat([out[b, lidx_end[b][r], self.hid_size:].unsqueeze(0) for b,r in rel_idxs], dim=0)
            rtar_f = torch.cat([out[b, ridx_start[b][r], :self.hid_size].unsqueeze(0) for b,r in rel_idxs], dim=0)
            rtar_b = torch.cat([out[b, ridx_end[b][r], self.hid_size:].unsqueeze(0) for b,r in rel_idxs], dim=0)
        
            out = self.dropout(torch.cat((ltar_f, ltar_b, rtar_f, rtar_b), dim=1))
            out = torch.cat((out, fts), dim=1)
            
            # linear prediction                                                                                      
            out = self.linear1(out)
            out = self.act(out)
            out = self.dropout(out)
            out = self.linear2(out)
            prob = self.softmax(out)            
            return out, prob

def get_pos_tag_idx(file_dir):
    tags = open(file_dir + "/pos_tags.txt")
    pos2idx = {}
    idx = 0
    for tag in tags:
        tag = tag.strip()
        pos2idx[tag] = idx
        idx += 1
    return pos2idx

def featurize_raw_data(sent, tokenizer, bert_model, args):
    doc_id = ('input1', )
    context_id = (0, )

    nlp = spacy.load("en_core_web_sm")
    sent = nlp(sent)
    orig_tokens = {tok.text:i for i, tok in enumerate(sent)}

    pos2idx = get_pos_tag_idx(args.other_dir)
    pos = [pos2idx[k.tag_] if k.tag_ in pos2idx.keys() else len(pos2idx) for k in sent]
    ent = [(doc_id[0], i) for i, _ in enumerate(sent)]

    # bert sentence segment ids
    segments_ids = []  # [0, ..., 0, 0, 1, 1, ...., 1]
    seg = 0
    bert_pos = []
    bert_ent = []

    # append sentence start
    bert_tokens = ["[CLS]"]
    # original token to bert word-piece token mapping
    orig_to_tok_map = []

    segments_ids.append(seg)
    bert_pos.append("[CLS]")

    # sent_start is non-event by default
    bert_ent.append((doc_id[0], "[CLS]"))

    sent = [tok.text.lower() for tok in sent]
    for i, token in enumerate(sent):
        orig_to_tok_map.append(len(bert_tokens))

        temp_tokens = tokenizer.tokenize(token)
        bert_tokens.extend(temp_tokens)
        for _ in temp_tokens:
            segments_ids.append(seg)
            bert_pos.append(pos[i])
            bert_ent.append(ent[i])

    orig_to_tok_map.append(len(bert_tokens))

    bert_tokens.append("[SEP]")
    bert_pos.append("[SEP]")
    bert_ent.append(('[SEP]', -1))

    ent_labels = torch.tensor([0]*len(bert_ent)).reshape(1, -1)

    segments_ids.append(seg)
    assert len(segments_ids) == len(bert_tokens)
    assert len(bert_pos) == len(bert_tokens)

    bert_sent = tokenizer.convert_tokens_to_ids(bert_tokens)

    bert_sent = torch.tensor([bert_sent])
    segs_sent = torch.tensor([segments_ids])

    out, _ = bert_model(bert_sent, segs_sent)
    sent = out[-1].reshape(1, -1, out.size()[-1])
    return doc_id, context_id, sent, [bert_ent], ent_labels, (bert_pos, ), [], [sent.size()[1]], orig_tokens

@dataclass()
class NNClassifier(nn.Module):
    def __init__(self):
        super(NNClassifier, self).__init__()

    def predict(self, model, featurized_data, args, test=False, gold=True, model_r=None):

        model.eval()
        
        criterion = nn.CrossEntropyLoss()                                                                           

        labels, probs, losses_t, losses_e = [], [], [], []
        pred_inds, docs, pairs, event_heads = [], [], [], []

        # stoare non-predicted rels in list
        nopred_rels = []

        ent_pred_map, ent_label_map = {}, {}

        doc_id, context_id, sents, ent_keys, ents, poss, rels, lengths, _ = featurized_data
        if args.cuda:
            sents = sents.cuda()
            ents = ents.cuda()

        ## predict entity first
        out_e, prob_e = model(sents, lengths, task='entity')

        labels_r, fts, rel_idxs, doc, pair, lidx_start, lidx_end, ridx_start, ridx_end, nopred_rel \
            = self.construct_relations(prob_e, lengths, rels, list(doc_id), poss, gold=gold)

        nopred_rels.extend(nopred_rel)
        ### predict relations
        if rel_idxs: # predicted relation could be empty --> skip
            docs.extend(doc)
            pairs.extend(pair)
            for li, lidx in enumerate(lidx_start):
                event_heads.extend(list(zip(lidx, ridx_start[li])))

            if args.cuda:
                labels_r = labels_r.cuda()
                fts = fts.cuda()

            if model_r:
                model_r.eval()
                out_r, prob_r = model_r(sents, lengths, fts=fts, rel_idxs=rel_idxs, lidx_start=lidx_start,
                                        lidx_end=lidx_end, ridx_start=ridx_start, ridx_end=ridx_end)
            else:
                out_r, prob_r = model(sents, lengths, fts=fts, rel_idxs=rel_idxs, lidx_start=lidx_start,
                                      lidx_end=lidx_end, ridx_start=ridx_start, ridx_end=ridx_end)
            loss_r = criterion(out_r, labels_r)
            predicted = (prob_r.data.max(1)[1]).long().view(-1)

            if args.cuda:
                loss_r = loss_r.cpu()
                prob_r = prob_r.cpu()
                labels_r = labels_r.cpu()

            losses_t.append(loss_r.data.numpy())
            probs.append(prob_r)
            labels.append(labels_r)

        # retrieve and flatten entity prediction for loss calculation
        ent_pred, ent_label, ent_prob, ent_key, ent_pos = [], [], [], [], []
        for i, l in enumerate(lengths):
            # flatten prediction
            ent_pred.append(out_e[i, :l])
            # flatten entity prob
            ent_prob.append(prob_e[i, :l])
            # flatten entity label
            ent_label.append(ents[i, :l])
            # flatten entity key - a list of original (extend)
            assert len(ent_keys[i]) == l
            ent_key.extend(ent_keys[i])
            # flatten pos tags
            ent_pos.extend([p for p in poss[i]])

        ent_pred = torch.cat(ent_pred, 0)
        ent_label = torch.cat(ent_label, 0)
        ent_probs = torch.cat(ent_prob, 0)

        assert ent_pred.size(0) == ent_label.size(0)
        assert ent_pred.size(0) == len(ent_key)

        loss_e = criterion(ent_pred, ent_label)
        losses_e.append(loss_e.cpu().data.numpy())

        ent_label = ent_label.tolist()

        for i, v in enumerate(ent_key):
            label_e = ent_label[i]
            prob_e = ent_probs[i]

            # exclude sent_start and sent_sep
            if v in ["[SEP]", "[CLS]"]:
                assert ent_pos[i] in ["[SEP]", "[CLS]"]

            if v not in ent_pred_map:
                # only store the probability of being 1 (is an event)
                ent_pred_map[v] = [prob_e.tolist()[1]]
                ent_label_map[v] = (label_e, ent_pos[i])
            else:
                # if key stored already, append another prediction
                ent_pred_map[v].append(prob_e.tolist()[1])
                # and ensure label is the same
                assert ent_label_map[v][0] == label_e
                assert ent_label_map[v][1] == ent_pos[i]

        ## collect relation prediction results
        probs = torch.cat(probs,dim=0)
        labels = torch.cat(labels,dim=0)

        assert labels.size(0) == probs.size(0)

        # calculate entity F1 score here
        # update ent_pred_map with [mean > 0.5 --> 1]
        ent_pred_map_agg = {k:1 if np.mean(v) > 0.5 else 0 for k,v in ent_pred_map.items()}

        n_correct = 0
        n_pred = 0
            
        pos_keys = OrderedDict([(k, v) for k, v in ent_label_map.items() if v[0]==1])
        n_true = len(pos_keys)

        for k,v in ent_label_map.items():
            if ent_pred_map_agg[k] == 1:
                n_pred += 1
            if ent_pred_map_agg[k] == 1 and ent_label_map[k][0] == 1:
                n_correct += 1

        def safe_division(numr, denr, on_err=0.0):
            return on_err if denr == 0.0 else float(numr) / float(denr)

        assert len(event_heads) == len(pairs)
        if test:
            return probs.data, np.mean(losses_t), labels, docs, pairs, 0.0, nopred_rels, event_heads
        else:
            return probs.data, np.mean(losses_t), labels, docs, pairs, n_pred, n_true, n_correct, nopred_rels

    def construct_relations(self, ent_probs, lengths, rels, doc, poss, gold=True):
        # many relation properties such rev and pred_ind are not used for now
        
        nopred_rels = []

        ## Case 1: only use gold relation
        if gold:
            pred_rels = rels

        ## Case 2: use candidate relation predicted by entity model
        else:
            def _is_gold(pred_span, gold_rel_span):
                return ((gold_rel_span[0] <= pred_span <= gold_rel_span[1]))
                
            batch_size = ent_probs.size(0)
            ent_probs = ent_probs.cpu()
            
            # select event based on prob > 0.5, but eliminate ent_pred > context length
            ent_locs = [[x for x in (ent_probs[b,:, 1] > 0.5).nonzero().view(-1).tolist() 
                         if x < lengths[b]] for b in range(batch_size)]
            #print(ent_locs)
            # all possible relation candidate based on pred_ent
            rel_locs = [list(combinations(el, 2)) for el in ent_locs]
            #print(rel_locs)

            pred_rels = []
            totl = 0
            # use the smallest postive sample id as start of neg id
            # this may not be perfect, but we really don't care about neg id
            neg_counter = 0#min([int(x[0][1:]) for rel in rels for x in rel])

            for i, rl in enumerate(rel_locs):
                temp_rels, temp_ids = [], []
                for r in rl:
                    neg_id = 'N%s' % neg_counter

                    # provide a random but unique id for event predicted if not matched in gold
                    left_id =  ('e%s' % r[0])
                    right_id = ('e%s' % r[1])
                    a_rel = (neg_id, (left_id, right_id), self._label_to_id['NONE'],
                             [float(r[1] - r[0])], False, (r[0], r[0], r[1], r[1]), True)

                    temp_rels.append(a_rel)
                    neg_counter += 1

                pred_rels.append(temp_rels)
                
        # relations are (flatten) lists of features
        # rel_idxs indicates (batch_id, rel_in_batch_id)
        docs, pairs = [], []
        rel_idxs, lidx_start, lidx_end, ridx_start, ridx_end = [],[],[],[],[]
        for i, rel in enumerate(pred_rels):
            rel_idxs.extend([(i, ii) for ii, _ in enumerate(rel)])
            lidx_start.append([x[5][0] for x in rel])
            lidx_end.append([x[5][1] for x in rel])
            ridx_start.append([x[5][2] for x in rel])
            ridx_end.append([x[5][3] for x in rel])
            pairs.extend([x[1] for x in rel])
            docs.extend([doc[i] for _ in rel])
        assert len(docs) == len(pairs)
            
        rels = [x for rel in pred_rels for x in rel]
        if rels == []:
            labels = torch.FloatTensor([])
            fts = torch.FloatTensor([])
        else:
            labels = torch.LongTensor([x[2] for x in rels])
            fts = torch.cat([torch.FloatTensor(x[3]) for x in rels]).unsqueeze(1)
        
        return labels, fts, rel_idxs, docs, pairs, lidx_start, lidx_end, ridx_start, ridx_end, nopred_rels

    def _train(self, train_data, eval_data, pos_emb, args):

        model = BertClassifier(args)

        if args.cuda:
            print("using cuda device: %s" % torch.cuda.current_device())
            assert torch.cuda.is_available()
            model.cuda()
        #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        #criterion_e = nn.CrossEntropyLoss()
        
        if args.data_type in ['tbd']:
            weights = torch.FloatTensor([1.0, 1.0, 1.0, args.uw, args.uw, args.uw, 1.0])
  
        else:
            weights = torch.FloatTensor([1.0, 1.0, 1.0, args.uw, 1.0])
            
        if args.cuda:
            weights = weights.cuda()

        if args.load_model == True:
            checkpoint = torch.load(os.path.join(args.output_dir, args.entity_model_file))
            model.load_state_dict(checkpoint['state_dict'])
            self.model = copy.deepcopy(model)
        best_eval_f1 = 0.0 
        best_epoch = 0

        return best_eval_f1, best_epoch
                          
    def train_epoch(self, train_data, dev_data, args, test_data = None):

        if args.data_type == "matres":
            label_map = matres_label_map
        if args.data_type == "tbd":
            label_map = tbd_label_map
        assert len(label_map) > 0

        all_labels = list(OrderedDict.fromkeys(label_map.values()))
        ## append negative pair label
        all_labels.append('NONE')

        self._label_to_id = OrderedDict([(all_labels[l],l) for l in range(len(all_labels))])
        self._id_to_label = OrderedDict([(l,all_labels[l]) for l in range(len(all_labels))])

        args.label_to_id = self._label_to_id

        ### pos embdding is not used for now, but can be added later
        pos_emb= np.zeros((len(args.pos2idx) + 1, len(args.pos2idx) + 1))
        for i in range(pos_emb.shape[0]):
            pos_emb[i, i] = 1.0

        best_f1, best_epoch = self._train(train_data, dev_data, pos_emb, args)
        return best_f1, best_epoch

    def weighted_f1(self, pred_labels, true_labels, ent_corr, ent_pred, ent_true, rw=0.0, ew=0.0):
        def safe_division(numr, denr, on_err=0.0):
            return on_err if denr == 0.0 else numr / denr

        assert len(pred_labels) == len(true_labels)

        weighted_f1_scores = {}
        if 'NONE' in self._label_to_id.keys():
            num_tests = len([x for x in true_labels if x != self._label_to_id['NONE']])
        else:
            num_tests = len([x for x in true_labels])

        #print("Total positive samples to eval: %s" % num_tests)
        total_true = Counter(true_labels)
        total_pred = Counter(pred_labels)

        labels = list(self._id_to_label.keys())

        n_correct = 0
        n_true = 0
        n_pred = 0

        if rw > 0:
            # f1 score is used for tcr and matres and hence exclude vague                            
            exclude_labels = ['VAGUE', 'NONE'] if len(self._label_to_id) == 5 else ['NONE']

            for label in labels:
                if self._id_to_label[label] not in exclude_labels:

                    true_count = total_true.get(label, 0)
                    pred_count = total_pred.get(label, 0)

                    n_true += true_count
                    n_pred += pred_count

                    correct_count = len([l for l in range(len(pred_labels))
                                         if pred_labels[l] == true_labels[l] and pred_labels[l] == label])
                    n_correct += correct_count
        if ew > 0:
            # add entity prediction results before calculating precision, recall and f1
            n_correct += ent_corr
            n_pred += ent_pred
            n_true += ent_true

        precision = safe_division(n_correct, n_pred)
        recall = safe_division(n_correct, n_true)
        f1_score = safe_division(2.0 * precision * recall, precision + recall)
        #print("Overall Precision: %.4f\tRecall: %.4f\tF1: %.4f" % (precision, recall, f1_score))

        return(f1_score)

class EventEvaluator:
    def __init__(self, model):
        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    def evaluate(self, raw_text, args):
        # load test data first since it needs to be executed twice in this function
        featurized_data = featurize_raw_data(raw_text, self.tokenizer, self.bert_model, args)
        preds, loss, true_labels, docs, pairs, ent_f1, nopred_rels, event_heads\
            = self.model.predict(self.model.model, featurized_data, args, test = True, gold = args.eval_gold)

        doc_id, context_id, sents, ent_keys, ents, poss, rels, lengths, orig_tokens = featurized_data

        predicted_rels = [self.model._id_to_label[ridx] for ridx in torch.argmax(preds, dim=1).tolist()]
        idx2text = {v:k for k,v in orig_tokens.items()}
        events = [(idx2text[ent_keys[0][eidx][1]], ent_keys[0][eidx][1]) for eidx in set([x for pair in event_heads for x in pair])]
        relations = [(ent_keys[0][int(e1[1:])][1], ent_keys[0][int(e2[1:])][1], predicted_rels[i])
                     for i, (e1, e2) in enumerate(pairs)]
        output = {"tokens": orig_tokens,
                  "events": events,
                 "relations": relations}
        return output

def print_ratio(triplet_counter, pair_counter):
    data_size = sum(pair_counter.values())
    for e1, e2, rel in triplet_counter:
        if pair_counter[(e1, e2)] / data_size > 0.03:
            ratio = triplet_counter[(e1, e2, rel)] / pair_counter[(e1, e2)]
            print("%s,%s,%s %s %s %s" % (e1, e2, rel, triplet_counter[(e1, e2, rel)], pair_counter[(e1, e2)], ratio))
    return

def compute_ratio(split, filedir):
    triplet_counter = Counter()
    pair_counter = Counter()
    pair_lookup = {}
    with open('%s%s.pickle' % (filedir, split), 'rb') as handle:
        data = pickle.load(handle)
        for ex_id, ex in data.items():
            pair_lookup[(ex['doc_id'], ex['left_event'].id,
                         ex['right_event'].id)] = (ex['left_event'].type, ex['right_event'].type)
            triplet = (ex['left_event'].type, ex['right_event'].type)
            label = ex['rel_type']
            pair_counter[(ex['left_event'].type, ex['right_event'].type)] += 1
            triplet_counter[(ex['left_event'].type, ex['right_event'].type, label)] += 1
    handle.close()
    
    return pair_lookup

class TempRelAPI:
    def __init__(self, base_dir='.'):
        self.args = argparse.Namespace()
        self.args.base_dir = base_dir
        # arguments for data processing
        self.args.data_dir = os.path.join(self.args.base_dir, '../data')
        self.args.other_dir = os.path.join(self.args.base_dir, '../other')
        self.args.output_dir = os.path.join(self.args.base_dir, '../models')
        # select model
        self.args.model = 'multitask/pipeline'
        # arguments for RNN model
        self.args.emb = 100
        self.args.hid = 90
        self.args.num_layers = 1
        self.args.batch = 1
        self.args.data_type = 'matres'
        self.args.epochs = 0
        self.args.pipe_epoch = -1 # 1000: no pipeline training; otherwise <= epochs 
        self.args.seed = 123
        self.args.lr = 0.0005
        self.args.num_classes = 2 # get updated in main()
        self.args.dropout = 0.4
        self.args.ngbrs = 15
        self.args.pos2idx = {}
        self.args.w2i = OrderedDict()
        self.args.glove = OrderedDict()
        self.args.cuda = False
        self.args.refit_all = False
        self.args.uw = 1.0
        self.args.params = {}
        self.args.n_splits = 5
        self.args.pred_win = 200
        self.args.n_fts = 1
        self.args.relation_weight = 1.0
        self.args.entity_weight = 16.0
        self.args.save_model = False
        self.args.save_stamp = "matres_entity_best"
        self.args.entity_model_file = "matres_pipeline_best_hid90_dropout0.4_ew15.0.pth.tar"
        self.args.relation_model_file = ""
        self.args.load_model = True
        self.args.bert_config = {}
        self.args.fine_tune = False
        self.args.eval_gold = False
        self.args.input_text = ""

        if torch.cuda.device_count() > 1:
            print('TempRelAPI: Number of cuda devices: ', torch.cuda.device_count())
            self.args.cuda = True

        # create pos_tag and vocabulary dictionaries
        # make sure raw data files are stored in the same directory as train/dev/test data
        tags = open(self.args.other_dir + "/pos_tags.txt")
        pos2idx = {}
        idx = 0
        for tag in tags:
            tag = tag.strip()
            pos2idx[tag] = idx
            idx += 1
        self.args.pos2idx = pos2idx
        
        self.args.idx2pos = {v+1:k for k,v in pos2idx.items()}

    def pred(self, input_text):
        model = NNClassifier()
        # load pretrained model
        model.train_epoch([], [], self.args)
        # construct predictor
        evaluator = EventEvaluator(model)

        raw_text = input_text
        output = evaluator.evaluate(raw_text, self.args)
        return output
    
def main(args):
    
    model = NNClassifier()
    # load pretrained model
    model.train_epoch([], [], args)
    # construct predictor
    evaluator = EventEvaluator(model)

    raw_text = args.input_text
    output = evaluator.evaluate(raw_text, args)
    print(output)
    
if __name__ == '__main__':
    # api = TempRelAPI()
    # output = api.pred(["Orders went out today to deploy 17,000 U.S. Army soldiers in the Persian Gulf region.",
    #                     "Brooklyn Beckham asked Nicola Peltz to marry him , and she said yes , the cameraman and model announced on Saturday."])
    # print(output)
    p = argparse.ArgumentParser()
    # arguments for data processing
    p.add_argument('-data_dir', type=str, default = '../data/')
    p.add_argument('-other_dir', type=str, default = '../other/')
    p.add_argument('-output_dir', type=str, default = '../models/')
    # select model
    p.add_argument('-model', type=str, default='multitask/pipeline')
    # arguments for RNN model
    p.add_argument('-emb', type=int, default=100)
    p.add_argument('-hid', type=int, default=90)
    p.add_argument('-num_layers', type=int, default=1)
    p.add_argument('-batch', type=int, default=1)
    p.add_argument('-data_type', type=str, default="matres")
    p.add_argument('-epochs', type=int, default=0)
    p.add_argument('-pipe_epoch', type=int, default=-1) # 1000: no pipeline training; otherwise <= epochs 
    p.add_argument('-seed', type=int, default=123)
    p.add_argument('-lr', type=float, default=0.0005)
    p.add_argument('-num_classes', type=int, default=2) # get updated in main()
    p.add_argument('-dropout', type=float, default=0.4)
    p.add_argument('-ngbrs', type=int, default = 15)                                   
    p.add_argument('-pos2idx', type=dict, default = {})
    p.add_argument('-w2i', type=OrderedDict)
    p.add_argument('-glove', type=OrderedDict)
    p.add_argument('-cuda', action='store_true')
    p.add_argument('-refit_all', type=bool, default=False)
    p.add_argument('-uw', type=float, default=1.0)
    p.add_argument('-params', type=dict, default={})
    p.add_argument('-n_splits', type=int, default=5)
    p.add_argument('-pred_win', type=int, default=200)
    p.add_argument('-n_fts', type=int, default=1)
    p.add_argument('-relation_weight', type=float, default=1.0)
    p.add_argument('-entity_weight', type=float, default=16.0)
    p.add_argument('-save_model', type=bool, default=False)
    p.add_argument('-save_stamp', type=str, default="matres_entity_best")
    p.add_argument('-entity_model_file', type=str, default="matres_pipeline_best_hid90_dropout0.4_ew15.0.pth.tar")
    p.add_argument('-relation_model_file', type=str, default="")
    p.add_argument('-load_model', type=bool, default=True)
    p.add_argument('-bert_config', type=dict, default={})
    p.add_argument('-fine_tune', type=bool, default=False)
    p.add_argument('-eval_gold',type=bool, default=False)

    p.add_argument('-input_text', type=str, default="")
    args = p.parse_args()

    # create pos_tag and vocabulary dictionaries
    # make sure raw data files are stored in the same directory as train/dev/test data
    tags = open(args.other_dir + "/pos_tags.txt")
    pos2idx = {}
    idx = 0
    for tag in tags:
        tag = tag.strip()
        pos2idx[tag] = idx
        idx += 1
    args.pos2idx = pos2idx
    
    args.idx2pos = {v+1:k for k,v in pos2idx.items()}

    main(args)