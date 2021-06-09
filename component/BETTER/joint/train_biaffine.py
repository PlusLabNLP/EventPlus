from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from dataset import *
import numpy as np
from util import Logger
from seqeval.metrics import f1_score, accuracy_score, classification_report
import copy
from CRF_util import kViterbi
from eval import *
from transformers import AdamW, get_linear_schedule_with_warmup

def write_pkl(sent_ids, y_trig_input, y_preds_e, out_pkl_dir, pkl_name):
    #ensure dir
    if not os.path.exists(out_pkl_dir):
        os.makedirs(out_pkl_dir)

    output_t_e_event = []
    for i in range(len(sent_ids)):
        output_t_e_event.append({'sent_id': sent_ids[i], 'pred_trigger': y_trig_input[i], 'pred_arg': y_preds_e[i]})
    with open('{}/{}'.format(out_pkl_dir, pkl_name), 'wb') as f:
        pickle.dump(output_t_e_event, f)
    print('pkl saved at {}{}'.format(out_pkl_dir, pkl_name))

def get_tri_idx_from_mix_sent(out_t, B2I):
    '''
    get trigger idx from a sent with possibly MULTIPLE events
    it is expected that the `out_t` is the predited trigger sequences from model
    it finds the idxs for BIO chunks
    Return:
        tri_idx: list(len:#events) of list(len:#tokens in an event) of int
        tri_type: list(len:#events) of int(the id of B-xxx)
    '''
    tri_idx = []
    tri_type = []
    in_chunk = False
    curr_idx = []
    curr_I = None
    for i in range(len(out_t)):
        # end of chunk
        if in_chunk:
            if out_t[i] != curr_I:
                tri_idx.append(curr_idx)
                tri_type.append(curr_I)
                curr_idx = []
                curr_I = None
                in_chunk = False
            elif out_t[i] == curr_I:
                curr_idx.append(i)
                if i == len(out_t) - 1:
                    # the last token is a I token
                    tri_idx.append(curr_idx)
                    tri_type.append(curr_I)

        # beginning of chunk
        if out_t[i] in B2I:
            curr_idx = [i]
            in_chunk = True
            curr_I = B2I[out_t[i]] # index of I-xxx
            if i == len(out_t) - 1:
                # the last token is a B token
                tri_idx.append(curr_idx)
                tri_type.append(curr_I)

    assert len(tri_idx) == len(tri_type)
    return tri_idx, tri_type

def tri_partial_match(pred_tri_idx, gold_tri_idx):
    #  seems like NOT including the head equal looks better
    if len(pred_tri_idx) == 1:
        # for predicted single-token trigger, require it to be contained in gold
        if pred_tri_idx[0] >= gold_tri_idx[0] and pred_tri_idx[0] <= gold_tri_idx[-1]:
            return True
    elif len(pred_tri_idx) > 1:
        # for predicted multi-token trigger, require it to be either included by the gold_tri_idx
        if pred_tri_idx[0] >= gold_tri_idx[0] and pred_tri_idx[-1] <= gold_tri_idx[-1]:
            return True

    return False

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

def get_gold(lengths, label):
    if label is not None:
        label_list = [label[i, :l].tolist() for i, l in enumerate(lengths)]
    else:
        label_list = []
    return label_list

class NNClassifier():
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model
        self.logger = Logger(args.log_dir)
        self.best_model = model
        self.best_model_t = model
        self.dev_cnt = 0
    def loss_argu_by_ent(self, biaffine_tensor, ent_to_arg_dicts, lengths):
        labels, preds = [], []
        loss = nn.CrossEntropyLoss()
        for b, biaffine_pred in enumerate(biaffine_tensor):
            # b: batch id; biaffine_pred: (len1, len2, #label)
            ent_to_arg = ent_to_arg_dicts[b]
            length = lengths[b]
            for ent, argus in ent_to_arg.items():
                ### first generate the target label
                label = torch.LongTensor([self.args._label_to_id_e['O']] * length)
                if ent[-1] - ent[0] >0:
                    # multi-token ent
                    label = [label] * (ent[-1] - ent[0] + 1)
                else:
                    label = [label]
                for argu in argus:
                    assert argu[0] == argu[1]  # trigger is single-token
                    label[0][argu[0]] = self.args._label_to_id_e['B-{}'.format(argu[2])]    # always use the B-xxx
                    for i in range(1, (ent[-1]-ent[0])+1):
                        label[i][argu[0]] = self.args._label_to_id_e['I-{}'.format(argu[2])]    # always use the B-xxx
                labels.extend(label)
                ### next select columns
                l_idx = ent[0]
                r_idx = ent[1]
                pred = [biaffine_pred[:length, i, :] for i in range(l_idx, r_idx+1)]
                preds.extend(pred)

        if len(labels) > 0:
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)
            if self.args.cuda:
                labels = labels.cuda()
            column_loss = loss(preds, labels)
        else:
            column_loss = torch.Tensor([0.])
        return column_loss



    def loss_argu_biaffine(self, biaffine_tensor, arg_exts, tri_idxs,
                           lengths_ext, n_events, batch_size, crf_a):
        '''
        Input:
            biaffine_tensor is the model output, which with size (batch, len1, len2, # of arglabel)
            arg_exts is the target output that  has been extended
                to size (total_event_inbatch, max_seq_len)
            tri_idxs is list(#of total events in batch) of list(events) of int
            lengths_ext is a list of int, which indicate the length that align with tri_idxs
            n_events is a list(# of batch) of int, each int indicate how many event in the batch
            batch_size is an int
            crf_a is a CRF object
        Return:
            `torch.Tensor`: a mean-pooling value of loss
        '''
        # The idea here is to first accumulate corresponding prediction logit,
        # then pass through crf module (if crf)
        # then finally compare output with arg_exts to calculate loss
        cnt = 0
        max_len = arg_exts.size(1)
        prediction_logits = []
        masks = []
        gold = []
        for b_idx,n in enumerate(n_events):
            if n > 0:
                for _ in range(n):
                    assert len(tri_idxs[cnt])>0
                    for token in [tri_idxs[cnt][0]]:    # only consider the first token in the multi-token event
                        mask = torch.arange(max_len) < torch.LongTensor([lengths_ext[cnt]])
                        if self.args.cuda:
                            mask = mask.cuda()
                        masks.append(mask) # batch_max_length
                        if len(n_events) == batch_size:
                            emissions = biaffine_tensor[b_idx][token] #len_2 x #argulabel
                        elif len(n_events) == batch_size*2:
                            emissions = biaffine_tensor[b_idx//2][token]
                        else:
                            assert False
                        prediction_logits.append(emissions)
                        gold.append(arg_exts[cnt])
                    cnt += 1
            elif n == 0:
                cnt += 1
            else:
                assert False
        assert len(arg_exts)==cnt
        if len(masks)==0:
            return torch.Tensor([0.0])
        masks = torch.stack(masks, dim=0)
        prediction_logits = torch.stack(prediction_logits, dim=0)
        gold = torch.stack(gold, dim=0)
        # pass thrugh crf
        if crf_a is not None:
            total_loss = -1 * crf_a(prediction_logits, gold, mask=masks, reduction='token_mean')
        else:
            log_probs = F.log_softmax(prediction_logits, dim=-1)
            log_probs = torch.gather(log_probs, dim=2, index=gold.unsqueeze(2)).squeeze(2)
            O_list = torch.Tensor([self.args._label_to_id_e['O']])
            if self.args.cuda:
                O_list = O_list.cuda()
            bias_mask = (gold.long() != O_list.long())
            value = log_probs.new_ones(log_probs.size())
            value = value.masked_fill_(bias_mask, self.args.argument_bias)
            loss = (-torch.sum(log_probs*masks*value))
            total_loss = loss/(torch.sum(masks))
        return total_loss

    def pooling_arg_pred(self, biaffine_prediction, event_id, length, crf_a):
        '''
        biaffine_prediction is a 3d matrix (len1, len2, #label)
        event_id is a list (can be empty)
        lengths is an int
        '''
        if crf_a is None:
            if len(event_id)==0:
                return ([self.args._label_to_id_e['O']]* length).tolist()
            #elif len(event_id)==1:
            #    biaffine_prediction = torch.argmax(biaffine_prediction, dim=-1) #(len1, len2)
            #    return biaffine_prediction[event_id[0]][:length].tolist()
            #else:
            #    preds = []
            #    for e in event_id:
            #        preds.append(biaffine_prediction[e][:length]) #(length, #label)
            #    #pooling by probs
            #    stack = torch.stack(preds, dim=0) #(#ofevent, length, #label)
            #    if self.args.argupooled_byprob:
            #        return torch.argmax(torch.max(stack, dim=0)[0], dim=1).tolist()
            #    else:
            #        return torch.max(torch.argmax(stack, dim=-1), dim=0)[0].tolist()
            else:
                biaffine_prediction = torch.argmax(biaffine_prediction, dim=-1) #(len1, len2)
                return biaffine_prediction[event_id[0]][:length].tolist()
        else:
            mask = (torch.arange(biaffine_prediction.size(1)) < torch.LongTensor([length])).unsqueeze(0) #1 x max_length
            if self.args.cuda:
                mask = mask.cuda()
            if len(event_id)==0:
                return ([self.args._label_to_id_e['O']]* length).tolist()
            elif len(event_id)==1:
                prediction, _ = kViterbi(crf_a, biaffine_prediction[event_id[0]].unsqueeze(0), 1, mask)
                return prediction[0]
            else:
                preds = []
                for e in event_id:
                    p, s = kViterbi(crf_a, biaffine_prediction[e].unsqueeze(0), 1, mask)
                    preds.append((p, s.item()))
                best_seq, best_score = sorted(preds, key=lambda x:x[1])[-1]
                return best_seq

    def predict(self, data, test=False, blind_test=False, use_gold_trig=False):
        '''
        if test is True, we will use our best_model to eval the performance,
            otherwise, we will use our current model
        if blind_test, we will not get test/dev loss,
            otherwise, we can collect predicting loss
        if use_gold_trig is True, we will pass our gold trigger to access argument prediction
            otherwise, we will use our predicted trigger
        Note: therefore, there's no way that we do blind test and use gold trig on the mean time
        '''
        assert not (blind_test and use_gold_trig)
        if test:
            self.model.load_state_dict(self.best_model_state_dict)
            # self.best_model.eval()
        if self.args.cuda:
            self.model.cuda()
            # if test:
            #     self.best_model.cuda()
        self.model.eval()
        count = 1
        y_trues_argu, y_preds_argu, y_trues_trig, y_preds_trig = [], [], [], []
        y_trig_input = []
        sent_ids_out = []

        for sent_ids, sents, poss, triggers, arguments, lengths, seq_pairs, all_pairs, glove_idx, orig_to_tok_map, bert_lengths, ent_to_arg_dict in data:
            if not blind_test:
                assert seq_pairs is not None
            batch_size = sents.size(0)
            crf_mask=None
            if self.args.use_crf:
                max_len = lengths[0]
                crf_mask = torch.arange(max_len).expand(len(lengths), max_len) < torch.LongTensor(lengths).unsqueeze(1)
            if self.args.finetune_bert:
                bert_max_len = max(bert_lengths)
                bert_attn_mask = torch.arange(bert_max_len).expand(len(bert_lengths), bert_max_len) < torch.LongTensor(bert_lengths).unsqueeze(1)
                bert_attn_mask = bert_attn_mask.type(torch.FloatTensor)
            else:
                bert_attn_mask = None

            if self.args.cuda:
                sents = sents.cuda()
                poss = poss.cuda()
                arguments = arguments.cuda()
                triggers = triggers.cuda()
                if glove_idx is not None:
                    glove_idx = glove_idx.cuda()
                if self.args.finetune_bert and len(bert_attn_mask) > 0:
                    bert_attn_mask = bert_attn_mask.cuda()
                if self.args.use_crf:
                    crf_mask = crf_mask.cuda()

            if test:
                # # predicted by best trigger model
                # trig_pred_t, trig_loss_t, biaffine_tensor_t, crf_a_t, _ =\
                #     self.best_model_t(sents, poss, lengths, decode=blind_test, gold_trig_seq=triggers, crf_mask=crf_mask,
                #                     glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map, bert_attn_mask=bert_attn_mask)
                # # predicted by best argument model
                # trig_pred_a, trig_loss_a, biaffine_tensor_a, crf_a_a, _ =\
                #     self.best_model(sents, poss, lengths, decode=blind_test, gold_trig_seq=triggers, crf_mask=crf_mask,
                #                     glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map, bert_attn_mask=bert_attn_mask)
                # # select the tri_pred from best trigger model adn biaffine_tensor from best argument model, respectively
                # trig_pred = trig_pred_t
                # trig_loss = trig_loss_t
                # biaffine_tensor = biaffine_tensor_a
                # crf_a = crf_a_a
                # predicted by best argument model
                trig_pred, trig_loss, biaffine_tensor, crf_a, _ =\
                    self.model(sents, poss, lengths, decode=blind_test, gold_trig_seq=triggers, crf_mask=crf_mask,
                                    glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map, bert_attn_mask=bert_attn_mask)

            else:
                trig_pred, trig_loss, biaffine_tensor, crf_a, _ =\
                    self.model(sents, poss, lengths, decode=blind_test, gold_trig_seq=triggers, crf_mask=crf_mask,
                               glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map, bert_attn_mask=bert_attn_mask)
            # start processing argument part
            # we follow the steps:
            # 1) based on whether we use gold trigger or not to generate
            # extended evaluation/gold argument data.
            #   if trigger is gold, we have the gold argument data
            #   if trigger is predicted, we can use gold-trig-argu pair to
            #   generate fake labels as a measurement (if the pair is given)
            # 2) generate our model prediction (by gold trigger or generated
            # trigger.
            # 3) calculate predicted loss if gold y can be inferred in (1).
            if blind_test:
                sent_ids_ext, sents_ext, poss_ext, args_ext, lengths_ext, tri_idxs, tri_types, n_events, new_pairs =\
                    self.expand_sents(sent_ids, sents, poss, lengths, seq_pairs=None,
                                      pred_trig=trig_pred, use_gold_trig=False, use_pred_trig=True)
            else:
                sent_ids_ext, sents_ext, poss_ext, args_ext, lengths_ext, tri_idxs, tri_types, n_events, new_pairs =\
                    self.expand_sents(sent_ids, sents, poss, lengths, seq_pairs,
                                      trig_pred, use_gold_trig, not (use_gold_trig))
            sent_ids_out.extend(sent_ids_ext)
            y_trig_input.extend([x[0] for x in new_pairs])
            if not blind_test:
                y_trues_argu.extend([x[1] for x in new_pairs])

            # generate model prediction
            arg_pred = []
            cnt = 0
            for b_idx, n in enumerate(n_events):
                if n > 0:
                    for i in range(n):
                        arg_pred.append(self.pooling_arg_pred(biaffine_tensor[b_idx], tri_idxs[cnt], lengths_ext[cnt], crf_a))
                        cnt +=1
                else:
                    arg_pred.append([self.args._label_to_id_e['O']]* lengths_ext[cnt])
                    cnt += 1
            assert cnt == len(tri_idxs)


            # calculate loss if it's not a blind test
            if not blind_test:
                total_argu_loss = self.loss_argu_biaffine(biaffine_tensor, args_ext, tri_idxs, lengths_ext, n_events, batch_size, crf_a)
                if not test:
                    self.logger.scalar_summary(tag='dev/trig_loss',
                                               value=trig_loss,
                                               step=self.dev_cnt)
                    self.logger.scalar_summary(tag='dev/argu_loss',
                                               value=total_argu_loss,
                                               step=self.dev_cnt)
                else:
                    self.logger.scalar_summary(tag='test/trig_loss',
                                               value=trig_loss,
                                               step=1)
                    self.logger.scalar_summary(tag='test/argu_loss',
                                               value=total_argu_loss,
                                               step=1)
                trig_label = get_gold(lengths, triggers)
                y_trues_trig.extend(trig_label)
            y_preds_trig.extend(trig_pred)
            y_preds_argu.extend(arg_pred)
        y_trues_argu = [[self.args._id_to_label_e[x] for x in y] for y in y_trues_argu]
        y_preds_argu = [[self.args._id_to_label_e[x] for x in y] for y in y_preds_argu]
        if self.args.merge_arguBIO:
            y_trues_argu = self.from_merge_to_IOB(y_trues_argu)
            y_preds_argu = self.from_merge_to_IOB(y_preds_argu)
        y_trues_trig = [[self.args._id_to_label_t[x] for x in y] for y in y_trues_trig]
        y_preds_trig = [[self.args._id_to_label_t[x] for x in y] for y in y_preds_trig]
        y_trig_input = [[self.args._id_to_label_t[x] for x in y] for y in y_trig_input]
        if not test:
            self.dev_cnt += 1
        return y_trues_trig, y_preds_trig, y_trues_argu, y_preds_argu, y_trig_input, sent_ids_out

    def from_merge_to_IOB(self, batch_of_argu_texts):
        '''
        Input:
            batch_of_argu_texts: is a list of argument label sequence. For example"
                [['O', 'O', 'AGENT', 'AGENT', 'PATIENT', O', ....]..[....]...]
        Return:
            Turning the input sequence with BIO tagging
        '''
        final_output = []
        for argu_texts in batch_of_argu_texts:
            sent_output = []
            in_chunk = False
            current_chunk = None
            for token in argu_texts:
                if in_chunk:
                    if token == current_chunk:
                        sent_output.append('I-'+token)
                    elif token in self.args.positive_argu:
                        sent_output.append('B-'+token)
                        current_chunk = token
                    else:
                        in_chunk = False
                        current_chunk = None
                        sent_output.append(token)
                else:
                    if token in self.args.positive_argu:
                        sent_output.append('B-'+token)
                        current_chunk = token
                        in_chunk = True
                    else:
                        sent_output.append(token)
            final_output.append(sent_output)
        return final_output

    def expand_sents_helper(self, seq_pair, sent_id, sent_ids_ext, sent, sents_ext,
                            pos, poss_ext, length, lengths_ext, tri_idxs, tri_types,
                            n_events, args_ext):
        n_event = len(seq_pair)
        n_events.append(n_event)
        sent_ids_ext.extend([sent_id] * max(1,n_event))
        sents_ext.extend([sent] * max(1,n_event))
        poss_ext.extend([pos] * max(1,n_event))
        lengths_ext.extend([length] * max(1,n_event))
        if n_event == 0:
            tri_idxs.append([])
            tri_types.append([])
            # the args label is an neg sample, consisting of all "O"s
            args_ext.append([self.args._label_to_id_e['O']]* length)
        else:
            for j in range(n_event):
                tri_idx, tri_type =\
                    get_tri_idx_from_mix_sent(seq_pair[j][0], self.args.B2I_trigger)
                try:
                    assert len(tri_idx)==1 # since it's gold
                except AssertionError:
                    pass
                try:
                    tri_idxs.append(tri_idx[0])
                    tri_types.append(tri_type[0])
                except:
                    pdb.set_trace()
            args_ext.extend([x[1] for x in seq_pair])
        return sent_ids_ext, sents_ext, poss_ext, lengths_ext, tri_idxs, tri_types, n_events, args_ext

    def expand_sents(self, sent_ids, sents, poss, lengths, seq_pairs=None,
                     pred_trig=None, use_gold_trig=True, use_pred_trig=False):
        '''
        expand the sents, poss and arguments label tensors, from sentence level
        instancse to event(trigger) level instances

        input:
            sent_ids: list of integer, in length of (batch_size)
            sents and poss: tensor of shape (batch_size, max_seq_len)
            lengths: list of integer, in length of (batch_size)
            seq_pairs: list of length batch_size, each item is a list of (seq,
                seq) pairs, the list's length == n_event in this sent
            pred_trig: list(batch) of list(seq_len)

        Do:
            for each batch:
                case 1--no seq_pairs:
                    splitting the pred_trig into chunks of events, and based on
                    the events to extend sent_ids, sents, etc. but left argument
                    target empty
                case 2--have gold seq_pairs, and use predicted trigger:
                    splitting the pred_trig into chunks of events, and based on
                    the events to extend sent_ids, sents, etc. And based on the
                    seq_pairs to generated (fake) labels for events
                case 3--have gold seq_pairs, and use gold trigger:
                    based on the seq_pairs to  generate gold input and argu. output
                case 4--mix case2 and case3

        output:
            sent_ids_ext, sents_ext, poss_ext, lengths_ext: extension of system input
            sent_ids_ext: list(len:#of total events in batch) of str
            sents_ext, poss_ext: `torch.Tensor` in size (#of total events in batch, max_len, dim)
            lengths_ext: list(len:#of total events in batch) of int

            arg_ext: `torch.Tensor` in size (#of total events in batch, max_len)
                will be `None` if seq_pairs is None
            tri_idxs is list(len:#of total events in batch) of list(len:#tokens in event) of int
            tri_types is list(len:#of total events in batch) of int(the id for type)
            n_events: list(batch) of integer(#of events for each batch)
            new_pairs: list(len:#of total events in batch) of tuple (trig_seq, argu_seq)
        '''
        sent_ids_ext, sents_ext, poss_ext, lengths_ext = [], [], [], []
        n_events, tri_idxs, tri_types, args_ext = [], [], [], []
        new_pairs = []
        assert (use_gold_trig or use_pred_trig)
        assert (seq_pairs is not None) or (pred_trig is not None)
        if seq_pairs is None:
            assert not (use_gold_trig)
        if seq_pairs is not None:
            for i in range(sents.size(0)):  #batch_size
                # case 3
                if use_gold_trig:
                    sent_ids_ext,sents_ext,poss_ext,lengths_ext,tri_idxs,tri_types,n_events,args_ext=\
                        self.expand_sents_helper(seq_pairs[i], sent_ids[i], sent_ids_ext, sents[i],
                                                 sents_ext, poss[i], poss_ext, lengths[i],
                                                 lengths_ext, tri_idxs, tri_types, n_events, args_ext)
                    if len(seq_pairs[i])==0:
                        new_pairs.append(([self.args._label_to_id_t['O']]*lengths[i],
                                          [self.args._label_to_id_e['O']]*lengths[i]))
                    elif len(seq_pairs[i])>0:
                        new_pairs.extend(seq_pairs[i])
                # case 2
                if use_pred_trig:
                    # first split predicted trigger
                    tri_idx, tri_type = get_tri_idx_from_mix_sent(pred_trig[i], self.args.B2I_trigger)
                    seq_len = len(pred_trig[i])
                    all_pairs, add_pairs = self.generate_arg_seq(tri_idx, tri_type, seq_pairs[i],
                                                                seq_len, self.args.B2I_trigger,
                                                                self.args.tri_partial_match)
                    if use_gold_trig:
                        sent_ids_ext,sents_ext,poss_ext,lengths_ext,tri_idxs,tri_types,n_events,args_ext=\
                            self.expand_sents_helper(add_pairs, sent_ids[i], sent_ids_ext, sents[i], sents_ext,
                                                     poss[i], poss_ext, lengths[i], lengths_ext, tri_idxs,
                                                     tri_types, n_events, args_ext)
                        if len(add_pairs)==0:
                            new_pairs.append(([self.args._label_to_id_t['O']]*lengths[i],
                                              [self.args._label_to_id_e['O']]*lengths[i]))
                        elif len(add_pairs)>0:
                            new_pairs.extend(add_pairs)
                    else: #if we haven't add gold trig sequence before, we need to add all pairs
                        sent_ids_ext,sents_ext,poss_ext,lengths_ext,tri_idxs,tri_types,n_events,args_ext=\
                            self.expand_sents_helper(all_pairs, sent_ids[i], sent_ids_ext, sents[i], sents_ext,
                                                     poss[i], poss_ext, lengths[i], lengths_ext, tri_idxs,
                                                     tri_types, n_events, args_ext)
                        if len(all_pairs)==0:
                            new_pairs.append(([self.args._label_to_id_t['O']]*lengths[i],
                                              [self.args._label_to_id_e['O']]*lengths[i]))
                        elif len(all_pairs)>0:
                            new_pairs.extend(all_pairs)
        else:
            for i in range(len(pred_trig)): # batch_size
                tri_idx, tri_type = get_tri_idx_from_mix_sent(pred_trig[i], self.args.B2I_trigger)
                n_event = len(tri_idx)
                n_events.append(n_event)

                sent_ids_ext.extend([sent_ids[i]] * max(1,n_event))
                sents_ext.extend([sents[i]] * max(1,n_event))
                poss_ext.extend([poss[i]] * max(1,n_event))
                lengths_ext.extend([lengths[i]] * max(1,n_event))

                if n_event == 0:
                    tri_idxs.append([])
                    tri_types.append([])
                else:
                    tri_idxs.extend(tri_idx) # tri_idx is a list of idxs
                    tri_types.extend(tri_type)
                tri_seqs = self.get_expand_event_level_trigger_seqs_from_idx(tri_idx, tri_type, lengths[i])
                if len(tri_seqs)==0:
                    new_pairs.append(([self.args._label_to_id_t['O']]*lengths[i],
                                      []))
                else:
                    for tri_seq in tri_seqs:
                        new_pairs.append((tri_seq, []))
            args_ext = None

        # pad each inputs
        sents_ext = pad_sequence([s for s in sents_ext], batch_first=True, padding_value=TOKEN_PAD_ID)
        poss_ext = pad_sequence([s for s in poss_ext], batch_first=True, padding_value=POS_PAD_ID)
        if self.args.cuda:
            sents_ext = sents_ext.cuda()
            poss_ext = poss_ext.cuda()
        if seq_pairs is not None:
            # Only during training will args_ext be available
            args_ext = pad_sequence([torch.LongTensor(s) for s in args_ext], batch_first=True, padding_value=ARGU_PAD_ID)
            if self.args.cuda:
                args_ext = args_ext.cuda()
            assert args_ext.size(0) == poss_ext.size(0) == sents_ext.size(0) == len(tri_idxs) == len(lengths_ext) == len(new_pairs)

        return sent_ids_ext, sents_ext, poss_ext, args_ext, lengths_ext, tri_idxs, tri_types, n_events, new_pairs

    def generate_arg_seq(self, tri_idxs, tri_types, seq_pairs, seq_len,
                         B2I_trigger, partial_match=False):
        '''
        Note: sentence-level not batch-level
        Input:
            tri_idxs: list of list--like [[2], [4,5], [10]], indicating the trigger idxs (possibly >1 events) in a sent
            tri_types: the corresponding event-types for tri_idxs
            seq_pairs is the gold pairs in this sent; list of tuple--(trig_seq, argu_seq)
            seq_len: int
            B2I-trigger: a dict like {'B-anchor':'I-anchor',...}
        Return:
            add_pairs: a list of tuple (trig_seq, argu_seq), that are generated by miss match
            all_pairs: a list of tuple (trig_seq, argu_seq), that are generated for all tri_idxs
            Note: if len(tri_idxs)==0, the output will be two empty list
        '''
        gold_tri_idx_to_arg_seq = {}
        all_pairs = []
        add_pairs = []

        # construct helper dict
        for i in range(len(seq_pairs)):
            gold_tri_idx, gold_tri_type = get_tri_idx_from_mix_sent(seq_pairs[i][0], B2I_trigger)
            gold_tri_idx_to_arg_seq[tuple(gold_tri_idx[0])] = seq_pairs[i][1] #gold arg seq (in index)

        tri_seqs = self.get_expand_event_level_trigger_seqs_from_idx(tri_idxs, tri_types, seq_len)
        for i, t in enumerate(tri_idxs): # #of events in the input sentence
            tri_type = tri_types[i]
            if partial_match is False:
                if gold_tri_idx_to_arg_seq.get(tuple(t)):
                    all_pairs.append((tri_seqs[i], gold_tri_idx_to_arg_seq.get(tuple(t)))) #gold argu seq (in index, non-padded)
                else:
                    # a seq of "O"s, as a neg sample
                    all_pairs.append((tri_seqs[i], [self.args._label_to_id_e['O']] * seq_len))
                    add_pairs.append((tri_seqs[i], [self.args._label_to_id_e['O']] * seq_len))
            elif partial_match is True:
                if gold_tri_idx_to_arg_seq.get(tuple(t)):
                    all_pairs.append((tri_seqs[i], gold_tri_idx_to_arg_seq.get(tuple(t)))) #gold argu seq (in index, non-padded)
                else:
                    find_flag=False
                    for gold_tri_idx in list(gold_tri_idx_to_arg_seq.keys()):
                        match = tri_partial_match(tuple(t), gold_tri_idx)
                        if match is True:
                            all_pairs.append((tri_seqs[i], gold_tri_idx_to_arg_seq.get(gold_tri_idx)))
                            add_pairs.append((tri_seqs[i], gold_tri_idx_to_arg_seq.get(gold_tri_idx)))
                            find_flag = True
                            break
                    if find_flag is False:
                        # a seq of "O"s, as a neg sample
                        all_pairs.append((tri_seqs[i], [self.args._label_to_id_e['O']] * seq_len))
                        add_pairs.append((tri_seqs[i], [self.args._label_to_id_e['O']] * seq_len))
        return all_pairs, add_pairs

    def get_expand_event_level_trigger_seqs_from_idx(self, tri_idx, tri_type, seq_len):
        '''
        Input:
            tri_idx: list(# of events) of list(# of tokens in the event) of int
            tri_type: list(# of events) of int (B-xxx index)
            seq_len: the length of the trigger sequence
        Return:
            a list of trigger sequence, each of which is paired with the input tri_idx list event
        '''
        assert len(tri_idx) == len(tri_type)
        output = []
        for i in range(len(tri_idx)): # #of events
            tri_seq = [self.args._label_to_id_t['O']] * seq_len
            if len(tri_idx[i]) == 0:
                # this sent has no predicted triggers
                tri_seq = tri_seq
            # elif len(tri_idx[i]) == 1:
            #     # single-token triggers
            #     tri_seq[tri_idx[i][0]] = tri_type[i]
            # elif len(tri_idx[i]) > 1:
            #     # multi-token triggers
            #     lidx = tri_idx[i][0]
            #     ridx = tri_idx[i][-1]
            #     tri_seq[lidx] = tri_type[i] #B-xxx token
            #     tri_seq[lidx+1:ridx+1] = [tri_type[i]] * (ridx - lidx) #I-xxx token
            else:
                # single-token triggers
                tri_seq[tri_idx[i][0]] = tri_type[i]
            output.append(tri_seq)
        return output

    def _eval_helper(self, eval_data, step, dev=True):
        if dev:
            y_trues_t,y_preds_t,y_trues_e,y_preds_e,y_trig_input,sent_ids_new = self.predict(eval_data, False, False, True)
        else:
            y_trues_t,y_preds_t,y_trues_e,y_preds_e,y_trig_input,sent_ids_new = self.predict(eval_data, True, False, True)
        text = 'dev' if dev else 'test'
        print("====feed gold trigger eval====")
        if len(y_trues_t) > 0:
            f1_t = f1_score(y_trues_t, y_preds_t)
            acc_t = accuracy_score(y_trues_t, y_preds_t)
            print('trigger f1: {:.4f}, trigger acc: {:.4f}'.format(f1_t, acc_t))
            self.logger.scalar_summary(tag='{}/trigger_f1'.format(text),
                                       value=f1_t,
                                       step=step)
            # print(classification_report(y_trues_t, y_preds_t))
        if len(y_trues_e) > 0:
            f1_e = f1_score(y_trues_e, y_preds_e)
            acc_e = accuracy_score(y_trues_e, y_preds_e)
            print('arguments f1: {:.4f}, arguments acc: {:.4f}'.format(f1_e, acc_e))
            self.logger.scalar_summary(tag='{}/argument_f1'.format(text),
                                       value=f1_e,
                                       step=step)
        out_pkl_dir = './tmp/{}'.format(self.args.text)
        if dev:
            pkl_name = 'dev_goldtrig_cls.pkl'
        else:
            pkl_name = 'test_goldtrig_cls.pkl'
        write_pkl(sent_ids_new, y_trig_input, y_preds_e, out_pkl_dir, pkl_name)

        print("====end2end eval====")
        if dev:
            y_trues_t,y_preds_t,y_trues_e,y_preds_e,y_trig_input,sent_ids_new = self.predict(eval_data, False, False, False)
        else:
            y_trues_t,y_preds_t,y_trues_e,y_preds_e,y_trig_input,sent_ids_new = self.predict(eval_data, True, False, False)

        out_pkl_dir = './tmp/{}'.format(self.args.text)
        B2I_trigger = self.args.B2I_trigger_string
        if dev:
            pkl_name = 'dev_end2end_cls.pkl'
        else:
            pkl_name = 'test_end2end_cls.pkl'
        write_pkl(sent_ids_new, y_trig_input, y_preds_e, out_pkl_dir, pkl_name)
        gold_file = 'tmp/gold_dev_biaffine_internal_cls.pkl' if dev else 'tmp/gold_test_biaffine_internal_cls.pkl'
        if os.path.isfile(gold_file) and os.path.isfile('{}/{}'.format(out_pkl_dir, pkl_name)):
            precs, recalls, f1_e2es = eval_ace(gold_file,
                                            '{}/{}'.format(out_pkl_dir, pkl_name),
                                            B2I_trigger,
                                            self.args.B2I_arg_string)
            self.logger.scalar_summary(tag='{}/f1_e2e'.format(text),
                                       value=f1_e2es[-1],
                                       step=step)
            print('end2end eval f1 {}'.format(f1_e2es[-1]))
        return f1_t, f1_e, f1_e2es

    def _train(self, train_data, eval_data, test_data, train_task):
        if self.args.cuda:
            print("using cuda device: %s" % torch.cuda.current_device())
            assert torch.cuda.is_available()
            self.model.cuda()

        if self.args.opt == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr)
        elif self.args.opt == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.opt == 'bertadam':
            params = [
                {'params': self.model.bert_encoder.parameters()},
                {'params': self.model.project_trigger.parameters(), 'lr': self.args.lr_other_t},
                {'params': self.model.project_argument.parameters(), 'lr': self.args.lr_other_a},
                {'params': self.model.biaffine.parameters(), 'lr': self.args.lr_other_a},
                {'params': self.model.trig_classifier.parameters(), 'lr': self.args.lr_other_t}
                   ]
            if self.args.use_crf:
                params += [
                        {'params': self.model.crf_t.parameters(), 'lr': self.args.lr_other_t}
                        ]
            if self.args.use_crf_a:
                params += [
                        {'params': self.model.crf_a.parameters(), 'lr': self.args.lr_other_a}
                        ]


            optimizer = AdamW(params, lr=self.args.lr)
            num_training_steps = len(train_data) * self.args.epochs // self.args.iter_size
            scheduler = get_linear_schedule_with_warmup(optimizer,
                    num_warmup_steps=self.args.num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler

        best_eval_f1 = 0.0
        best_eval_f1_t = 0.0
        best_epoch = 0
        step = 0
        patience = 0
        print(self.model)
        for epoch in range(self.args.epochs):
            print()
            print("*"*10+"Training Epoch #%s..." % epoch+"*"*10)

            self.model.train()
            epoch_loss_argu, epoch_loss_column, epoch_loss_t = 0.0, 0.0, 0.0
            n_iter = 0
            # start running batch
            for sent_ids, sents, poss, triggers, arguments, lengths, seq_pairs, all_pairs, glove_idx, orig_to_tok_map, bert_lengths, ent_to_arg_dict in train_data:
                assert self.args.tri_start_epochs < self.args.pipe_epochs
                batch_size = sents.size(0)
                crf_mask=None
                if self.args.use_crf:
                    max_len = lengths[0]
                    crf_mask = torch.arange(max_len).expand(len(lengths), max_len) < torch.LongTensor(lengths).unsqueeze(1)
                if self.args.finetune_bert:
                    bert_max_len = max(bert_lengths)
                    bert_attn_mask = torch.arange(bert_max_len).expand(len(bert_lengths), bert_max_len) < torch.LongTensor(bert_lengths).unsqueeze(1)
                    bert_attn_mask = bert_attn_mask.type(torch.FloatTensor)
                else:
                    bert_attn_mask = None

                if self.args.cuda:
                    sents = sents.cuda()
                    poss = poss.cuda()
                    triggers = triggers.cuda()
                    arguments = arguments.cuda()
                    if glove_idx is not None:
                        glove_idx = glove_idx.cuda()
                    if self.args.finetune_bert and len(bert_attn_mask) > 0:
                        bert_attn_mask = bert_attn_mask.cuda()
                    if self.args.use_crf:
                        crf_mask = crf_mask.cuda()

                self.model.zero_grad()
                # loss = torch.Tensor([0.0]).to(torch.device("cuda" if self.args.cuda else "cpu"))
                loss = 0.0
                ## first predict triggers
                trig_pred, trig_loss, biaffine_tensor, crf_a, trig_emit = self.model(sents, poss, lengths, decode=False, gold_trig_seq=triggers, crf_mask=crf_mask,
                                                                                     glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map, bert_attn_mask=bert_attn_mask)
                if epoch > self.args.tri_start_epochs or epoch < 3:
                    loss += self.args.trigger_weight * trig_loss
                self.logger.scalar_summary(tag='train/norm_tri_loss',
                                           value=trig_loss.item(),
                                           step=step)
                column_loss = self.loss_argu_by_ent(biaffine_tensor, ent_to_arg_dict, lengths)
                if column_loss.item() != 0:
                    loss += self.args.column_weight * column_loss

                ## next predict arguments
                if epoch < self.args.pipe_epochs:
                    sent_ids_ext,sents_ext,poss_ext,args_ext,lengths_ext,tri_idxs,tri_types,n_events,new_pairs =\
                        self.expand_sents(sent_ids, sents, poss, lengths, seq_pairs,
                                          trig_pred, True, False)
                else:
                    sent_ids_ext,sents_ext,poss_ext,args_ext,lengths_ext,tri_idxs,tri_types,n_events,new_pairs =\
                        self.expand_sents(sent_ids, sents, poss, lengths, seq_pairs,
                                          trig_pred, not(self.args.train_on_e2e_data), True)

                argu_loss = self.loss_argu_biaffine(biaffine_tensor, args_ext, tri_idxs, lengths_ext, n_events, batch_size, crf_a)
                if argu_loss.item() != 0:
                    loss += self.args.argument_weight * argu_loss
                self.logger.scalar_summary(tag='train/norm_argu_loss',
                                           value=argu_loss.item(),
                                           step=step)
                ## next gather sentence level argument loss
                if self.args.sent_level_loss:
                    if (epoch < self.args.pipe_epochs) or (self.args.entloss_in_joint):
                        # trig_emit is in (batch, len1, #triglabel), first calculate
                        # the prob that each token's prob to be a trigger. The prob
                        # is used as a weight
                        prob_trig = torch.sum(trig_emit[:, :, 2:].detach(), dim=2).unsqueeze(2).unsqueeze(3) #(batch, len1, 1, 1)
                        biaffine_prob = F.softmax(biaffine_tensor, dim=-1)
                        if self.args.weighted_bytrig:
                            sent_argu = prob_trig*biaffine_prob #(batch, len1, len2, #arglabel)
                        else:
                            sent_argu = biaffine_prob
                        temp1 = sent_argu[:, :, :, 0:2]
                        temp2 = torch.sum(sent_argu[:, :, :, 2:], dim=-1, keepdim=True)
                        sent_argu = torch.cat((temp1, temp2), dim=-1)
                        # Now we aggregate the prob of p(a|x, t) as 0-1 label
                        # Next, we use logsumexp as a softway of "max" operation,
                        # we scale up 100 times so that logsumexp works more
                        # properly.
                        # The idea of MAX operation is because: if either one of the
                        # argument prediction is set to be true, then the token can
                        # be an entity in sentence level.
                        sent_argu_pad = (torch.logsumexp(sent_argu[:,:,:,0]*100, dim=1)/100).unsqueeze(2)
                        sent_argu_true = (torch.logsumexp(sent_argu[:,:,:,2]*100, dim=1)/100).unsqueeze(2)
                        maximum = (torch.logsumexp(torch.Tensor([0.50001]*self.args.batch)*100, dim =0)/100).item()*2
                        sent_argu_false = maximum-sent_argu_pad-sent_argu_true
                        sent_argu_new = torch.cat((sent_argu_pad, sent_argu_false, sent_argu_true), dim=2)
                        eps = 1e-7
                        sent_argu_pred = torch.log(F.relu(sent_argu_new)+eps) #(batch, len2, 3)
                        log_probs_sent = torch.gather(sent_argu_pred, dim=2, index=arguments.unsqueeze(2)).squeeze(2) #(batch, len2)
                        loss_sent = (-torch.sum(log_probs_sent*crf_mask))/(torch.sum(crf_mask).item())
                        loss += self.args.sent_level_weight * loss_sent
                        self.logger.scalar_summary(tag='train/norm_sent_argu_loss',
                                                   value=loss_sent.item(),
                                                   step=step)
                if epoch <= self.args.tri_start_epochs:
                    if argu_loss.item() == 0:
                        # during the no-training trigger phase, if no argument loss either (no events in the batch)
                        # then do not back propogate
                        continue
                loss = loss / self.args.iter_size  # normalize gradients
                loss.backward()
                n_iter += 1   # when backward() was called once, increase n_iter by 1
                if (n_iter) % self.args.iter_size == 0:
                    # accumulate gradients for iter_size batches
                    optimizer.step()
                    if self.args.opt == 'bertadam':
                        scheduler.step()
                    optimizer.zero_grad()
                    self.model.zero_grad()
                if argu_loss.item() != 0:
                    epoch_loss_argu += argu_loss
                if column_loss.item() != 0:
                    epoch_loss_column += column_loss
                epoch_loss_t += trig_loss
                step += 1
            print("Epoch loss trigger: {}, argument: {}, column: {}".format(epoch_loss_t, epoch_loss_argu, epoch_loss_column))
            # Evaluate at the end of each epoch
            print("*"*10+'Evaluating on Dev Set.....'+'*'*10)
            if len(eval_data) > 0:
                f1_t, f1_e, f1_e2es = self._eval_helper(eval_data, epoch, True)
                # f1_t, f1_e, f1_e2es = self._eval_helper(test_data, epoch, False)
                if epoch < self.args.pipe_epochs:
                    #eval_f1 = f1_e
                    eval_f1_a = f1_e2es[-1]
                    eval_f1_t = f1_e2es[1]
                else:
                    eval_f1_a = f1_e2es[-1]  # when entered joint phase, prioritize e2e f1
                    eval_f1_t = f1_e2es[1]

                if epoch == self.args.pipe_epochs:
                    # when entered joint phase(using predicetd triggers)
                    # update the best_f1 to be the new one eval on predicted triggers
                    print('====Entered joint phase.====')
                    #print('Update best_eval_f1 from {} to {}'.format(best_eval_f1, eval_f1))
                    #best_eval_f1 = eval_f1

                # use Arg CLS score to select the argument model
                if eval_f1_a > best_eval_f1:
                    print('Update eval f1 arg: {}'.format(eval_f1_a))
                    best_eval_f1 = eval_f1_a
                    self.best_model_state_dict = {k:v.to('cpu') for k, v in self.model.state_dict().items()}
                    # self.best_model = copy.deepcopy(self.model)
                    self.save(self.args.save_dir+'/best_model.pt', epoch)
                    best_epoch = epoch
                    patience = 0
                else:
                    patience += 1
                # # save best trigger model checkpoint
                # if eval_f1_t > best_eval_f1_t:
                #     print('Update eval f1 tri: {}'.format(eval_f1_t))
                #     best_eval_f1_t = eval_f1_t
                #     self.best_model_t = copy.deepcopy(self.model)
                #     self.save(self.args.save_dir+'/best_model_t.pt', epoch)


                if (epoch > self.args.pipe_epochs) or (self.args.pipe_epochs > 200):
                    if patience > self.args.patience:
                        print('Exceed patience limit. Break at epoch{}'.format(epoch))
                        break

        print("Final Evaluation Best F1: {:.4f} at Epoch {}".format(best_eval_f1, best_epoch))
        print('=====Evaluation on test=====')
        test_f1_t, test_f1_e, test_f1_e2es = self._eval_helper(test_data, 0, False)

        return best_eval_f1, best_epoch, test_f1_t, test_f1_e, test_f1_e2es[-1]

    def train_epoch(self, train_data, dev_data, test_data, task):
        best_f1, best_epoch, test_f1_t, test_f1_e, test_f1_e2e = self._train(train_data, dev_data, test_data, task)
        print("Final Dev F1: %.4f" % best_f1)
        return best_f1, best_epoch, test_f1_t, test_f1_e, test_f1_e2e

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.args
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed ... continuing anyway.]")

    def load(self, filename=None):
        try:
            if self.args.cuda:
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            if filename:
                # multitask model
                checkpoint = torch.load(filename, map_location=device)
                self.model.load_state_dict(checkpoint['model'])
                self.best_model.load_state_dict(checkpoint['model'])
                print("biaffine model load from {}".format(filename))

        except BaseException as e:
            print(e)
            print("Cannot load model from {}".format(filename))
