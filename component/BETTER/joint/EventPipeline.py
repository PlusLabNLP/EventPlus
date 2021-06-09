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
import pdb
from eval import *
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel, BertConfig
from neural_model import BertClassifier
from generate_data.contextualized_features_bert import bert_token



def get_tri_idx_from_mix_sent(out_t, B2I):
    '''
    get trigger idx from a sent with possibly MULTIPLE events
    it is expected that the `out_t` is the predited trigger sequences from model
    it finds the idxs for BIO chunks
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
            curr_I = B2I[out_t[i]]
            if i == len(out_t) - 1:
                # the last token is a B token
                tri_idx.append(curr_idx)
                tri_type.append(curr_I)

    assert len(tri_idx) == len(tri_type)
    return tri_idx, tri_type


def get_expand_trigger_seqs_from_idxs(tri_idxs, tri_types, seq_lens):
    assert len(tri_idxs) == len(seq_lens)
    output = []
    for i in range(len(tri_idxs)):
        tri_seq = [1] * seq_lens[i]
        if len(tri_idxs[i]) == 0:
            # this sent has no predicted triggers
            tri_seq = tri_seq
        # elif len(tri_idxs[i]) == 1:
        #     # single-token triggers
        #     tri_seq[tri_idxs[i][0]] = tri_types[i]
        # elif len(tri_idxs[i]) > 1:
        #     # multi-token triggers
        #     lidx = tri_idxs[i][0]
        #     ridx = tri_idxs[i][-1]
        #     tri_seq[lidx] = tri_types[i]
        #     tri_seq[lidx + 1 : ridx + 1] = [tri_types[i] + 1] * (ridx - lidx)
        else:
            # single-token triggers
            tri_seq[tri_idxs[i][0]] = tri_types[i]


        output.append(tri_seq)
    return output

def tri_partial_match(pred_tri_idx, gold_tri_idx):
    #  seems like NOT including the head equal looks better
    if len(pred_tri_idx) == 1:
        # for predicted single-token trigger, require it to be either inside gold_tri_idx or fall on the boundary
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

def get_prediction_crf(lengths, label, crf_out):
    prediction = [crf_out[i][:l] for i, l in enumerate(lengths)]
    if label is not None:
        label_list = [label[i, :l].tolist() for i, l in enumerate(lengths)]
    else:
        label_list = []
    return label_list, prediction

def get_prediction_mlp(lengths, label, pred_logit):
    prediction = [pred_logit[i, :l] for i, l in enumerate(lengths)] # list of 2d tensor
    prediction = [x.max(dim=1, keepdim=False)[1].tolist() for x in prediction] # list of list
    if label is not None:
        label_list = [label[i, :l].tolist() for i, l in enumerate(lengths)] # list of 1d tensor
    else:
        label_list = []
    return label_list, prediction


class EventPipeline(nn.Module):
    def __init__(self, args, model, model_t=None, model_ner=None):
        super(EventPipeline, self).__init__()
        self.args = args
        self.model = model
        # self.logger = Logger(args.log_dir)
        # self.best_model = model
        self.best_model_t = model_t
        self.best_model_ner = model_ner
        # self.best_model_e = model_e
        # if model_t:
        #     self.best_model_t = model_t
        # if model_e:
        #     self.best_model_e = model_e
        # if self.best_model_t and self.best_model_e:
        #     # dont want the multitask best_model loaded when loading two single models
        #     assert self.best_model is None
        self.dev_cnt = 0
        self.bias_tensor_t = torch.ones(1, 1, len(self.args._id_to_label_t)) * self.args.bias_t
        self.bias_tensor_t[0,0,1] = 1.0  # [0,0,1] corresponds to 'O'
        self.bias_tensor_a = torch.ones(1, 1, len(self.args._id_to_label_e)) * self.args.bias_a
        self.bias_tensor_a[0,0,1] = 1.0  # [0,0,1] corresponds to 'O'
        if self.args.cuda:
            self.bias_tensor_t = self.bias_tensor_t.cuda()
            self.bias_tensor_a = self.bias_tensor_a.cuda()

    def make_trigger_mask(self, argu_cands):
        '''
        The purpose is to set this constraint: If a position is entity, then
        this position cannot be trigger
        `argu_cands` is the Padded, UN-expanded, batch-level Entity sequences
        `n_events` is the num of events in a batch
        '''
        n_tri_class = len(self.args._id_to_label_t)
        bs, seq_len = argu_cands.size()
        out_mask = torch.BoolTensor(np.ones((bs, seq_len, n_tri_class)))
        for b in range(bs):
            for l in range(seq_len):
                if argu_cands[b, l] > 1:
                    # this position is entity, mask out All tri_class except O
                    out_mask[b,l,1] = 0
                elif argu_cands[b, l] == 1:
                    # this position is not entity, None of tri_class should be masked out
                    out_mask[b, l] = 0
        return out_mask

    def make_argu_cands_mask(self, argu_cands, n_events):
        '''
        `argu_cands` is the Padded, UN-expanded, batch-level Entity sequences
        `n_events` is the num of events in a batch
        '''
        n_arg_class = len(self.args._id_to_label_e)
        bs, seq_len = argu_cands.size()
        out_mask = []
        assert len(n_events) == bs
        for i, n_event in enumerate(n_events):
            mask = torch.BoolTensor(np.ones((seq_len, n_arg_class)))
            argu_cand = argu_cands[i]
            for j, argu_tag in enumerate(argu_cand):
                if argu_tag > 1:
                    # if the argu_tag is not O / not <PAD>, then we should not mask this position
                    mask[j] = 0
                elif argu_tag == 1:
                    # if the argu_tag is O, then we should only keep the O position un-masked in the num_class axis
                    mask[j, 1] = 0

            if n_event == 0:
                out_mask.append(mask)
            elif n_event > 0:
                out_mask.extend([mask] * n_event)
        out_mask = torch.cat([x.unsqueeze(0) for x in out_mask], dim=0)
        return out_mask
    def make_valid_argu_roles_mask_by_ent(self, argu_cands, n_events):
        '''
        `argu_cands` is the Padded, UN-expanded, batch-level Entity sequences
        `n_events` is the num of events in a batch
        '''
        n_arg_class = len(self.args._id_to_label_e)
        bs, seq_len = argu_cands.size()
        out_mask = []
        assert len(n_events) == bs
        for i, n_event in enumerate(n_events):
            mask = torch.BoolTensor(np.ones((seq_len, n_arg_class)))
            argu_cand = argu_cands[i]
            for j, argu_tag in enumerate(argu_cand):
                if argu_tag > 1:
                    # if the argu_tag is not O / not <PAD>, then we should not mask this position
                    valid_arg_idxs = self.args._ner_id_to_e_id[argu_tag]
                    mask[j, valid_arg_idxs] = 0
                    # also, keep the 'O' un-masked
                    mask[j, 1] = 0
                elif argu_tag == 1:
                    # if the argu_tag is O, then we should only keep the O position un-masked in the num_class axis
                    mask[j, 1] = 0

            if n_event == 0:
                out_mask.append(mask)
            elif n_event > 0:
                out_mask.extend([mask] * n_event)
        out_mask = torch.cat([x.unsqueeze(0) for x in out_mask], dim=0)
        return out_mask
    def make_valid_argu_roles_mask_by_tri(self, argu_cands, n_events, tri_types):
        '''
        `argu_cands` is the Padded, UN-expanded, batch-level Entity sequences
        `n_events` is the num of events in a batch
        `tri_types` is the trigger types in the batch, used for finding masks of valid arg roles
        '''
        n_arg_class = len(self.args._id_to_label_e)
        bs, seq_len = argu_cands.size()
        out_mask = []
        assert len(n_events) == bs
        cnt = 0
        for i, n_event in enumerate(n_events):
            if n_event == 0:
                # no mask can be applied, force mask to be False ONLY at "O", i.e. mask out all other non-O roles
                mask = torch.BoolTensor(np.ones((seq_len, n_arg_class)))
                argu_cand = argu_cands[i]
                cnt += 1
                for j, argu_tag in enumerate(argu_cand):
                    mask[j, 1] = 0
                out_mask.append(mask)
            elif n_event > 0:
                # in this case, we mask using the valid combs with the given tri_type
                tri_type_sel = tri_types[cnt:cnt+n_event]
                cnt += n_event
                for tri_type in tri_type_sel:
                    mask = torch.BoolTensor(np.ones((seq_len, n_arg_class)))
                    argu_cand = argu_cands[i]
                    for j, argu_tag in enumerate(argu_cand):
                        # mask the valid arg roles for current tri_type
                        valid_arg_idxs = self.args._t_id_to_e_id[tri_type]
                        mask[j, valid_arg_idxs] = 0
                        # also, keep the 'O' un-masked
                        mask[j, 1] = 0
                    out_mask.append(mask)
        out_mask = torch.cat([x.unsqueeze(0) for x in out_mask], dim=0)
        return out_mask

    def predict_pipeline(self, input_sent):
        self.model.load_state_dict(self.best_model_state_dict)
        # self.best_model.eval()
        self.model.eval()
        if self.best_model_t:
            self.best_model_t.eval()
        if self.best_model_ner:
            self.best_model_ner.eval()
        if self.args.cuda:
            self.model.cuda()
            # self.best_model.cuda()
            if self.best_model_t:
                self.best_model_t.cuda()
            if self.best_model_ner:
                self.best_model_ner.cuda()

        y_trues_e, y_preds_e, y_trues_t, y_preds_t = [], [], [], []
        y_trues_ner, y_preds_ner = [], []
        sent_ids_out = []

        # assign a sent id to the input example
        sent_ids = ['test_sample']

        # prepare the bert-large-case tokenzier required by the NER mdoel
        MODELS = [(BertConfig, BertModel, BertTokenizer, 'bert-large-cased')]
        for config_class, model_class, tokenizer_class, pretrained_weights in MODELS:
            config = config_class.from_pretrained(pretrained_weights, output_hidden_states=True)
            bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        sent_bert_tokens, sent_bert_ids, orig_to_tok_map = bert_token(input_sent, bert_tokenizer)
        sents = torch.LongTensor(sent_bert_ids).unsqueeze(0)
        lengths = [len(input_sent)]
        bert_lengths = [len(sent_bert_ids)]

        # crf mask
        max_len = lengths[0]
        crf_mask = torch.arange(max_len).expand(len(lengths), max_len) < torch.LongTensor(lengths).unsqueeze(1)
        # bert mask
        bert_max_len = max(bert_lengths)
        bert_attn_mask = torch.arange(bert_max_len).expand(len(bert_lengths), bert_max_len) < torch.LongTensor(bert_lengths).unsqueeze(1)
        bert_attn_mask = bert_attn_mask.type(torch.FloatTensor)

        if self.args.cuda:
            sents = sents.cuda()
            bert_attn_mask = bert_attn_mask.cuda()
            crf_mask = crf_mask.cuda()

        # NER prediction
        self.args.bert_encode_mthd='max'
        out_ner, _, crf_loss_ner, _ = self.best_model_ner(sents, None, lengths, task='ner',
                                   crf=True, seq_tags=None, crf_mask=crf_mask,
                                   use_att=True, orig_to_tok_map=[orig_to_tok_map], bert_attn_mask=bert_attn_mask)
        ner_label, ner_pred = get_prediction_crf(lengths, None, out_ner)
        y_preds_ner.extend(ner_pred)
        argu_cands = torch.LongTensor(ner_pred)


        # prepare the bert-large-uncase tokenzier required by the tigger mdoel and argument model
        MODELS = [(BertConfig, BertModel, BertTokenizer, 'bert-large-uncased')]
        for config_class, model_class, tokenizer_class, pretrained_weights in MODELS:
            config = config_class.from_pretrained(pretrained_weights, output_hidden_states=True)
            bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        sent_bert_tokens, sent_bert_ids, orig_to_tok_map = bert_token(input_sent, bert_tokenizer)
        sents = torch.LongTensor(sent_bert_ids).unsqueeze(0)
        lengths = [len(input_sent)]
        bert_lengths = [len(sent_bert_ids)]

        # crf mask
        max_len = lengths[0]
        crf_mask = torch.arange(max_len).expand(len(lengths), max_len) < torch.LongTensor(lengths).unsqueeze(1)
        # bert mask
        bert_max_len = max(bert_lengths)
        bert_attn_mask = torch.arange(bert_max_len).expand(len(bert_lengths), bert_max_len) < torch.LongTensor(bert_lengths).unsqueeze(1)
        bert_attn_mask = bert_attn_mask.type(torch.FloatTensor)
        if self.args.cuda:
            sents = sents.cuda()
            bert_attn_mask = bert_attn_mask.cuda()
            crf_mask = crf_mask.cuda()

        # trigger mask with given gold entities - these entities positions will be masked out, so that entities cannot be triggers
        if self.args.decode_w_trigger_mask:
            trigger_mask = self.make_trigger_mask(argu_cands)
            if self.args.cuda:
                trigger_mask = trigger_mask.cuda()
        else:
            trigger_mask = None

        self.args.bert_encode_mthd='head'
        # trigger prediction
        out_t, _, crf_loss_t, _ = self.best_model_t(sents,None, lengths, task='trigger',
                                   crf=True, seq_tags=None, crf_mask=crf_mask,
                                   use_att=self.args.use_att, orig_to_tok_map=[orig_to_tok_map], bert_attn_mask=bert_attn_mask,
                                   trigger_mask=trigger_mask)
        tri_label, tri_pred = get_prediction_crf(lengths, None, out_t)

        # in end2end test case, assume seq_pairs are not given
        sent_ids_ext, sents_ext, args_ext, lengths_ext, bert_lengths_ext, tri_idxs, tri_types, n_events, orig_to_tok_map_ext = \
                                                                               self.expand_sents(sent_ids, sents, lengths, bert_lengths, [orig_to_tok_map],
                                                                                               seq_pairs=None, out_t=tri_pred, use_gold=False, e2e=True)
        tri_pred_ext = get_expand_trigger_seqs_from_idxs(tri_idxs, tri_types, lengths_ext)
        y_preds_t.extend(tri_pred_ext)

        # mask of given gold entities - only these entities positions will NOT be masked, for arg role prediction
        if self.args.decode_w_ents_mask:
            argu_cands_mask = self.make_argu_cands_mask(argu_cands, n_events)
            if self.args.cuda:
                argu_cands_mask = argu_cands_mask.cuda()
        else:
            argu_cands_mask = None
        # mask of valid arg roles - only these valid arg roles for the given TRIGGER types will be NOT be masked
        if self.args.decode_w_arg_role_mask_by_tri:
            argu_roles_mask_by_tri = self.make_valid_argu_roles_mask_by_tri(argu_cands, n_events, tri_types)
            if self.args.cuda:
                argu_roles_mask_by_tri = argu_roles_mask_by_tri.cuda()
        else:
            argu_roles_mask_by_tri = None
        # mask of valid arg roles - only these valid arg roles for the given ENTITY types will be NOT be masked
        if self.args.decode_w_arg_role_mask_by_ent:
            argu_roles_mask_by_ent = self.make_valid_argu_roles_mask_by_ent(argu_cands, n_events)
            if self.args.cuda:
                argu_roles_mask_by_ent = argu_roles_mask_by_ent.cuda()
        else:
            argu_roles_mask_by_ent = None

        # crf mask and bert mask needed by argument model
        bert_max_len = max(bert_lengths_ext)
        bert_attn_mask = torch.arange(bert_max_len).expand(len(bert_lengths_ext), bert_max_len) < torch.LongTensor(bert_lengths_ext).unsqueeze(1)
        bert_attn_mask = bert_attn_mask.type(torch.FloatTensor)
        if self.args.cuda:
            bert_attn_mask = bert_attn_mask.cuda()

        crf_mask_ext = torch.arange(max_len).expand(len(lengths_ext), max_len) < torch.LongTensor(lengths_ext).unsqueeze(1)
        if self.args.cuda:
            crf_mask_ext = crf_mask_ext.cuda()
        out_e, _, crf_loss_e, _ = self.model(sents_ext, None,  lengths_ext,
                                   task='argument', tri_idxs=tri_idxs,
                                   att_pool=self.args.att_pool,
                                   att_mthd=self.args.att_mthd,
                                   crf=True, seq_tags=None,
                                   crf_mask=crf_mask_ext, use_att=self.args.use_att,
                                   orig_to_tok_map=orig_to_tok_map_ext, bert_attn_mask=bert_attn_mask,
                                   argu_cands_mask=argu_cands_mask, argu_roles_mask_by_tri=argu_roles_mask_by_tri,
                                   argu_roles_mask_by_ent=argu_roles_mask_by_ent)
        arg_label, arg_pred = get_prediction_crf(lengths_ext, args_ext, out_e)
        y_preds_e.extend(arg_pred)

        sent_ids_out.extend(sent_ids_ext)
        y_trues_e = [[self.args._id_to_label_e[x] for x in y] for y in y_trues_e]
        y_preds_e = [[self.args._id_to_label_e[x] for x in y] for y in y_preds_e]
        y_trues_t = [[self.args._id_to_label_t[x] for x in y] for y in y_trues_t]
        y_preds_t = [[self.args._id_to_label_t[x] for x in y] for y in y_preds_t]
        y_trues_ner = [[self.args._id_to_label_e_sent[x] for x in y] for y in y_trues_ner]
        y_preds_ner = [[self.args._id_to_label_e_sent[x] for x in y] for y in y_preds_ner]

        return y_trues_t, y_preds_t, y_trues_e, y_preds_e, sent_ids_out, y_trues_ner, y_preds_ner


    def expand_sents(self, sent_ids, sents, lengths, bert_lengths=None, orig_to_tok_map=None, seq_pairs=None, out_t=None, use_gold=True, e2e=False):
        '''
        expand the sents, poss and arguments label tensors, from sentence level
        instancse to event(trigger) level instances for Attention usage

        input:
            sents and poss: tensor of shape (batch_size, max_seq_len)
            triggers: tensor of shape (batch_size, max_seq_len)
            seq_pairs: list of length batch_size, each item is a list of (seq, seq) pairs, length == n_event in this sent
            use_gold is true, e2e is false --> use gold triggers
            use_gold is false, e2e is false --> use gold + predicted triggers
            use_gold is false, e2e is true --> use ONLY predicted triggers
        '''
        sents_ext,  args_ext, sent_ids_ext =  [], [], []
        n_events = []
        lengths_ext = []
        bert_lengths_ext = []
        orig_to_tok_map_ext = []
        tri_idxs = []
        tri_types = []

        if use_gold:
            # gold trigger-arg seq pairs are given
            for i in range(sents.size(0)):  # batch_size
                n_event = len(seq_pairs[i])
                n_events.append(n_event)

                if n_event == 0:
                    # this sentence has no trigger
                    sent_ids_ext.extend([sent_ids[i]])
                    sents_ext.extend([sents[i]])
                    lengths_ext.extend([lengths[i]])
                    if bert_lengths is not None:
                        bert_lengths_ext.extend([bert_lengths[i]])
                        orig_to_tok_map_ext.extend([orig_to_tok_map[i]])
                    # empty trigger idx to be handled by get_query func
                    tri_idxs.append([])
                    tri_types.append([])
                    # the args label is an neg sample, consisting of all "O"s
                    args_ext.append([self.args._label_to_id_e['O']]* lengths[i])
                else:
                    sent_ids_ext.extend([sent_ids[i]] * n_event)
                    sents_ext.extend([sents[i]] * n_event)
                    lengths_ext.extend([lengths[i]] * n_event)
                    if bert_lengths is not None:
                        bert_lengths_ext.extend([bert_lengths[i]] * n_event)
                        orig_to_tok_map_ext.extend([orig_to_tok_map[i]] * n_event)

                    for j in range(n_event):
                        tri_idx, tri_type = get_tri_idx_from_mix_sent(seq_pairs[i][j][0], self.args.B2I_trigger)
                        tri_idxs.append(tri_idx[0]) # tri_idx is a list of idxs, we know this `tri_idxs` is constructed from a SINGLE event
                        tri_types.append(tri_type[0])
                    args_ext.extend([x[1] for x in seq_pairs[i]])
        else:
            assert out_t is not None
            if e2e is False:
                # use gold + predicted triggers
                # first add the additional seq_paris to original seq_pairs
                seq_pairs_ext = []
                for i in range(len(out_t)):   # batch_size
                    tri_idx, tri_type = get_tri_idx_from_mix_sent(out_t[i], self.args.B2I_trigger)
                    seq_len = len(out_t[i])
                    seq_pairs_ext.append(seq_pairs[i])
                    arg_seqs, add_pairs = self.get_arg_seqs_from_seq_pairs(tri_idx, tri_type, seq_pairs[i], seq_len, self.args.B2I_trigger, self.args.tri_partial_match)
                    seq_pairs_ext[i].extend(add_pairs)
                # next, exactly the same as in gold case, difference is use the extended seq pairs
                for i in range(sents.size(0)):  # batch_size
                    n_event = len(seq_pairs_ext[i])
                    n_events.append(n_event)

                    if n_event == 0:
                        # this sentence has no trigger
                        sent_ids_ext.extend([sent_ids[i]])
                        sents_ext.extend([sents[i]])
                        lengths_ext.extend([lengths[i]])
                        if bert_lengths is not None:
                            bert_lengths_ext.extend([bert_lengths[i]])
                            orig_to_tok_map_ext.extend([orig_to_tok_map[i]])
                        # empty trigger idx to be handled by get_query func
                        tri_idxs.append([])
                        tri_types.append([])
                        # the args label is an neg sample, consisting of all "O"s
                        args_ext.append([self.args._label_to_id_e['O']]* lengths[i])
                    else:
                        sent_ids_ext.extend([sent_ids[i]] * n_event)
                        sents_ext.extend([sents[i]] * n_event)
                        lengths_ext.extend([lengths[i]] * n_event)
                        if bert_lengths is not None:
                            bert_lengths_ext.extend([bert_lengths[i]] * n_event)
                            orig_to_tok_map_ext.extend([orig_to_tok_map[i]] * n_event)
                        for j in range(n_event):
                            tri_idx, tri_type = get_tri_idx_from_mix_sent(seq_pairs_ext[i][j][0], self.args.B2I_trigger)
                            tri_idxs.append(tri_idx[0]) # tri_idx is a list of idxs, we know this `tri_idxs` is constructed from a SINGLE event
                            tri_types.append(tri_type[0])
                        args_ext.extend([x[1] for x in seq_pairs_ext[i]])

            else:
                # *ONLY* use predicted triggers
                # but during training need the seq_pairs to assign labels for constructed seq pairs
                for i in range(len(out_t)):   # batch_size
                    tri_idx, tri_type = get_tri_idx_from_mix_sent(out_t[i], self.args.B2I_trigger)
                    n_event = len(tri_idx)
                    n_events.append(n_event)
                    seq_len = len(out_t[i])

                    if n_event == 0:
                        sent_ids_ext.extend([sent_ids[i]])
                        sents_ext.extend([sents[i]])
                        lengths_ext.extend([lengths[i]])
                        if bert_lengths is not None:
                            bert_lengths_ext.extend([bert_lengths[i]])
                            orig_to_tok_map_ext.extend([orig_to_tok_map[i]])
                        # empty trigger idx to be handled by get_query func
                        tri_idxs.append([])
                        tri_types.append([])
                        # the args label is an neg sample, consisting of all "O"s
                        args_ext.append([self.args._label_to_id_e['O']]* lengths[i])
                    else:
                        sent_ids_ext.extend([sent_ids[i]] * n_event)
                        sents_ext.extend([sents[i]] * n_event)
                        lengths_ext.extend([lengths[i]] * n_event)
                        if bert_lengths is not None:
                            bert_lengths_ext.extend([bert_lengths[i]] * n_event)
                            orig_to_tok_map_ext.extend([orig_to_tok_map[i]] * n_event)

                        if seq_pairs is not None:
                            # Only during training will seq_pairs be available, so we can generate argu label arg_ext.
                            arg_seqs, add_output = self.get_arg_seqs_from_seq_pairs(tri_idx, tri_type, seq_pairs[i], seq_len, self.args.B2I_trigger, self.args.tri_partial_match)
                            args_ext.extend(arg_seqs)
                        tri_idxs.extend(tri_idx) # tri_idx is a list of idxs
                        tri_types.extend(tri_type)

        sents_ext = pad_sequence([s for s in sents_ext], batch_first=True, padding_value=TOKEN_PAD_ID)
        if seq_pairs is not None:
            # Only during training will args_ext be available
            args_ext = pad_sequence([torch.LongTensor(s) for s in args_ext], batch_first=True, padding_value=ARGU_PAD_ID)

            assert args_ext.size(0) == sents_ext.size(0)
            assert args_ext.size(0) == len(tri_idxs)
            assert args_ext.size(0) == len(lengths_ext)
        else:
            args_ext = None
        assert len(tri_idxs) == len(tri_types), pdb.set_trace()
        if bert_lengths is not None:
            assert len(bert_lengths_ext) == len(lengths_ext)
            assert len(orig_to_tok_map_ext) == len(lengths_ext)

        return sent_ids_ext, sents_ext, args_ext, lengths_ext, bert_lengths_ext, tri_idxs, tri_types, n_events, orig_to_tok_map_ext

    def get_arg_seqs_from_seq_pairs(self, tri_idxs, tri_types, seq_pairs, seq_len, B2I_trigger, partial_match=False):
        '''
        tri_idxs is like [[2], [4,5], [10]], indicating the trigger idxs (possibly >1 events) in a sent
        seq_pairs is the gold pairs in this sent
        it will return the true arg seq label if the tri_idx corresponds to a gold trigger
                o.w. it returns a seq of "O"s
                the `add_pairs` are additional constructed negative samples (seq pairs).
                Among `add_pairs` pairs, the arg seqs are inclued in the `output`
        '''
        # gold_tri_idxs, gold_arg_seqs = [], []
        gold_tri_idx_to_arg_seq = {}
        output = []
        add_pairs = []
        # seq_len = len(seq_pairs[0][0])

        # construct helper dict
        for i in range(len(seq_pairs)):
            gold_tri_idx, gold_tri_type = get_tri_idx_from_mix_sent(seq_pairs[i][0], B2I_trigger)
            gold_tri_idx_to_arg_seq[tuple(gold_tri_idx[0])] = seq_pairs[i][1]

        for i, t in enumerate(tri_idxs):
            tri_type = tri_types[i]
            if partial_match is False:
                if gold_tri_idx_to_arg_seq.get(tuple(t)):
                    output.append(gold_tri_idx_to_arg_seq.get(tuple(t)))
                else:
                    # a seq of "O"s, as a neg sample
                    output.append([1] * seq_len)
                    tri_seq = get_expand_trigger_seqs_from_idxs([t], [tri_type],[seq_len])[0]
                    add_pairs.append((tri_seq, [1] * seq_len))
            elif partial_match is True:
                find_flag=False
                for gold_tri_idx in list(gold_tri_idx_to_arg_seq.keys()):
                    match = tri_partial_match(tuple(t), gold_tri_idx)
                    if match is True:
                        output.append(gold_tri_idx_to_arg_seq.get(gold_tri_idx))
                        find_flag = True
                        break
                if find_flag is False:
                    # a seq of "O"s, as a neg sample
                    output.append([1] * seq_len)
                    tri_seq = get_expand_trigger_seqs_from_idxs([t], [tri_type],[seq_len])[0]
                    add_pairs.append((tri_seq, [1] * seq_len))
        return output, add_pairs


    def save(self, filename, epoch):
        params = {
            'model': self.model.module.state_dict() if torch.cuda.device_count() > 1 and self.args.multigpu else self.model.state_dict(),
            'config': self.args
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed ... continuing anyway.]")

    def load(self, filename=None, filename_t=None, filename_ner=None):
        try:
            if self.args.cuda:
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            if filename:
                # multitask model
                checkpoint = torch.load(filename, map_location=device)
                self.model.load_state_dict(checkpoint['model'])
                # self.best_model.load_state_dict(checkpoint['model'])
                self.best_model_state_dict = checkpoint['model']
                print("multitask model load from {}".format(filename))
            if filename_t:
                # trigger model
                checkpoint = torch.load(filename_t, map_location=device)
                # self.model.load_state_dict(checkpoint['model'])
                self.best_model_t.load_state_dict(checkpoint['model'])
                # self.best_model_state_dict = checkpoint['model']
                print("trigger model load from {}".format(filename_t))
            if filename_ner:
                # arg model
                checkpoint = torch.load(filename_ner, map_location=device)
                # self.model.load_state_dict(checkpoint['model'])
                # self.best_model_e.load_state_dict(checkpoint['model'])
                # self.best_model_state_dict = checkpoint['model']
                self.best_model_ner.load_state_dict(checkpoint['model'])
                print("NER model load from {}".format(filename_ner))
        except BaseException as e:
            print(e)
            print("Cannot load model from {}".format(filename))
