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
from neural_model import BertClassifier


def write_pkl(sent_ids, y_trues_t, y_preds_t, y_trues_e, y_preds_e, y_trues_e_sent, y_preds_e_sent, out_pkl_dir, pkl_name):
    #ensure dir
    if not os.path.exists(out_pkl_dir):
        os.makedirs(out_pkl_dir)

    if len(y_trues_e) == 0 and len(y_preds_e) > 0:
        # meaning this is the final_test case, i.e. end2end for event-level arg cls
        output_t_e_event = []
        # output_e_event = []
        for i in range(len(sent_ids)):
            output_t_e_event.append({'sent_id': sent_ids[i], 'pred_trigger': y_preds_t[i], 'pred_arg': y_preds_e[i]})
        with open('{}/{}'.format(out_pkl_dir, pkl_name), 'wb') as f:
            pickle.dump(output_t_e_event, f)
        print('pkl saved at {}/{}'.format(out_pkl_dir, pkl_name))


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


class NNClassifier(nn.Module):
    def __init__(self, args, model, model_t=None, model_ner=None):
        super(NNClassifier, self).__init__()
        self.args = args
        self.model = model
        self.logger = Logger(args.log_dir)
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


    def predict(self, data, train_task, test=False, use_gold=True, e2e_eval=False, generate_eval_file=False):
        '''
        when e2e_eval is True, it will return empty y_trues_t, y_trues_e, y_trues_sent
        '''
        if test:
            self.model.load_state_dict(self.best_model_state_dict)
            # self.best_model.eval()
            self.model.eval()
            if self.best_model_t:
                self.best_model_t.eval()
            if self.best_model_ner:
                self.best_model_ner.eval()
        if self.args.cuda:
            self.model.cuda()
            if test:
                # self.best_model.cuda()
                if self.best_model_t:
                    self.best_model_t.cuda()
                if self.best_model_ner:
                    self.best_model_ner.cuda()

        count = 1
        y_trues_e, y_preds_e, y_trues_t, y_preds_t = [], [], [], []
        y_trues_ner, y_preds_ner = [], []
        y_trues_e_sent, y_preds_e_sent = [], []
        sent_ids_out = []

        for sent_ids, sents, poss, triggers, argu_cands, lengths, seq_pairs, all_pairs, glove_idx, orig_to_tok_map, bert_lengths, ent_to_arg_dict in data:
            if self.args.use_crf_t:
                max_len = lengths[0]
                crf_mask = torch.arange(max_len).expand(len(lengths), max_len) < torch.LongTensor(lengths).unsqueeze(1)
            if self.args.finetune_bert:
                bert_max_len = max(bert_lengths)
                bert_attn_mask = torch.arange(bert_max_len).expand(len(bert_lengths), bert_max_len) < torch.LongTensor(bert_lengths).unsqueeze(1)
                bert_attn_mask = bert_attn_mask.type(torch.FloatTensor)

            if self.args.cuda:
                sents = sents.cuda()
                poss = poss.cuda()
                argu_cands = argu_cands.cuda()
                triggers = triggers.cuda()
                if self.args.use_glove and glove_idx is None:
                    glove_idx = glove_idx.cuda()
                if self.args.finetune_bert and len(bert_attn_mask) > 0:
                    bert_attn_mask = bert_attn_mask.cuda()
                if self.args.use_crf_t:
                    crf_mask = crf_mask.cuda()

            ## first NER
            if train_task in ['singletask_ner']:
                if self.args.use_crf_ner:
                    if test:
                        if self.best_model_ner:
                            out_ner, _, crf_loss_ner, _ = self.best_model_ner(sents, poss, lengths, task='ner',
                                                       crf=True, seq_tags=None, crf_mask=crf_mask,
                                                       use_att=self.args.use_att,
                                                       bias_tensor_t=None, bias_tensor_a=None,
                                                       glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map, bert_attn_mask=bert_attn_mask)
                        else:
                            out_ner, _, crf_loss_ner, _ = self.model(sents, poss, lengths, task='ner',
                                                       crf=True, seq_tags=None, crf_mask=crf_mask,
                                                       use_att=self.args.use_att,
                                                       bias_tensor_t=None, bias_tensor_a=None,
                                                       glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map, bert_attn_mask=bert_attn_mask)
                    else:
                        out_ner, _, crf_loss_ner, _ = self.model(sents, poss, lengths, task='ner',
                                                       crf=True, seq_tags=argu_cands,
                                                       crf_mask=crf_mask, use_att=self.args.use_att,
                                                       bias_tensor_t=None, bias_tensor_a=None,
                                                       glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map, bert_attn_mask=bert_attn_mask)
                        loss_ner = -1.0 * crf_loss_ner
                    ner_label, ner_pred = get_prediction_crf(lengths, argu_cands, out_ner)
                    if not e2e_eval:
                        y_trues_ner.extend(ner_label)
                        y_preds_ner.extend(ner_pred)
                else:
                    if test:
                        out_ner, prob_ner = self.model(sents, poss, lengths, task='ner', use_att=self.args.use_att,
                                                        bias_tensor_t=None, bias_tensor_a=None,
                                                        glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map, bert_attn_mask=bert_attn_mask)
                    else:
                        out_ner, prob_ner = self.model(sents, poss, lengths, task='ner', use_att=self.args.use_att,
                                                   bias_tensor_t=None, bias_tensor_a=None,
                                                   glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map, bert_attn_mask=bert_attn_mask)
                        loss_ner = get_loss_mlp(lengths, argu_cands, out_ner, nn.CrossEntropyLoss())
                    ner_label, ner_pred = get_prediction_mlp(lengths, argu_cands, out_ner)
                    if not e2e_eval:
                        y_trues_ner.extend(ner_label)
                        y_preds_ner.extend(ner_pred)
                if not test:
                    self.logger.scalar_summary(tag='dev/norm_ner_loss',
                                               value=loss_ner,
                                               step=self.dev_cnt)

            ## first predict triggers
            if train_task in ['multitask_pipe', 'multitask_joint', 'singletask_trigger']:
                # trigger mask with given gold entities - these entities positions will be masked out, so that entities cannot be triggers
                if self.args.decode_w_trigger_mask:
                    trigger_mask = self.make_trigger_mask(argu_cands)
                    if self.args.cuda:
                        trigger_mask = trigger_mask.cuda()
                else:
                    trigger_mask = None
                if self.args.use_crf_t:
                    if test:
                        if self.best_model_t:
                            out_t, _, crf_loss_t, _ = self.best_model_t(sents, poss, lengths, task='trigger',
                                                       crf=True, seq_tags=None, crf_mask=crf_mask,
                                                       use_att=self.args.use_att,
                                                       bias_tensor_t=self.bias_tensor_t, bias_tensor_a=self.bias_tensor_a,
                                                       glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map, bert_attn_mask=bert_attn_mask,
                                                       trigger_mask=trigger_mask)
                        else:
                            out_t, _, crf_loss_t, _ = self.model(sents, poss, lengths, task='trigger',
                                                       crf=True, seq_tags=None, crf_mask=crf_mask,
                                                       use_att=self.args.use_att,
                                                       bias_tensor_t=self.bias_tensor_t, bias_tensor_a=self.bias_tensor_a,
                                                       glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map, bert_attn_mask=bert_attn_mask,
                                                       trigger_mask=trigger_mask)
                    else:
                        out_t, _, crf_loss_t, _ = self.model(sents, poss, lengths, task='trigger',
                                                       crf=True, seq_tags=triggers,
                                                       crf_mask=crf_mask, use_att=self.args.use_att,
                                                       bias_tensor_t=self.bias_tensor_t, bias_tensor_a=self.bias_tensor_a,
                                                       glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map, bert_attn_mask=bert_attn_mask,
                                                       trigger_mask=trigger_mask)
                        loss_t = -1.0 * crf_loss_t
                    tri_label, tri_pred = get_prediction_crf(lengths, triggers, out_t)
                    if not e2e_eval:
                        y_trues_t.extend(tri_label)
                        y_preds_t.extend(tri_pred)
                else:
                    if test:
                        out_t, prob_t = self.model(sents, poss, lengths, task='trigger', use_att=self.args.use_att,
                                                        bias_tensor_t=self.bias_tensor_t, bias_tensor_a=self.bias_tensor_a,
                                                        glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map, bert_attn_mask=bert_attn_mask)
                    else:
                        out_t, prob_t = self.model(sents, poss, lengths, task='trigger', use_att=self.args.use_att,
                                                   bias_tensor_t=self.bias_tensor_t, bias_tensor_a=self.bias_tensor_a,
                                                   glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map, bert_attn_mask=bert_attn_mask)
                        loss_t = get_loss_mlp(lengths, triggers, out_t, nn.CrossEntropyLoss())
                    tri_label, tri_pred = get_prediction_mlp(lengths, triggers, out_t)
                    if not e2e_eval:
                        y_trues_t.extend(tri_label)
                        y_preds_t.extend(tri_pred)
                if not test:
                    self.logger.scalar_summary(tag='dev/norm_tri_loss',
                                               value=loss_t,
                                               step=self.dev_cnt)

            ## next predict arguments
            if train_task in ['multitask_pipe', 'multitask_joint', 'singletask_arg']:
                if train_task == 'singletask_arg':
                    tri_pred=None
                if self.args.use_att:
                    if use_gold is False and e2e_eval is True:
                        # in end2end test case, assume seq_pairs are not given
                        sent_ids_ext, sents_ext, poss_ext, args_ext, lengths_ext, bert_lengths_ext, tri_idxs, tri_types, n_events, orig_to_tok_map_ext = \
                                                                                               self.expand_sents(sent_ids, sents, poss, lengths, bert_lengths, orig_to_tok_map,
                                                                                               seq_pairs=None, out_t=tri_pred, use_gold=use_gold, e2e=e2e_eval)
                    else:
                        sent_ids_ext, sents_ext, poss_ext, args_ext, lengths_ext, bert_lengths_ext, tri_idxs, tri_types, n_events, orig_to_tok_map_ext = \
                                                                                               self.expand_sents(sent_ids, sents, poss, lengths, bert_lengths, orig_to_tok_map,
                                                                                               seq_pairs, out_t=tri_pred, use_gold=use_gold, e2e=e2e_eval)
                    if e2e_eval:
                        tri_pred_ext = get_expand_trigger_seqs_from_idxs(tri_idxs, tri_types, lengths_ext)
                        if not generate_eval_file:
                            y_preds_t.extend(tri_pred_ext)
                        else:
                            assert use_gold is True
                            ############
                            y_trues_t.extend(tri_pred_ext)
                            ############
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
                else:
                    sents_ext, poss_ext, args_ext, lengths_ext = sents, poss, argu_cands, lengths
                    tri_idxs=[]
                    n_events=[]
                if self.args.finetune_bert:
                    bert_max_len = max(bert_lengths_ext)
                    bert_attn_mask = torch.arange(bert_max_len).expand(len(bert_lengths_ext), bert_max_len) < torch.LongTensor(bert_lengths_ext).unsqueeze(1)
                    bert_attn_mask = bert_attn_mask.type(torch.FloatTensor)
                    if self.args.cuda:
                        bert_attn_mask = bert_attn_mask.cuda()

                if self.args.cuda:
                    if args_ext is not None:
                        args_ext = args_ext.cuda()
                if self.args.use_crf_a:
                    crf_mask_ext = torch.arange(max_len).expand(len(lengths_ext), max_len) < torch.LongTensor(lengths_ext).unsqueeze(1)
                    if self.args.cuda:
                        crf_mask_ext = crf_mask_ext.cuda()
                    if test:
                        if self.best_model_t:
                            out_e, _, crf_loss_e, _ = self.model(sents_ext, poss_ext, lengths_ext,
                                                       task='argument', tri_idxs=tri_idxs,
                                                       att_pool=self.args.att_pool,
                                                       att_mthd=self.args.att_mthd,
                                                       crf=True, seq_tags=None,
                                                       crf_mask=crf_mask_ext, use_att=self.args.use_att,
                                                       bias_tensor_t=self.bias_tensor_t, bias_tensor_a=self.bias_tensor_a,
                                                       glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map_ext, bert_attn_mask=bert_attn_mask,
                                                       argu_cands_mask=argu_cands_mask, argu_roles_mask_by_tri=argu_roles_mask_by_tri,
                                                       argu_roles_mask_by_ent=argu_roles_mask_by_ent)
                        else:
                            out_e, _, crf_loss_e, _ = self.model(sents_ext, poss_ext, lengths_ext,
                                                       task='argument', tri_idxs=tri_idxs,
                                                       att_pool=self.args.att_pool,
                                                       att_mthd=self.args.att_mthd,
                                                       crf=True, seq_tags=None,
                                                       crf_mask=crf_mask_ext, use_att=self.args.use_att,
                                                       bias_tensor_t=self.bias_tensor_t, bias_tensor_a=self.bias_tensor_a,
                                                       glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map_ext, bert_attn_mask=bert_attn_mask,
                                                       argu_cands_mask=argu_cands_mask, argu_roles_mask_by_tri=argu_roles_mask_by_tri,
                                                       argu_roles_mask_by_ent=argu_roles_mask_by_ent)
                    else:
                        out_e, _, crf_loss_e, _ = self.model(sents_ext, poss_ext, lengths_ext,
                                                             task='argument', tri_idxs=tri_idxs,
                                                             att_pool=self.args.att_pool,
                                                             att_mthd=self.args.att_mthd,
                                                             crf=True, seq_tags=args_ext,
                                                             crf_mask=crf_mask_ext, use_att=self.args.use_att,
                                                             bias_tensor_t=self.bias_tensor_t, bias_tensor_a=self.bias_tensor_a,
                                                             glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map_ext, bert_attn_mask=bert_attn_mask,
                                                             argu_cands_mask=argu_cands_mask, argu_roles_mask_by_tri=argu_roles_mask_by_tri,
                                                             argu_roles_mask_by_ent=argu_roles_mask_by_ent)
                        if use_gold is True:
                            # only in the non-end2end case, we compute loss
                            loss_e = -1.0 * crf_loss_e
                    arg_label, arg_pred = get_prediction_crf(lengths_ext, args_ext, out_e)
                else:
                    if test:
                        out_e, prob_e = self.model(sents_ext, poss_ext, lengths_ext,
                                                        task='argument', tri_idxs=tri_idxs,
                                                        att_pool=self.args.att_pool,
                                                        att_mthd=self.args.att_mthd,
                                                        use_att=self.args.use_att,
                                                        bias_tensor_t=self.bias_tensor_t, bias_tensor_a=self.bias_tensor_a,
                                                        glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map_ext,bert_attn_mask=bert_attn_mask,
                                                        argu_cands_mask=argu_cands_mask, argu_roles_mask_by_tri=argu_roles_mask_by_tri,
                                                        argu_roles_mask_by_ent=argu_roles_mask_by_ent)
                    else:
                        out_e, prob_e = self.model(sents_ext, poss_ext, lengths_ext,
                                                   task='argument', tri_idxs=tri_idxs,
                                                   att_pool=self.args.att_pool,
                                                   att_mthd=self.args.att_mthd,
                                                   use_att=self.args.use_att,
                                                   bias_tensor_t=self.bias_tensor_t, bias_tensor_a=self.bias_tensor_a,
                                                   glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map_ext, bert_attn_mask=bert_attn_mask,
                                                   argu_cands_mask=argu_cands_mask, argu_roles_mask_by_tri=argu_roles_mask_by_tri,
                                                   argu_roles_mask_by_ent=argu_roles_mask_by_ent)
                        if use_gold is True:
                            loss_e = get_loss_mlp(lengths_ext, args_ext, out_e, nn.CrossEntropyLoss())
                    arg_label, arg_pred = get_prediction_mlp(lengths_ext, args_ext, out_e)
                if e2e_eval is False:
                    y_trues_e.extend(arg_label)
                if generate_eval_file is False:
                    y_preds_e.extend(arg_pred)
                else:
                    #######
                    y_trues_e.extend(arg_label)
                    #######
                if test is False and use_gold is True:
                    # only in the non-end2ed case, we compute loss
                    self.logger.scalar_summary(tag='dev/norm_argu_loss',
                                               value=loss_e,
                                               step=self.dev_cnt)

                # if self.args.use_att:
                if self.args.eval_sent_arg_role:
                    ## need to generate the sent level label
                    arg_label_sent = [argu_cands[i, :l].tolist() for i, l in enumerate(lengths)]
                    arg_pred_sent = self.merge_sent_level_label(arg_pred, n_events)
                    arg_label_sent = [[self.args._id_to_label_e_sent[x] for x in y] for y in arg_label_sent]
                    arg_pred_sent = [[self.args._id_to_label_e_sent[x] for x in y] for y in arg_pred_sent]
                    y_trues_e_sent.extend(arg_label_sent)
                    y_preds_e_sent.extend(arg_pred_sent)

            if train_task in ['multitask_pipe', 'multitask_joint', 'singletask_arg']:
                sent_ids_out.extend(sent_ids_ext)
            elif train_task in ['singletask_ner']:
                sent_ids_out.extend(sent_ids)

        y_trues_e = [[self.args._id_to_label_e[x] for x in y] for y in y_trues_e]
        y_preds_e = [[self.args._id_to_label_e[x] for x in y] for y in y_preds_e]
        y_trues_t = [[self.args._id_to_label_t[x] for x in y] for y in y_trues_t]
        y_preds_t = [[self.args._id_to_label_t[x] for x in y] for y in y_preds_t]
        y_trues_ner = [[self.args._id_to_label_e_sent[x] for x in y] for y in y_trues_ner]
        y_preds_ner = [[self.args._id_to_label_e_sent[x] for x in y] for y in y_preds_ner]
        if not test:
            self.dev_cnt += 1
        return y_trues_t, y_preds_t, y_trues_e, y_preds_e, y_trues_e_sent, y_preds_e_sent, sent_ids_out, y_trues_ner, y_preds_ner

    def expand_sents(self, sent_ids, sents, poss, lengths, bert_lengths=None, orig_to_tok_map=None, seq_pairs=None, out_t=None, use_gold=True, e2e=False):
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
        sents_ext, poss_ext, args_ext, sent_ids_ext = [], [], [], []
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
                    poss_ext.extend([poss[i]])
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
                    poss_ext.extend([poss[i]] * n_event)
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
                        poss_ext.extend([poss[i]])
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
                        poss_ext.extend([poss[i]] * n_event)
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
                        poss_ext.extend([poss[i]])
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
                        poss_ext.extend([poss[i]] * n_event)
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

                # pdb.set_trace()

        sents_ext = pad_sequence([s for s in sents_ext], batch_first=True, padding_value=TOKEN_PAD_ID)
        poss_ext = pad_sequence([s for s in poss_ext], batch_first=True, padding_value=POS_PAD_ID)
        if seq_pairs is not None:
            # Only during training will args_ext be available
            args_ext = pad_sequence([torch.LongTensor(s) for s in args_ext], batch_first=True, padding_value=ARGU_PAD_ID)

            assert args_ext.size(0) == poss_ext.size(0)
            assert args_ext.size(0) == sents_ext.size(0)
            assert args_ext.size(0) == len(tri_idxs)
            assert args_ext.size(0) == len(lengths_ext)
        else:
            args_ext = None
        assert len(tri_idxs) == len(tri_types), pdb.set_trace()
        if bert_lengths is not None:
            assert len(bert_lengths_ext) == len(lengths_ext)
            assert len(orig_to_tok_map_ext) == len(lengths_ext)

        return sent_ids_ext, sents_ext, poss_ext, args_ext, lengths_ext, bert_lengths_ext, tri_idxs, tri_types, n_events, orig_to_tok_map_ext

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

    def merge_sent_level_label(self, ys, n_events):
        merged_ys = []
        cnt = 0
        for i in range(len(n_events)):
            if n_events[i] == 0:
                merged_ys.append([self.args._label_to_id_e['O']] * len(ys[cnt]))
                cnt += 1
                continue
            sent_ys = ys[cnt : cnt + n_events[i]]
            sent_y_merged = []
            assert len(set([len(x) for x in sent_ys])) == 1
            sent_len = len(sent_ys[0])
            sent_m = np.concatenate([np.array(y).reshape(1, -1) for y in sent_ys], axis=0)
            for j in range(sent_len):
                uniq_values = list(np.unique(sent_m[:, j]))
                if len(uniq_values) == 1:
                    sent_y_merged.append(uniq_values[0])
                else:
                    if 2 in uniq_values:
                        # heuristic: priotize 2
                        sent_y_merged.append(2)
                    elif 3 in uniq_values:
                        sent_y_merged.append(3)
                    elif 1 in uniq_values:
                        sent_y_merged.append(1)
                    elif 0 in uniq_values:
                        sent_y_merged.append(0)
                    else:
                        print('Uncovered situation!!!!!!!')
                        assert False
            assert len(sent_y_merged) == sent_len
            merged_ys.append(sent_y_merged)
            cnt += n_events[i]
        return merged_ys

    def _train(self, train_data, eval_data, test_data, train_task):
        if self.args.cuda:
            print("using cuda device: %s" % torch.cuda.current_device())
            assert torch.cuda.is_available()
            self.model.cuda()

        criterion_ner = nn.CrossEntropyLoss() #NER
        criterion_t = nn.CrossEntropyLoss() #trigger
        criterion_e = nn.CrossEntropyLoss() #argument

        if self.args.opt == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr)
        elif self.args.opt == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.opt == 'bertadam':
            if self.args.multigpu:
                params = [
                        {'params': self.model.module.bert_encoder.parameters(), 'lr': self.args.lr},
                        {'params': self.model.module.linear1_tri.parameters(), 'lr': self.args.lr_other_t},
                        {'params': self.model.module.linear2_tri.parameters(), 'lr': self.args.lr_other_t},
                        {'params': self.model.module.linear1_arg.parameters(), 'lr': self.args.lr_other_a},
                        {'params': self.model.module.linear2_arg.parameters(), 'lr': self.args.lr_other_a}
                       ]
                if self.args.use_crf_ner:
                    params += [
                            {'params': self.model.module.crf_ner.parameters(), 'lr': self.args.lr_other_ner}
                            ]
                if self.args.use_crf_t:
                    params += [
                            {'params': self.model.module.crf_t.parameters(), 'lr': self.args.lr_other_t}
                            ]
                if self.args.use_crf_a:
                    params += [
                            {'params': self.model.module.crf_e.parameters(), 'lr': self.args.lr_other_a}
                            ]
            else:
                params = [
                        {'params': self.model.bert_encoder.parameters(), 'lr': self.args.lr},
                        {'params': self.model.linear1_tri.parameters(), 'lr': self.args.lr_other_t},
                        {'params': self.model.linear2_tri.parameters(), 'lr': self.args.lr_other_t},
                        {'params': self.model.linear1_arg.parameters(), 'lr': self.args.lr_other_a},
                        {'params': self.model.linear2_arg.parameters(), 'lr': self.args.lr_other_a}
                       ]
                if self.args.use_crf_ner:
                    params += [
                            {'params': self.model.crf_ner.parameters(), 'lr': self.args.lr_other_ner}
                            ]
                if self.args.use_crf_t:
                    params += [
                            {'params': self.model.crf_t.parameters(), 'lr': self.args.lr_other_t}
                            ]
                if self.args.use_crf_a:
                    params += [
                            {'params': self.model.crf_e.parameters(), 'lr': self.args.lr_other_a}
                            ]
            # params = [
            #         {'params': self.model.bert_encoder.parameters(), 'lr': self.args.lr, 'weight_decay': 1e-5},
            #         {'params': self.model.linear1_ner.parameters(), 'lr': self.args.lr_other_ner, 'weight_decay': 1e-3},
            #         {'params': self.model.linear2_ner.parameters(), 'lr': self.args.lr_other_ner, 'weight_decay': 1e-3},
            #         {'params': self.model.linear1_tri.parameters(), 'lr': self.args.lr_other_t, 'weight_decay': 1e-3},
            #         {'params': self.model.linear2_tri.parameters(), 'lr': self.args.lr_other_t, 'weight_decay': 1e-3},
            #         {'params': self.model.linear1_arg.parameters(), 'lr': self.args.lr_other_a, 'weight_decay': 1e-3},
            #         {'params': self.model.linear2_arg.parameters(), 'lr': self.args.lr_other_a, 'weight_decay': 1e-3}
            #        ]
            # if self.args.use_crf_ner:
            #     params += [
            #             {'params': self.model.crf_ner.parameters(), 'lr': self.args.lr_other_ner, 'weight_decay': 1e-3}
            #             ]
            # if self.args.use_crf_t:
            #     params += [
            #             {'params': self.model.crf_t.parameters(), 'lr': self.args.lr_other_t, 'weight_decay': 1e-3}
            #             ]
            # if self.args.use_crf_a:
            #     params += [
            #             {'params': self.model.crf_e.parameters(), 'lr': self.args.lr_other_a, 'weight_decay': 1e-3}
            #             ]

            optimizer = AdamW(params, lr=self.args.lr)
            num_training_steps = len(train_data) * self.args.epochs // self.args.iter_size
            scheduler = get_linear_schedule_with_warmup(optimizer,
                    num_warmup_steps=self.args.num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler


        best_eval_f1 = 0.0
        best_epoch = 0
        step = 0
        patience = 0



        print(self.model)
        for epoch in range(self.args.epochs):
            print()
            print("*"*10+"Training Epoch #%s..." % epoch+"*"*10)

            self.model.train()
            loss_hist_e, loss_hist_t, loss_hist_ner = [], [], []
            epoch_loss_e = 0
            epoch_loss_t = 0
            epoch_loss_ner = 0
            self.model.zero_grad()
            # start running batch
            for n_iter, (sent_ids, sents, poss, triggers, argu_cands, lengths, seq_pairs, all_pairs, glove_idx, orig_to_tok_map, bert_lengths, ent_to_arg_dict) in enumerate(train_data, 1):
                if self.args.use_crf_t:
                    max_len = lengths[0]
                    crf_mask = torch.arange(max_len).expand(len(lengths), max_len) < torch.LongTensor(lengths).unsqueeze(1)
                if self.args.finetune_bert:
                    bert_max_len = max(bert_lengths)
                    bert_attn_mask = torch.arange(bert_max_len).expand(len(bert_lengths), bert_max_len) < torch.LongTensor(bert_lengths).unsqueeze(1)
                    bert_attn_mask = bert_attn_mask.type(torch.FloatTensor)

                if self.args.cuda:
                    sents = sents.cuda()
                    poss = poss.cuda()
                    triggers = triggers.cuda()
                    argu_cands = argu_cands.cuda()
                    if self.args.use_glove and glove_idx is None:
                        glove_idx = glove_idx.cuda()
                    if self.args.finetune_bert and len(bert_attn_mask) > 0:
                        bert_attn_mask = bert_attn_mask.cuda()
                    if self.args.use_crf_t:
                        crf_mask = crf_mask.cuda()

                # self.model.zero_grad()
                loss = 0.
                loss_ner = torch.Tensor([0.])
                loss_t = torch.Tensor([0.])
                loss_e = torch.Tensor([0.])
                ## NER
                if train_task in ['singletask_ner']:
                    if self.args.use_crf_ner:
                        out_ner, _, crf_loss_ner, _ = self.model(sents, poss, lengths, task='ner',
                                                       crf=True, seq_tags=argu_cands,
                                                       crf_mask=crf_mask, use_att=self.args.use_att,
                                                       bias_tensor_t=None, bias_tensor_a=None,
                                                       glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map, bert_attn_mask=bert_attn_mask)
                        # pytorch-crf return log likelihood, not neg log likelihood
                        loss_ner = -1.0 * crf_loss_ner
                    else:
                        out_ner, prob_ner = self.model(sents, poss, lengths, task='ner', use_att=self.args.use_att,
                                                   bias_tensor_t=None, bias_tensor_a=None,
                                                   glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map,bert_attn_mask=bert_attn_mask)
                        loss_ner = get_loss_mlp(lengths, argu_cands, out_ner, criterion_ner)
                    loss += self.args.ner_weight * loss_ner
                    self.logger.scalar_summary(tag='train/norm_ner_loss',
                                               value=loss_ner.item(),
                                               step=step)
                ## first predict triggers
                if train_task in ['multitask_pipe', 'multitask_joint', 'singletask_trigger'] and \
                   epoch > self.args.tri_start_epochs:
                    if self.args.use_crf_t:
                        out_t, _, crf_loss_t, _ = self.model(sents, poss, lengths, task='trigger',
                                                       crf=True, seq_tags=triggers,
                                                       crf_mask=crf_mask, use_att=self.args.use_att,
                                                       bias_tensor_t=self.bias_tensor_t, bias_tensor_a=self.bias_tensor_a,
                                                       glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map, bert_attn_mask=bert_attn_mask)
                        # pytorch-crf return log likelihood, not neg log likelihood
                        loss_t = -1.0 * crf_loss_t
                    else:
                        out_t, prob_t = self.model(sents, poss, lengths, task='trigger', use_att=self.args.use_att,
                                                   bias_tensor_t=self.bias_tensor_t, bias_tensor_a=self.bias_tensor_a,
                                                   glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map,bert_attn_mask=bert_attn_mask)
                        loss_t = get_loss_mlp(lengths, triggers, out_t, criterion_t)
                    if epoch > self.args.tri_start_epochs:
                        loss += self.args.trigger_weight * loss_t
                    self.logger.scalar_summary(tag='train/norm_tri_loss',
                                               value=loss_t.item(),
                                               step=step)

                if orig_to_tok_map is None:
                    pdb.set_trace()

                ## next predict arguments
                if train_task in ['multitask_pipe', 'multitask_joint','singletask_arg']:
                    use_gold = True
                    e2e = False
                    tri_pred=None
                    if epoch > self.args.pipe_epochs:
                        _, tri_pred = get_prediction_crf(lengths, triggers, out_t)
                        use_gold = False
                        if self.args.train_on_e2e_data:
                            e2e = True
                        else:
                            e2e = False
                    ## to use attention, need to first expand sents and corresponding args label to event(trigger) level
                    if self.args.use_att:
                        sent_ids_ext, sents_ext, poss_ext, args_ext, lengths_ext, bert_lengths_ext, tri_idxs, tri_types, n_events, orig_to_tok_map_ext = \
                                                                                               self.expand_sents(sent_ids, sents, poss, lengths, bert_lengths, orig_to_tok_map,
                                                                                               seq_pairs, out_t=tri_pred, use_gold=use_gold, e2e=e2e)
                        # sents_ext = sents_ext.detach()
                        # poss_ext = poss_ext.detach()
                        # args_ext = args_ext.detach()
                        if orig_to_tok_map is None or orig_to_tok_map_ext is None:
                            pdb.set_trace()
                    else:
                        sent_ids_ext, sents_ext, poss_ext, args_ext, lengths_ext = sent_ids, sents, poss, argu_cands, lengths
                        tri_idxs=[]
                        n_events=[]
                    if self.args.cuda:
                        args_ext = args_ext.cuda()

                    if self.args.finetune_bert:
                        bert_max_len = max(bert_lengths_ext)
                        bert_attn_mask = torch.arange(bert_max_len).expand(len(bert_lengths_ext), bert_max_len) < torch.LongTensor(bert_lengths_ext).unsqueeze(1)
                        bert_attn_mask = bert_attn_mask.type(torch.FloatTensor)
                        if self.args.cuda:
                            bert_attn_mask = bert_attn_mask.cuda()

                    if self.args.use_crf_a:
                        crf_mask_ext = torch.arange(max_len).expand(len(lengths_ext), max_len) < torch.LongTensor(lengths_ext).unsqueeze(1)
                        if self.args.cuda:
                            crf_mask_ext = crf_mask_ext.cuda()
                        out_e, _, crf_loss_e, _ = self.model(sents_ext, poss_ext, lengths_ext, task='argument',
                                                       tri_idxs=tri_idxs, att_pool=self.args.att_pool,
                                                       att_mthd=self.args.att_mthd, crf=True,
                                                       seq_tags=args_ext, crf_mask=crf_mask_ext,
                                                       use_att=self.args.use_att,
                                                       bias_tensor_t=self.bias_tensor_t, bias_tensor_a=self.bias_tensor_a,
                                                       glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map_ext, bert_attn_mask=bert_attn_mask)
                        loss_e = -1.0 * crf_loss_e
                    else:
                        out_e, prob_e = self.model(sents_ext, poss_ext, lengths_ext,
                                                   task='argument', tri_idxs=tri_idxs,
                                                   att_pool=self.args.att_pool,
                                                   att_mthd=self.args.att_mthd,
                                                   use_att=self.args.use_att,
                                                   bias_tensor_t=self.bias_tensor_t, bias_tensor_a=self.bias_tensor_a,
                                                   glove_idx=glove_idx, orig_to_tok_map=orig_to_tok_map_ext, bert_attn_mask=bert_attn_mask)
                        loss_e = get_loss_mlp(lengths_ext, args_ext, out_e, criterion_e)
                    loss += self.args.argument_weight * loss_e
                    self.logger.scalar_summary(tag='train/norm_argu_loss',
                                               value=loss_e.item(),
                                               step=step)
                loss = loss / self.args.iter_size  # normalize gradients
                loss.backward()
                if (n_iter) % self.args.iter_size == 0:
                    # accumulate gradients for iter_size batches
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    self.model.zero_grad()
                # if n_iter % 1000 == 0:
                #     print('Trained {} batchs'.format(n_iter))
                #     print('batch loss {}, epoch_loss_t {}, epoch_loss_e {}'.format(loss, epoch_loss_t, epoch_loss_e))
                loss_hist_e.append(loss_e.data.cpu().numpy())
                loss_hist_t.append(loss_t.data.cpu().numpy())
                loss_hist_ner.append(loss_ner.data.cpu().numpy())
                epoch_loss_e += loss_e
                epoch_loss_t += loss_t
                epoch_loss_ner += loss_ner
                step += 1
            print("Epoch loss ner: {}, trigger: {}, argument: {}".format(epoch_loss_ner, epoch_loss_t, epoch_loss_e))
            # Evaluate at the end of each epoch
            print("*"*10+'Evaluating on Dev Set.....'+'*'*10)
            if len(eval_data) > 0:
                y_trues_t, y_preds_t, y_trues_e, y_preds_e, y_trues_e_sent, y_preds_e_sent, _, y_trues_ner, y_preds_ner = \
                        self.predict(eval_data, train_task, test=False, use_gold=True, e2e_eval=False)
                # if epoch == 3:
                #     pdb.set_trace()
                if len(y_trues_ner) > 0:
                    f1_ner = f1_score(y_trues_ner, y_preds_ner)
                    acc_ner = accuracy_score(y_trues_ner, y_preds_ner)
                    print('ner f1: {:.4f}, ner acc: {:.4f}'.format(f1_ner, acc_ner))
                    eval_f1 = f1_ner
                    self.logger.scalar_summary(tag='dev/ner_f1',
                                               value=f1_ner,
                                               step=epoch)
                if len(y_trues_t) > 0:
                    f1_t = f1_score(y_trues_t, y_preds_t)
                    acc_t = accuracy_score(y_trues_t, y_preds_t)
                    print('trigger f1: {:.4f}, trigger acc: {:.4f}'.format(f1_t, acc_t))
                    eval_f1 = f1_t
                    self.logger.scalar_summary(tag='dev/trigger_f1',
                                               value=f1_t,
                                               step=epoch)
                    # print(classification_report(y_trues_t, y_preds_t))
                if len(y_trues_e) > 0:
                    f1_e = f1_score(y_trues_e, y_preds_e)
                    acc_e = accuracy_score(y_trues_e, y_preds_e)
                    print('arguments f1: {:.4f}, arguments acc: {:.4f}'.format(f1_e, acc_e))
                    eval_f1 = f1_e
                    # if self.args.use_att:
                    if self.args.eval_sent_arg_role:
                        f1_arg_sent = f1_score(y_trues_e_sent, y_preds_e_sent)
                        acc_arg_sent = accuracy_score(y_trues_e_sent, y_preds_e_sent)
                        print('arguments merged sent level f1: {:.4f}, arguments merged sent acc: {:.4f}'.format(f1_arg_sent, acc_arg_sent))
                        eval_f1 = f1_arg_sent # eval f1 prioritize arg_sent, arg, then trigger
                        self.logger.scalar_summary(tag='dev/argu_sent_f1',
                                                   value=f1_arg_sent,
                                                   step=epoch)
                    self.logger.scalar_summary(tag='dev/argument_f1',
                                               value=f1_e,
                                               step=epoch)
                if train_task in ['multitask_pipe', 'multitask_joint']:
                    if epoch > self.args.tri_start_epochs:
                        print("====end2end eval====")
                        y_trues_t, y_preds_t, y_trues_e, y_preds_e, y_trues_e_sent, y_preds_e_sent, sent_ids, y_trues_ner, y_preds_ner = \
                                self.predict(eval_data, train_task, test=False, use_gold=False, e2e_eval=True)
                        # out_pkl_dir = './tmp/{}_e2edata{}_pipepochs{}_epochs{}'.format(train_task, self.args.train_on_e2e_data, self.args.pipe_epochs, self.args.epochs)
                        out_pkl_dir = './tmp/{}_e2edata{}_pipepochs{}_epochs{}_tb{}_ab{}_tw{}_aw{}'.\
                                       format(train_task, self.args.train_on_e2e_data, self.args.pipe_epochs, self.args.epochs, \
                                              self.args.bias_t, self.args.bias_a, self.args.trigger_weight, self.args.argument_weight)
                        if self.args.split_arg_role_label is True:
                            pkl_name = 'dev_end2end_cls.pkl'
                            B2I_trigger = self.args.B2I_trigger_string
                            B2I_arg = self.args.B2I_arg_string
                        elif self.args.split_arg_role_label is False:
                            pkl_name = 'dev_end2end_id.pkl'
                            B2I_trigger = self.args.B2I_trigger_string
                            B2I_arg = self.args.B2I_arg_string
                        write_pkl(sent_ids, y_trues_t, y_preds_t, y_trues_e, y_preds_e, y_trues_e_sent, y_preds_e_sent, out_pkl_dir, pkl_name)
                        if os.path.isfile('tmp/gold_dev_internal_tritypeTrue_ft_uw_cls.pkl') and \
                           os.path.isfile('{}/{}'.format(out_pkl_dir, pkl_name)):
                            precs, recalls, f1_e2es = eval_ace('tmp/gold_dev_internal_tritypeTrue_ft_uw_cls.pkl',
                                                            '{}/{}'.format(out_pkl_dir, pkl_name),
                                                            B2I_trigger,
                                                            B2I_arg,
                                                            eval_argu=(not train_task=='singletask_trigger'))
                            print('end2end eval f1 {}'.format(f1_e2es[-1]))
                    if epoch <= self.args.tri_start_epochs:
                        eval_f1 = f1_e
                    elif epoch <= self.args.pipe_epochs + 1:
                        if train_task == 'singletask_trigger':
                            eval_f1 = f1_e2es[1]
                        else:
                            eval_f1 = f1_e2es[-1]  # when entered joint phase, prioritize e2e f1
                    else:
                        # average of trigger score and arg score, b/c when predicting trigger type the trigger score is pretty low
                        # we might want it to select model as well
                        # eval_f1 = f1_e
                        # eval_f1 = f1_t
                        # eval_f1 = (f1_t + f1_e) / 2
                        if train_task == 'singletask_trigger':
                            eval_f1 = f1_e2es[1]
                        else:
                            eval_f1 = f1_e2es[-1]  # always prioritize e2e f1

                if epoch == self.args.pipe_epochs + 1:
                    # when entered joint phase(using predicetd triggers)
                    # update the best_f1 to be the new one eval on predicted triggers
                    print('====Entered joint phase.====')
                    # print('Update best_eval_f1 from {} to {}'.format(best_eval_f1, eval_f1))
                    # best_eval_f1 = eval_f1

                if eval_f1  > best_eval_f1:
                    print('Update eval f1: {}'.format(eval_f1))
                    best_eval_f1 = eval_f1
                    self.best_model_state_dict = {k:v.to('cpu') for k, v in self.model.state_dict().items()}
                    # self.best_model = copy.deepcopy(model_cpu)
                    # self.best_model = self.best_model.cpu()  # move the best model to cpu to save gpu memory
                    self.save(self.args.save_dir+'/best_model.pt', epoch)
                    best_epoch = epoch
                    patience = 0
                else:
                    patience += 1

                if patience > self.args.patience:
                    print('Exceed patience limit. Break at epoch{}'.format(epoch))
                    break

        print("Final Evaluation Best F1: {:.4f} at Epoch {}".format(best_eval_f1, best_epoch))
        print('=====testing on test=====')
        y_trues_t, y_preds_t, y_trues_e, y_preds_e, y_trues_e_sent, y_preds_e_sent, sent_ids_ext, y_trues_ner, y_preds_ner = \
                self.predict(test_data, train_task, test=True, use_gold=True, e2e_eval=False)
        # pdb.set_trace()
        test_f1_ner = 0
        test_f1_e = 0
        test_f1_t = 0
        if len(y_trues_ner) > 0:
            f1_ner = f1_score(y_trues_ner, y_preds_ner)
            acc_ner = accuracy_score(y_trues_ner, y_preds_ner)
            print('ner f1: {:.4f}, ner acc: {:.4f}'.format(f1_ner, acc_ner))
            test_f1_ner = f1_ner
            self.logger.scalar_summary(tag='dev/ner_f1',
                                       value=f1_ner,
                                       step=0)
        if len(y_trues_t) > 0:
            f1_t = f1_score(y_trues_t, y_preds_t)
            acc_t = accuracy_score(y_trues_t, y_preds_t)
            print('trigger f1: {:.4f}, trigger acc: {:.4f}'.format(f1_t, acc_t))
            test_f1_t = f1_t
            self.logger.scalar_summary(tag='test/trigger_f1',
                                       value=f1_t,
                                       step=0)
        if len(y_trues_e) > 0:
            f1_e = f1_score(y_trues_e, y_preds_e)
            acc_e = accuracy_score(y_trues_e, y_preds_e)
            print('arguments f1: {:.4f}, arguments acc: {:.4f}'.format(f1_e, acc_e))
            test_f1_e = f1_e
            self.logger.scalar_summary(tag='test/argument_f1',
                                       value=f1_e,
                                       step=0)
            # if self.args.use_att:
            if self.args.eval_sent_arg_role:
                f1_arg_sent = f1_score(y_trues_e_sent, y_preds_e_sent)
                acc_arg_sent = accuracy_score(y_trues_e_sent, y_preds_e_sent)
                print('arguments merged sent level f1: {:.4f}, arguments merged sent acc: {:.4f}'.format(f1_arg_sent, acc_arg_sent))
                test_f1_e = f1_arg_sent
                self.logger.scalar_summary(tag='test/argu_sent_f1',
                                           value=f1_arg_sent,
                                           step=0)

        if train_task in ['multitask_pipe', 'multitask_joint', 'singletask_trigger']:
            print("====end2end eval====")
            y_trues_t, y_preds_t, y_trues_e, y_preds_e, y_trues_e_sent, y_preds_e_sent, sent_ids, y_trues_ner, y_preds_ner =\
                    self.predict(test_data, train_task, test=True, use_gold=False, e2e_eval=True)
            # out_pkl_dir = './tmp/{}_e2edata{}_pipepochs{}_epochs{}'.format(train_task, self.args.train_on_e2e_data, self.args.pipe_epochs, self.args.epochs)
            out_pkl_dir = './tmp/{}_e2edata{}_pipepochs{}_epochs{}_tb{}_ab{}_tw{}_aw{}'.\
                           format(train_task, self.args.train_on_e2e_data, self.args.pipe_epochs, self.args.epochs, \
                                  self.args.bias_t, self.args.bias_a, self.args.trigger_weight, self.args.argument_weight)
            if self.args.split_arg_role_label is True:
                pkl_name = 'test_end2end_cls.pkl'
                B2I_trigger = self.args.B2I_trigger_string
                B2I_arg = self.args.B2I_arg_string
            elif self.args.split_arg_role_label is False:
                pkl_name = 'test_end2end_id.pkl'
                B2I_trigger = self.args.B2I_trigger_string
                B2I_arg = self.args.B2I_arg_string
            write_pkl(sent_ids, y_trues_t, y_preds_t, y_trues_e, y_preds_e, y_trues_e_sent, y_preds_e_sent, out_pkl_dir, pkl_name)
            if os.path.isfile('tmp/gold_test_internal_tritypeTrue_ft_uw_cls.pkl') and \
               os.path.isfile('{}/{}'.format(out_pkl_dir, pkl_name)):
                precs, recalls, f1_e2es = eval_ace('tmp/gold_test_internal_tritypeTrue_ft_uw_cls.pkl',
                                                '{}/{}'.format(out_pkl_dir, pkl_name),
                                                B2I_trigger,
                                                B2I_arg,
                                                eval_argu=(not train_task=='singletask_trigger'))
                print('end2end eval f1 {}'.format(f1_e2es[-1]))

        return best_eval_f1, best_epoch, test_f1_e, test_f1_t

    def train_epoch(self, train_data, dev_data, test_data, task):
        best_f1, best_epoch, test_f1_e, test_f1_t = self._train(train_data, dev_data, test_data, task)
        print("Final Dev F1: %.4f" % best_f1)
        return best_f1, best_epoch, test_f1_e, test_f1_t

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
