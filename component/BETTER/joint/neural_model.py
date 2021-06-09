import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
import numpy as np
from torchnlp.nn import Attention
from torchcrf import CRF
from CRF_util import kViterbi
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
from torch.nn.utils.rnn import pad_sequence
import math
from util import get_loss_mlp, get_output_rel, get_loss_rel
from transformers import BertTokenizer, BertModel, BertConfig
import pdb

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        # x += -1e9   # add a very small value to avoid computing std() for all zeros tensor - this will lead to NaN gradients
        mean = x.mean(dim=-1, keepdim=True)
        std = (x - x.mean(dim=-1, keepdim=True)).norm(p=2, dim=-1, keepdim=True) / (x.size(-1)**0.5) + self.eps
        norm = self.alpha * (x - mean) / std + self.bias
        return norm

class Gate(nn.Module):
    def __init__(self, d_in, d_control, act='tanh'):
        '''
        d_intput is the hid_states, d_control is the signal to control the gate (trigger)
        '''
        super().__init__()

        self.linear = nn.Linear(d_in + d_control, d_in)

        if act:
            if act =='tanh':
                self.act = nn.Tanh()
            elif act =='relu':
                self.act = nn.ReLU()
            elif act =='leakyrelu':
                self.act = nn.LeakyReLU(negative_slope=0.1)
            elif act =='prelu':
                self.act = nn.PReLU()
        else:
            self.act = None

    def forward(self, input, control, mask=None):
        '''
        input size should be (bs, seq_len, d_in)
        control size should be (bs, 1, d_control)
        return size (bs, seq_len, d_in)
        '''
        seq_len = input.size(1)
        x = torch.cat([input, control.expand(-1, seq_len, -1)], dim=-1)
        if mask:
            x = x.masked_fill(mask==0, value=0.0)
        gate = self.act(self.linear(x))
        output = gate * (control.expand(-1, seq_len, -1)) + (1 - gate) * input
        if mask:
            output = output.masked_fill(mask==0, value=0.0)

        return output

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)  # (bs, 1, 1, sl)
        mask = mask.cuda(q.get_device()) if q.is_cuda else mask
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    # scores: (bs * h * 1 * sl)
    # v: (bs * h * sl * d_k)
    output = torch.matmul(scores, v)
    return scores, output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads = 1, dropout = 0.1, att_func = 'general', use_att_linear_out=False):
        '''
        att_func: function to calculate the attention weights
                  `general` means a matrix multiplication for Q, K, V before fead in the attention function for dot product
                  `dot` means direct dot product of Q and K, no matrix multiplication
                  `bidaf` TODO??
        '''
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.att_func = att_func
        self.use_att_linear_out = use_att_linear_out

        if self.att_func in ['general']:
            self.q_linear = nn.Linear(d_model, d_model)
            self.v_linear = nn.Linear(d_model, d_model)
            self.k_linear = nn.Linear(d_model, d_model)
        elif self.att_func in ['bidaf']:
            self.linear_s = nn.Linear(d_model*3, 1)
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout=None

        if self.use_att_linear_out:
            self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        '''
        input Q, K, V shape Batch_size * seq_len * d_model
        '''

        bs = q.size(0)

        if self.att_func in ['general']:
            # perform linear operation and split into h heads
            if self.dropout:
                k = self.dropout(self.k_linear(k)).view(bs, -1, self.h, self.d_k)
                q = self.dropout(self.q_linear(q)).view(bs, -1, self.h, self.d_k)
                v = self.dropout(self.v_linear(v)).view(bs, -1, self.h, self.d_k)
            else:
                k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
                q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
                v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        elif self.att_func in ['dot']:
            # split into h heads w/o linear
            k = k.view(bs, -1, self.h, self.d_k)
            q = q.view(bs, -1, self.h, self.d_k)
            v = v.view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_k
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next
        # q, k, v shape: bs * h * sl * d_k
        # returned shape: scores: (bs, h, 1, sl) output: (bs, h, 1, d_k)
        scores, output = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        output = output.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        # get the single-head attention
        seq_len = scores.size(-1)
        scores = scores.transpose(1,2).contiguous().view(bs, -1, seq_len)

        if self.use_att_linear_out:
            output = self.out(output)

        return output, scores

def create_emb_layer(weights_matrix, trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.weight = Parameter(torch.FloatTensor(weights_matrix))
    if not trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer

class BertClassifier(nn.Module):
    'neural network architecture'
    def __init__(self, args):
        super(BertClassifier, self).__init__()
        self.args = args
        if self.args.finetune_bert:
            if self.args.bert_model_type in ['bert-base-uncased', 'bert-base-cased']:
                self.hid_size = int(768 / 2)    # 2 account for bi-directional hid representation in orig linear layers
            elif self.args.bert_model_type in ['bert-large-uncased', 'bert-large-cased']:
                self.hid_size = int(1024 / 2)    # 2 account for bi-directional hid representation in orig linear layers
            self.hid_size_lastmlp = self.args.hid_lastmlp
        else:
            self.hid_size = args.hid
            self.hid_last_mlp = self.hid_size
        self.batch_size = args.batch
        self.num_layers = args.num_layers
        if args.gold_ent:
            self.num_arg_classes = max(args._label_to_id_te.values()) + 1
        else:
            self.num_arg_classes = max(args._label_to_id_e.values()) + 1
        self.num_tri_classes = max(args._label_to_id_t.values()) + 1
        self.num_ner_classes = max(args._label_to_id_e_sent.values()) + 1  # the `_label_to_id_e_sent` name is bad. This is just the NER label space

        self.use_pos = False
        if not self.args.finetune_bert:
            if self.args.use_bert:
                if self.args.use_glove:
                    self.word_emb = create_emb_layer(args.word_emb, trainable=args.trainable_emb)
                    self.word_emb_shape = args.bert_dim + args.word_emb.shape[1]
                else:
                    self.word_emb_shape = args.bert_dim
            else:
                self.word_emb_shape = args.word_emb.shape[1]
                self.word_emb = create_emb_layer(args.word_emb, trainable=args.trainable_emb)
            if args.use_pos:
                self.use_pos = True
                self.pos_emb = create_emb_layer(args.pos_emb, trainable=args.trainable_pos_emb)
                self.lstm = nn.LSTM(self.word_emb_shape+args.pos_emb.shape[1], self.hid_size,
                                    self.num_layers, bidirectional=True, batch_first=True, dropout=args.dropout)
            else:
                self.lstm = nn.LSTM(self.word_emb_shape, self.hid_size, self.num_layers,
                                    bidirectional=True, batch_first=True, dropout=args.dropout)
        else:
            MODELS = [(BertConfig, BertModel, BertTokenizer, self.args.bert_model_type)]
            for config_class, model_class, tokenizer_class, pretrained_weights in MODELS:
                config = config_class.from_pretrained(pretrained_weights, output_hidden_states=True)
                self.bert_encoder = model_class.from_pretrained(pretrained_weights, config=config)
                device = torch.device("cuda" if args.cuda else "cpu")
                self.bert_encoder.to(device)

        self.dropout = nn.Dropout(p=args.dropout)

        # mlp classifier for trigger
        self.linear1_tri = nn.Linear(self.hid_size*2, self.hid_size_lastmlp)
        self.linear2_tri = nn.Linear(self.hid_size_lastmlp, self.num_tri_classes)
        if self.args.ner_weight > 0:
            # mlp classifier for ner
            self.linear1_ner = nn.Linear(self.hid_size*2, self.hid_size_lastmlp)
            self.linear2_ner = nn.Linear(self.hid_size_lastmlp, self.num_ner_classes)

        if args.use_att:
            if args.att_mthd in ['cat']:
                # *4 account for concatenate trigger word with current argument word
                # self.linear1_arg = nn.Linear(self.hid_size*4, self.hid_size*2)
                # self.linear2_arg = nn.Linear(self.hid_size*2, self.num_arg_classes)
                self.linear1_arg = nn.Linear(self.hid_size*4, self.hid_size_lastmlp*2)
                self.linear2_arg = nn.Linear(self.hid_size_lastmlp*2, self.num_arg_classes)
                if self.args.norm:
                    self.norm = Norm(self.hid_size*4)
            elif args.att_mthd in ['gate']:
                self.attention = MultiHeadAttention(d_model=self.hid_size*2, heads=1, dropout=self.args.att_dropout,
                                                    att_func=self.args.att_func, use_att_linear_out=self.args.use_att_linear_out)
                self.gate = Gate(self.hid_size*2, self.hid_size*2)
                self.linear1_arg = nn.Linear(self.hid_size*2, self.hid_size)
                self.linear2_arg = nn.Linear(self.hid_size, self.num_arg_classes)
                if self.args.norm:
                    self.norm = Norm(self.hid_size*2)
            elif args.att_mthd in ['att_cat', 'att_mul_cat', 'att_sum',  'att_mul_sum', 'att_mul_replace', \
                                   'att_sub',  'att_mul_sub', \
                                   'cat_self_att_sub_elem_prod', 'cat_self_att_sub', 'cat_self_att_elem_prod']:
                # self.attention = Attention(self.hid_size*2)
                self.attention = MultiHeadAttention(d_model=self.hid_size*2, heads=1, dropout=self.args.att_dropout,
                                                    att_func=self.args.att_func, use_att_linear_out=self.args.use_att_linear_out)

                if args.att_mthd in ['att_cat', 'att_mul_cat']:
                    self.linear1_arg = nn.Linear(self.hid_size*4, self.hid_size)
                    self.linear2_arg = nn.Linear(self.hid_size, self.num_arg_classes)
                    if self.args.norm:
                        self.norm = Norm(self.hid_size*4)

                elif args.att_mthd in['att_sum',  'att_mul_sum', 'att_mul_replace',\
                                      'att_sub',  'att_mul_sub']:
                    self.linear1_arg = nn.Linear(self.hid_size*2, self.hid_size)
                    self.linear2_arg = nn.Linear(self.hid_size, self.num_arg_classes)
                    if self.args.norm:
                        self.norm = Norm(self.hid_size*2)
                elif args.att_mthd in ['cat_self_att_sub_elem_prod']:
                    self.linear1_arg = nn.Linear(self.hid_size*8, self.hid_size)
                    self.linear2_arg = nn.Linear(self.hid_size, self.num_arg_classes)
                    if self.args.norm:
                        self.norm = Norm(self.hid_size*8)
                elif args.att_mthd in ['cat_self_att_sub', 'cat_self_att_elem_prod']:
                    self.linear1_arg = nn.Linear(self.hid_size*6, self.hid_size)
                    self.linear2_arg = nn.Linear(self.hid_size, self.num_arg_classes)
                    if self.args.norm:
                        self.norm = Norm(self.hid_size*6)
            else:
                # do not use attention
                self.linear1_arg = nn.Linear(self.hid_size*2, self.hid_size_lastmlp)
                self.linear2_arg = nn.Linear(self.hid_size_lastmlp, self.num_arg_classes)
        else:
            self.linear1_arg = nn.Linear(self.hid_size*2, self.hid_size_lastmlp)
            self.linear2_arg = nn.Linear(self.hid_size_lastmlp, self.num_arg_classes)

        if self.args.activation:
            if self.args.activation=='tanh':
                self.act = nn.Tanh()
            elif self.args.activation=='relu':
                self.act = nn.ReLU()
            elif self.args.activation=='leakyrelu':
                self.act = nn.LeakyReLU(negative_slope=0.1)
            elif self.args.activation=='prelu':
                self.act = nn.PReLU()

        self.softmax_ent = nn.Softmax(dim=2)

        # crf layer
        if args.use_crf_ner:
            self.crf_ner = CRF(self.num_ner_classes, batch_first=True)
        if args.use_crf_t:
            self.crf_t = CRF(self.num_tri_classes, batch_first=True)
        if args.use_crf_a:
            self.crf_e = CRF(self.num_arg_classes, batch_first=True)

    def get_query(self, in_tensor, tri_idxs, att_pool='max'):
        '''
        in_tensor is the lstm output hid states, size(batch_size, seq_len, hid_dim)
        returned size (batch_size, 1, hid_dim)
        '''
        out_tensor = []
        hid_dim = in_tensor.size(-1)

        for i, idx in enumerate(tri_idxs):
            if idx == []:
                # weird case where no anchors appear
                # zero out the query tensor
                sel = torch.zeros(1, hid_dim, device=in_tensor.device)
            else:
                # select all trigger word representations
                sel = in_tensor[i, idx, :].view(-1, hid_dim)
            if att_pool == 'max':
                sel_max, _ = torch.max(sel, dim=0, keepdim=True)
            out_tensor.append(sel_max.unsqueeze(0))
        out_tensor = torch.cat(out_tensor, dim=0)
        assert out_tensor.size(1) == 1
        return out_tensor#.detach()

    def get_repre_from_align(self, hid_state, orig_to_tok_map, bert_attn_mask, pooling='average'):
        '''
        hid_state is (seq_len, hid_dim) tensor (NOTE not in batch)
        orig_to_tok_map is a list, recording the bert alignment
        bert_attn_mask not in batch
        '''
        bert_length = len([x for x in bert_attn_mask if x==1])
        out = []
        for orig_idx, bert_idx in enumerate(orig_to_tok_map):
            if orig_idx != len(orig_to_tok_map) - 1:
                sel_idx = list(range(orig_to_tok_map[orig_idx], orig_to_tok_map[orig_idx+1]))
            else:
                # last token
                sel_idx = list(range(orig_to_tok_map[orig_idx], bert_length-1))  # do not use the [SEP] representation
            sel = hid_state[sel_idx, :]
            if pooling == 'average':
                sel_repre = torch.mean(sel, dim=0, keepdim=True)
            elif pooling == 'max':
                sel_repre = torch.max(sel, dim=0, keepdim=True)[0]
            out.append(sel_repre)
        return torch.cat(out, dim=0)

    def forward(self, sents, pos_tags, lengths, task='trigger',
                tri_idxs=[], att_pool=None, att_mthd=None,
                crf=False, seq_tags=None, crf_mask=None, use_att=False,
                k_ner=1, k_tri=1, k_arg=1,
                bias_tensor_t=None, bias_tensor_a=None,
                glove_idx=None, orig_to_tok_map=None, bert_attn_mask=None,
                argu_cands_mask=None, argu_roles_mask_by_tri=None,
                trigger_mask=None, argu_roles_mask_by_ent=None):
        if not self.args.finetune_bert:
            #word
            if self.args.use_bert:
                word_emb = sents
                if self.args.use_glove:
                    glove_emb = self.word_emb(glove_idx)
                    word_emb = torch.cat([word_emb, glove_emb], dim=-1)
            else:
                word_emb = self.word_emb(sents)

            #pos
            if self.use_pos:
                pos_emb = self.pos_emb(pos_tags)
                word_emb = torch.cat((word_emb, pos_emb), dim=2)

            out = self.dropout(word_emb)
            self.lstm.flatten_parameters()
            out, _ = self.lstm(pack(out,lengths,batch_first=True))
            out, _ = unpack(out, batch_first=True)
        elif self.args.finetune_bert:
            assert orig_to_tok_map, pdb.set_trace()  # when finetuning bert, this map(head_index) is required for torch.index_select()
            ## retreive the representations according to the BERT alignment
            bert_output = self.bert_encoder(sents, attention_mask=bert_attn_mask)
            bert_hid_stat = bert_output[0]
            # bert_hid_stat = torch.cat([bert_output[0], bert_output[2][-3]], dim=-1)
            out = []
            max_len = bert_hid_stat.size(1)
            for i, (a, sel_idxs) in enumerate(zip(bert_hid_stat, orig_to_tok_map)):
                assert lengths[i] == len(sel_idxs)
                if self.args.bert_encode_mthd in ['average', 'max']:
                    sel_repres = self.get_repre_from_align(a, sel_idxs, bert_attn_mask[i], self.args.bert_encode_mthd)
                    out.append(sel_repres)
                elif self.args.bert_encode_mthd =='head':
                    if self.args.cuda:
                        out.append(torch.index_select(a, 0, torch.LongTensor(sel_idxs).cuda()))
                    else:
                        out.append(torch.index_select(a, 0, torch.LongTensor(sel_idxs)))
            out = pad_sequence(out, batch_first=True)


        if use_att:
            max_len = lengths[0]
            # att_mask will zero out att_weights for padded tokens
            att_mask = torch.arange(max_len).expand(len(lengths), max_len) < torch.LongTensor(lengths).unsqueeze(1)
            att_mask = att_mask.unsqueeze(1)
            att_mask = att_mask.cuda(out.get_device()) if out.is_cuda else att_mask
            if att_pool:
                query = self.get_query(out, tri_idxs, att_pool) # batch x 1 x hid_dim

            if att_mthd:
                if att_mthd == 'cat':
                    seq_len = out.size(1)
                    out = torch.cat([out, query.expand(-1, seq_len, -1)], dim=2)
                elif att_mthd == 'gate':
                    att, att_w = self.attention(out, out, out, mask=att_mask)
                    out = self.gate(att, query)
                elif att_mthd in ['att_cat', 'att_mul_cat', 'att_sum', 'att_sub', 'att_mul_replace', \
                                  'att_mul_sum', 'att_mul_sub']:
                    # first compute attn according using trigger as query
                    # att, att_w = self.attention(query, out)
                    att, att_w = self.attention(query, out, out, mask=att_mask)
                    seq_len = out.size(1)
                    # next construct the final hid states according to different cases
                    if att_mthd == 'att_cat':
                        out = torch.cat([out, att.expand(-1, seq_len, -1)], dim=2)
                    elif att_mthd == 'att_mul_cat':
                        # broadcast
                        out_att = torch.transpose(att_w, 1, 2) * out
                        out = torch.cat([out, out_att], dim=2)
                    elif att_mthd == 'att_sum':
                        out = att + out
                    elif att_mthd == 'att_sub':
                        out = out - att
                    elif att_mthd == 'att_mul_replace':
                        # broadcast
                        out = torch.transpose(att_w, 1, 2) * out
                    elif att_mthd == 'att_mul_sum':
                        # broadcast
                        out_att = torch.transpose(att_w, 1, 2) * out
                        out = out_att + out
                    elif att_mthd == 'att_mul_sub':
                        # broadcast
                        out_att = torch.transpose(att_w, 1, 2) * out
                        out = out - out_att
                elif att_mthd in ['cat_self_att_sub_elem_prod', 'cat_self_att_sub', 'cat_self_att_elem_prod']:
                    # to perform FULL self-attention, feed in the `out` as Q, instead of query as Q
                    att, att_w = self.attention(out, out, out, mask=att_mask)
                    # still compute a attn for trigger
                    att_tri, att_w_tri = self.attention(query, out, out, mask=att_mask)
                    seq_len = out.size(1)
                    if att_mthd == 'cat_self_att_sub_elem_prod':
                        # concat X, X-C, X.C, same as CoVe
                        out = torch.cat([out, out - att, out * att, att_tri.expand(-1, seq_len, -1)], dim=2)
                    elif att_mthd == 'cat_self_att_sub':
                        # same as CoVe, except removing the elemwise-product term
                        out = torch.cat([out, out - att, att_tri.expand(-1, seq_len, -1)], dim=2)
                    elif att_mthd == 'cat_self_att_elem_prod':
                        # same as CoVe, except removing the subtraction term
                        out = torch.cat([out, out * att, att_tri.expand(-1, seq_len, -1)], dim=2)

                out = out.masked_fill(att_mask.transpose(1,2)==0, value=0.0)  # zero out the values at padded positions

        ### NER
        if task == 'ner':
            out_ner = self.linear1_ner(self.dropout(out))
            if self.args.activation:
                out_ner = self.act(out_ner)
            out_ner = self.linear2_ner(out_ner)

            if crf:
                if seq_tags is not None:
                    # during training the gold tags are provided to calculate loss
                    crf_loss = self.crf_ner(out_ner, seq_tags, mask=crf_mask, reduction='mean')
                else:
                    crf_loss = None
                crf_out_k, crf_out_k_prob = kViterbi(self.crf_ner, out_ner, k_ner, crf_mask)
                return crf_out_k, crf_out_k_prob, crf_loss, out_ner
            else:
                prob_ner = self.softmax_ent(out_ner)
                return out_ner, prob_ner

        ### trigger prediction - predict each input token
        if task == 'trigger':
            out_tri = self.linear1_tri(self.dropout(out))
            if self.args.activation:
                out_tri = self.act(out_tri)
            out_tri = self.linear2_tri(out_tri)
            # will use the bias term for non-O triggers to compute loss
            # out_tri_bias = out_tri * bias_tensor_t
            if bias_tensor_t is not None:
                out_tri = out_tri + torch.log(bias_tensor_t)

            if self.args.decode_w_trigger_mask:
                if trigger_mask is not None:
                    out_tri = out_tri.masked_fill(trigger_mask, value=-1e7)

            if crf:
                if seq_tags is not None:
                    # during training the gold tags are provided to calculate loss
                    crf_loss = self.crf_t(out_tri, seq_tags, mask=crf_mask, reduction='mean')
                else:
                    crf_loss = None
                crf_out_k, crf_out_k_prob = kViterbi(self.crf_t, out_tri, k_tri, crf_mask)
                return crf_out_k, crf_out_k_prob, crf_loss, out_tri
            else:
                prob_tri = self.softmax_ent(out_tri)
                return out_tri, prob_tri

        ### argument prediction - predict each input token
        if task == 'argument':
            if self.args.norm:
                out = self.norm(out)
            out_arg = self.linear1_arg(self.dropout(out))
            if self.args.activation:
                out_arg = self.act(out_arg)
            out_arg = self.linear2_arg(out_arg) # batch x seq_len x num_class
            # will use the bias term for non-O triggers to compute loss
            # out_arg_bias = out_arg * bias_tensor_a
            if bias_tensor_a is not None:
                out_arg = out_arg + torch.log(bias_tensor_a)


            if self.args.decode_w_ents_mask:
                if argu_cands_mask is not None:
                    out_arg = out_arg.masked_fill(argu_cands_mask, value=-1e7)

            if self.args.decode_w_arg_role_mask_by_tri:
                if argu_roles_mask_by_tri is not None:
                    out_arg = out_arg.masked_fill(argu_roles_mask_by_tri, value=-1e7)
            if self.args.decode_w_arg_role_mask_by_ent:
                if argu_roles_mask_by_ent is not None:
                    out_arg = out_arg.masked_fill(argu_roles_mask_by_ent, value=-1e7)

            if crf:
                if seq_tags is not None:
                    # during training the gold tags are provided to calculate loss
                    crf_loss = self.crf_e(out_arg, seq_tags, mask=crf_mask, reduction='mean')
                else:
                    crf_loss = None
                crf_out_k, crf_out_k_prob = kViterbi(self.crf_e, out_arg, k_arg, crf_mask)
                return crf_out_k, crf_out_k_prob, crf_loss, out_arg # used for SSVM training with k_arg=1
            else:
                prob_arg = self.softmax_ent(out_arg)
                return out_arg, prob_arg

########## Biaffine part ###############
class BiaffineModule(nn.Module):
    def __init__(self, in1_feature_dim, in2_feature_dim, out_feature_dim, bias=(True, True)):
        super().__init__()
        self.in1_feature_dim = in1_feature_dim
        self.in2_feature_dim = in2_feature_dim
        self.out_feature_dim = out_feature_dim
        self.bias = bias
        self.linear_input_size = in1_feature_dim + int(bias[0])
        self.linear_output_size = out_feature_dim * (in2_feature_dim+int(bias[1]))
        self.linear = nn.Linear(self.linear_input_size, self.linear_output_size, bias=False)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        _, len2, dim2 = input2.size()
        if self.bias[0]:
            ones1 = input1.new_ones((batch_size, len1, 1), requires_grad=False)
            input1 = torch.cat((input1, ones1), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones1 = input2.new_ones((batch_size, len2, 1), requires_grad=False)
            input2 = torch.cat((input2, ones2), dim=2)
            dim2 += 1
        affine = self.linear(input1)
        affine = affine.view(batch_size, len1*self.out_feature_dim, dim2)
        input2 = torch.transpose(input2, 1, 2) # batch x dim2, len2
        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2) # batch x len2, len1*out_fea
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_feature_dim)

        #return F.log_softmax(torch.transpose(biaffine, 1, 2), dim=-1) # batch x len1 x len2 x out_fea
        return torch.transpose(biaffine, 1, 2) # batch x len1 x len2 x out_fea

    def __repr__(self):
        return self.__class__.__name__+'(in1_feat_dim='+str(self.in1_feature_dim) \
            +', in2_feat_dim='+str(self.in2_feature_dim) \
            +', out_feat_dim='+str(self.out_feature_dim)+')'

class NonLinear(nn.Module):
    def __init__(self, args, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        #self.linear2 = nn.Linear(hidden_size, hidden_size, bias=False)
        #self.linear1 = nn.Linear(input_size, hidden_size, bias=False)
        self.act = None
        if args.activation:
            if args.activation=='tanh':
                self.act = nn.Tanh()
            elif args.activation=='relu':
                self.act = nn.ReLU()
            elif args.activation=='leakyrelu':
                self.act = nn.LeakyReLU(negative_slope=0.1)
            elif args.activation=='prelu':
                self.act = nn.PReLU()
            else:
                raise ValueError
    def forward(self, x):
        y = self.linear(x)
        #y = self.linear1(x)
        if self.act:
            return self.act(y)
            #return self.linear2(self.act(y))
        else:
            return y
            #return self.linear2(y)

class BiaffineClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()


        self.args = args
        if self.args.finetune_bert:
            if self.args.bert_model_type == 'bert-base-uncased':
                self.hid_size = int(768 / 2)    # 2 account for bi-directional hid representation in orig linear layers
            elif self.args.bert_model_type == 'bert-large-uncased':
                self.hid_size = int(1024 / 2)    # 2 account for bi-directional hid representation in orig linear layers
        else:
            self.hid_size = args.hid
        self.hid_lastmlp = args.hid_lastmlp
        self.batch_size = args.batch
        self.num_layers = args.num_layers
        if args.gold_ent:
            self.num_arg_classes = max(args._label_to_id_te.values()) + 1
        else:
            self.num_arg_classes = max(args._label_to_id_e.values()) + 1
        self.num_tri_classes = max(args._label_to_id_t.values()) + 1

        self.use_pos = False
        if not self.args.finetune_bert:
            if self.args.use_bert:
                if self.args.use_glove:
                    self.word_emb = create_emb_layer(args.word_emb, trainable=args.trainable_emb)
                    self.word_emb_shape = args.bert_dim + args.word_emb.shape[1]
                else:
                    self.word_emb_shape = args.bert_dim
            else:
                self.word_emb_shape = args.word_emb.shape[1]
                self.word_emb = create_emb_layer(args.word_emb, trainable=args.trainable_emb)
            if args.use_pos:
                self.use_pos = True
                self.pos_emb = create_emb_layer(args.pos_emb, trainable=args.trainable_pos_emb)
                self.lstm = nn.LSTM(self.word_emb_shape+args.pos_emb.shape[1], self.hid_size,
                                    self.num_layers, bidirectional=True, batch_first=True, dropout=args.dropout)
            else:
                self.lstm = nn.LSTM(self.word_emb_shape, self.hid_size, self.num_layers,
                                    bidirectional=True, batch_first=True, dropout=args.dropout)
        else:
            # directly use the bert model as the encoder
            self.bert_encoder = self.args.bert_encoder
        # self.project = nn.Linear(self.hid_size*2, self.hid_lastmlp)
        self.dropout = nn.Dropout(p=args.dropout)

        # projected layer
        # self.project_trigger = NonLinear(args, self.hid_lastmlp, self.hid_lastmlp//2)
        # self.project_argument = NonLinear(args, self.hid_lastmlp, self.hid_lastmlp//2)
        self.project_trigger = NonLinear(args, self.hid_size*2, self.hid_lastmlp)
        self.project_argument = NonLinear(args, self.hid_size*2, self.hid_lastmlp)

        # biaffine layer
        # self.biaffine = BiaffineModule(self.hid_lastmlp//2, self.hid_lastmlp//2, self.num_arg_classes, bias=(True, False))
        self.biaffine = BiaffineModule(self.hid_lastmlp, self.hid_lastmlp, self.num_arg_classes, bias=(True, False))

        # mlp classifier for trigger
        # self.trig_classifier = nn.Linear(self.hid_lastmlp//2, self.num_tri_classes)
        self.trig_classifier = nn.Linear(self.hid_lastmlp, self.num_tri_classes)
        # crf layer
        if args.use_crf:
            self.crf_t = CRF(self.num_tri_classes, batch_first=True)
        if args.use_crf_a:
            self.crf_a = CRF(self.num_arg_classes, batch_first=True)
        else:
            self.crf_a = None

    def forward(self, sents, pos_tags, lengths,
                decode=True, gold_trig_seq=None, crf_mask=None, k_tri=1,
                glove_idx=None, orig_to_tok_map=None, bert_attn_mask=None):
        if not decode:
            assert gold_trig_seq is not None

        if not self.args.finetune_bert:
            #word
            if self.args.use_bert:
                word_emb = sents
                if self.args.use_glove:
                    glove_emb = self.word_emb(glove_idx)
                    word_emb = torch.cat([word_emb, glove_emb], dim=-1)
            else:
                word_emb = self.word_emb(sents)

            #pos
            if self.use_pos:
                pos_emb = self.pos_emb(pos_tags)
                word_emb = torch.cat((word_emb, pos_emb), dim=2)

            out = self.dropout(word_emb)
            self.lstm.flatten_parameters()
            out, _ = self.lstm(pack(out,lengths,batch_first=True))
            out, _ = unpack(out, batch_first=True)
        elif self.args.finetune_bert:
            assert orig_to_tok_map  # when finetuning bert, this map(head_index) is required for torch.index_select()
            bert_output = self.bert_encoder(sents, attention_mask=bert_attn_mask)
            bert_hid_stat = bert_output[0]
            out = []
            max_len = bert_hid_stat.size(1)
            for i, (a, sel_idxs) in enumerate(zip(bert_hid_stat, orig_to_tok_map)):
                assert lengths[i] == len(sel_idxs)
                if self.args.cuda:
                    out.append(torch.index_select(a, 0, torch.LongTensor(sel_idxs).cuda()))
                else:
                    out.append(torch.index_select(a, 0, torch.LongTensor(sel_idxs)))
            out = pad_sequence(out, batch_first=True)


        out = self.dropout(out)
        # out = self.project(out)
        trig_repre = self.project_trigger(out)
        argu_repre = self.project_argument(out)

        ### trigger prediction - predict each input token
        out_tri = self.trig_classifier(self.dropout(trig_repre))
        if self.args.use_crf:
            trigger_prediction, crf_out_k_prob = kViterbi(self.crf_t, out_tri, k_tri, crf_mask)
            if decode:
                tri_loss = None
            else:
                tri_loss = -1*self.crf_t(out_tri, gold_trig_seq, mask=crf_mask, reduction='mean')
        else:
            trigger_prediction = get_prediction_mlp(lengths, out_tri)
            if decode:
                tri_loss = None
            else:
                tri_loss = get_loss_mlp(lengths, gold_trig_seq, out_tri, nn.CrossEntropyLoss())

        ### argument prediction - predict each input token
        biaffine_tensor = self.biaffine(trig_repre, argu_repre)
        return trigger_prediction, tri_loss, biaffine_tensor, self.crf_a, F.softmax(out_tri, dim=-1)
