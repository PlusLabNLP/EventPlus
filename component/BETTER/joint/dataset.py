from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
import torch
import pickle
from generate_data.contextualized_features_bert import bert_token

TOKEN_PAD_ID = 0
POS_PAD_ID = 6
TRI_PAD_ID = 0
ARGU_PAD_ID = 0


class EventDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, pkl_file, args):
        self.args = args
        # load data
        with open(pkl_file, 'rb') as handle:
            self.data = pickle.load(handle)

        # preprocessing
        new_data = list()
        for i in range(len(self.data)):
            out = list()
            if args.use_bert:
                out.append(self.data[i]['contextual_feature'])
            elif args.finetune_bert:
                sent_bert_tokens, sent_bert_ids, orig_to_tok_map = bert_token(self.data[i]['tokens'], args.bert_tokenizer)
                out.append(sent_bert_ids)

            else:
                if args.lower:
                    out.append([args.word2idx[x.lower()] for x in self.data[i]['tokens']])
                else:
                    out.append([args.word2idx[x] for x in self.data[i]['tokens']])
            out.append([args.pos2idx[x] for x in self.data[i]['pos_tag']])
            if args.trigger_type:
                out.append([args._label_to_id_t[x] for x in self.data[i]['sent_tri_label_type']])
            else:
                out.append([args._label_to_id_t[x] for x in self.data[i]['trigger_label']])
            # if args.decode_w_ents_mask is False:
                # out.append([args._label_to_id_e_sent[x] for x in self.data[i]['argu_label']])
            # elif args.decode_w_ents_mask is True:
            out.append([args._label_to_id_e_sent[x] for x in self.data[i]['ent_label']])     # now this item is argument candidates, instead of arguments
            if args.trigger_type:
                out.append([([args._label_to_id_t[x] for x in i[0]], [args._label_to_id_e[y] for y in i[1]]) \
                        for i in self.data[i]['sent_tri_arg_pairs_type']])
            else:
                out.append([([args._label_to_id_t[x] for x in i[0]], [args._label_to_id_e[y] for y in i[1]]) \
                        for i in self.data[i]['tri_arg_pairs']])
            # case 0 : use permutation of gold trigger and gold argument
            # out.append([(x[0], x[1], x[2], x[3], args._label_to_id_r[x[4]])\
            #             for x in self.data[i]['all_tri_arg_pairs']])
            out.append([])  ##### TODO, now dont do the `all_tri_arg_pairs` items so this is an empty list
            # case 1 : use candidate augmented pairs
            #out.append([(x[0], x[1], x[2], x[3], args._label_to_id_r[x[4]])\
            #            for x in self.data[i]['all_pairs_by_cand']])

            out.append(self.data[i]['sent_id'])
            if args.use_glove:
                if args.lower:
                    out.append([args.word2idx[x.lower()] for x in self.data[i]['tokens']])
                else:
                    out.append([args.word2idx[x] for x in self.data[i]['tokens']])
            else:
                out.append([])
            if args.finetune_bert:
                out.append(orig_to_tok_map)
            else:
                out.append([])

            out.append(self.data[i]['ent_to_arg'])

            new_data.append(out)
        self.data = new_data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, idx):
        'Generates one sample of data'
        sample = self.data[idx]
        sent_token = sample[0]
        sent_pos = sample[1]
        sent_label_t = sample[2]
        sent_label_e = sample[3]
        sent_tri_arg_pairs = sample[4]  # each pair is (seq, seq)
        all_pairs = sample[5]  # each pair is(l_start, l_end, r_start, r_end, arg_role)
        sent_id = sample[6]
        glove_idx = sample[7]
        orig_to_tok_map = sample[8]
        ent_to_arg_dict = sample[9]
        return sent_token, sent_pos, sent_label_t, sent_label_e, sent_tri_arg_pairs, all_pairs, sent_id, glove_idx, orig_to_tok_map, ent_to_arg_dict

def pad_collate(batch):
    if len(batch) >= 1:
        # sort sents in each batch according to the sent len
        bs = list(zip(*[ex for ex in sorted(batch, key=lambda x: len(x[0]), reverse=True)]))
        lengths = [len(x) for x in bs[0]]
        sents = pad_sequence([torch.LongTensor(s) for s in bs[0]], batch_first=True, padding_value=TOKEN_PAD_ID)
        poss = pad_sequence([torch.LongTensor(s) for s in bs[1]], batch_first=True, padding_value=POS_PAD_ID)
        triggers = pad_sequence([torch.LongTensor(s) for s in bs[2]], batch_first=True, padding_value=TRI_PAD_ID)
        arguments = pad_sequence([torch.LongTensor(s) for s in bs[3]], batch_first=True, padding_value=ARGU_PAD_ID)
        seq_pairs = bs[4]
        all_pairs = bs[5]
        sent_ids = bs[6]

    return sent_ids, sents, poss, triggers, arguments, lengths, seq_pairs, all_pairs

def pad_collate_bert(batch):
    if len(batch) >= 1:
        # sort sents in each batch according to the sent len
        bs = list(zip(*[ex for ex in sorted(batch, key=lambda x: len(x[0]), reverse=True)]))
        lengths = [len(x) for x in bs[0]]
        bert_lengths = []
        sents = pad_sequence([torch.FloatTensor(s) for s in bs[0]], batch_first=True, padding_value=0.)
        poss = pad_sequence([torch.LongTensor(s) for s in bs[1]], batch_first=True, padding_value=POS_PAD_ID)
        triggers = pad_sequence([torch.LongTensor(s) for s in bs[2]], batch_first=True, padding_value=TRI_PAD_ID)
        arguments = pad_sequence([torch.LongTensor(s) for s in bs[3]], batch_first=True, padding_value=ARGU_PAD_ID)
        seq_pairs = bs[4]
        all_pairs = bs[5]
        sent_ids = bs[6]
        if len(bs[7]) > 0:
            glove_idx = pad_sequence([torch.LongTensor(s) for s in bs[7]], batch_first=True, padding_value=TOKEN_PAD_ID)
        else:
            glove_idx = None
        orig_to_tok_map = None

    return sent_ids, sents, poss, triggers, arguments, lengths, seq_pairs, all_pairs, glove_idx, orig_to_tok_map, bert_lengths

def pad_collate_bert_finetune(batch):
    if len(batch) >= 1:
        # sort sents in each batch according to the sent len
        bs = list(zip(*[ex for ex in sorted(batch, key=lambda x: len(x[2]), reverse=True)]))
        lengths = [len(x) for x in bs[2]]  # NOTE, here have to use the triggers as original length, b/c the length of tokens(bs[0]) has changed
        bert_lengths = [len(x) for x in bs[0]]
        sents = pad_sequence([torch.LongTensor(s) for s in bs[0]], batch_first=True, padding_value=TOKEN_PAD_ID)
        poss = pad_sequence([torch.LongTensor(s) for s in bs[1]], batch_first=True, padding_value=POS_PAD_ID)
        triggers = pad_sequence([torch.LongTensor(s) for s in bs[2]], batch_first=True, padding_value=TRI_PAD_ID)
        arguments = pad_sequence([torch.LongTensor(s) for s in bs[3]], batch_first=True, padding_value=ARGU_PAD_ID)
        seq_pairs = bs[4]
        all_pairs = bs[5]
        sent_ids = bs[6]
        glove_idx = pad_sequence([torch.LongTensor(s) for s in bs[7]], batch_first=True, padding_value=TOKEN_PAD_ID)  #None   # in finetune case, do not include glove_idx
        orig_to_tok_map = bs[8]
        ent_to_arg_dict = bs[9]

    return sent_ids, sents, poss, triggers, arguments, lengths, seq_pairs, all_pairs, glove_idx, orig_to_tok_map, bert_lengths, ent_to_arg_dict
