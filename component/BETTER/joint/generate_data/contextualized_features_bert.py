import torch
import argparse
import pickle
from util import *
from transformers import *
import pdb
import tqdm

def bert_token(sent_orig_tokens, tokenizer):

    orig_to_tok_map = []
    sent_bert_tokens = []
    sent_bert_ids = []
    sent_bert_tokens.append("[CLS]")
    sent_bert_ids.extend(tokenizer.encode("[CLS]", add_special_tokens=False))

    for idx, orig_token in enumerate(sent_orig_tokens):
        orig_to_tok_map.append(len(sent_bert_tokens))
        # if orig_token != ' ':
        #     sent_bert_tokens.extend(tokenizer.tokenize(orig_token))
        #     sent_bert_ids.extend(tokenizer.encode(orig_token, add_special_tokens=False))
        # else:
        #     sent_bert_ids.extend(tokenizer.convert_tokens_to_ids([orig_token]))
        #     sent_bert_tokens.extend(tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids([orig_token])))
        if len(tokenizer.tokenize(orig_token)) > 0:
            sent_bert_tokens.extend(tokenizer.tokenize(orig_token))
            sent_bert_ids.extend(tokenizer.encode(orig_token, add_special_tokens=False))
        elif len(tokenizer.tokenize(orig_token)) == 0:
            # case of some special chars that cause bert tokenizer return empty
            sent_bert_ids.extend(tokenizer.convert_tokens_to_ids([orig_token]))
            sent_bert_tokens.extend(tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids([orig_token])))
    sent_bert_tokens.append("[SEP]")
    sent_bert_ids.extend(tokenizer.encode("[SEP]", add_special_tokens=False))
    return sent_bert_tokens, sent_bert_ids, orig_to_tok_map

def get_bert_embedding(last_hid_state, orig_to_tok_map):
    '''
    last_hid_state is a tensor of shape (batch_size, seq_len, hid_dim)
    orig_to_tok_map is a list, len(orig_to_tok_map) = len(sent_orig_tokens)
    '''
    out_feats = []
    for orig_idx, bert_idx in enumerate(orig_to_tok_map):
        if orig_idx != len(orig_to_tok_map) - 1:
            sel_idx = list(range(orig_to_tok_map[orig_idx], orig_to_tok_map[orig_idx+1]))
        else:
            # last token
            sel_idx = list(range(orig_to_tok_map[orig_idx], last_hid_state.size(1) - 1))  # do not use the [SEP] representation
        sel = last_hid_state[:, sel_idx, :]
        sel_mean = torch.mean(sel, dim=1, keepdim=True)
        out_feats.append(sel_mean)
    out_feats = torch.cat(out_feats, dim=1)
    return out_feats

if __name__ == '__main__':
    # MODELS = [(XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-large')]
    MODELS = [(BertModel, BertTokenizer, 'bert-large-uncased')]
    #MODELS = [(RobertaModel,    RobertaTokenizer,    'roberta-large')]

    p = argparse.ArgumentParser()
    p.add_argument('input_file', type=str,
                   help="Input pkl file (converted from internal JSON)")
    p.add_argument('output_file', type=str,
                   help="Where to save the output features pkl file")
    args = p.parse_args()

    for model_class, tokenizer_class, pretrained_weights in MODELS:
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        data = pickle.load(open(args.input_file, 'rb'))
        output = list()
        cnt = 0
        for d in tqdm.tqdm(data):
            ori_sent = d['ori_sent']
            tokens = d['tokens']
            sent_bert_tokens, sent_bert_ids, orig_to_tok_map = bert_token(tokens, tokenizer)
            assert len(sent_bert_tokens) == len(sent_bert_ids)
            assert len(tokens) == len(orig_to_tok_map)
            with torch.no_grad():
                bert_output = model(torch.tensor([sent_bert_ids]).to(device))
            last_hid_state = bert_output[0].cpu()
            out_feats = get_bert_embedding(last_hid_state, orig_to_tok_map)
            assert out_feats.size(1) == len(tokens)  # orig seq_len
            d['contextual_feature'] = out_feats.squeeze(0)
            output.append(d)
            cnt += 1
        print(cnt)
        with open(args.output_file, 'wb') as of:
            pickle.dump(output, of)
