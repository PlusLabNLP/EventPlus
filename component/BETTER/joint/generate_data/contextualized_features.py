import torch
import argparse
import pickle
from util import *
from transformers import *
import tqdm

MODELS = [(XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-large')]
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
        # Encode text
        ori_sent = clean_ori_sent(ori_sent)
        input_ids = torch.tensor([tokenizer.encode(ori_sent, add_special_tokens=True)], device=device)
        input_tok_list = [tokenizer.decode([x]) for x in input_ids[0]]
        assert input_ids.size(1) == len(input_tok_list)
        try:
            alignment = align_bpe_to_words(input_tok_list, tokens)
        except:
            # print('Align BPE failed. Skipped')
            continue
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]
        features = align_features_to_words((last_hidden_states[0]).cpu(), alignment)
        try:
            assert features.size(0) == len(tokens)
        except:
            print('Align contextualized features failed. Skipped')
            continue
        d['contextual_feature'] = features
        output.append(d)
        cnt += 1
    print(cnt)
    with open(args.output_file, 'wb') as of:
        pickle.dump(output, of)
