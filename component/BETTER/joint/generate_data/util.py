from typing import List
from collections import Counter
import torch
import re

def clean_ori_sent(ori_sent):
    ori_sent = re.sub(r"\.\.\.\.", "...", ori_sent)
    ori_sent = re.sub(r"---", "--", ori_sent)
    ori_sent = re.sub(r"``", '"', ori_sent)
    ori_sent = re.sub(r"''", '"', ori_sent)
    ori_sent = re.sub(r"`", "'", ori_sent)
    ori_sent = re.sub(r"\.{3,}", "...", ori_sent)
    ori_sent = re.sub(r"etc\.$", "etc. .", ori_sent)
    ori_sent = re.sub(r"etc\.\)$", "etc. .)", ori_sent)
    return ori_sent

def align_bpe_to_words(bert_tokens: List[str], other_tokens: List[str]):
    def clean(text):
        text = text.strip()
        if text=='---':
            return '--'
        else:
            return text
    def clean_stanford(text):
        text = text.strip()
        text = text.replace(u"\xa0", "")
        text = re.sub(r"-LRB-", '(', text)
        text = re.sub(r"-RRB-", ')', text)
        text = re.sub(r"-LSB-", '[', text)
        text = re.sub(r"-RSB-", ']', text)
        text = re.sub(r"-LCB-", '{', text)
        text = re.sub(r"-RCB-", '}', text)
        text = re.sub(r"``", '"', text)
        text = re.sub(r"''", '"', text)
        text = re.sub(r"`", "'", text)
        text = re.sub(r"---------", "------", text)
        text = re.sub(r"---------------------", "-------------------", text)
        if text =='-------------------':
            return '--------------------'
        if text =='------------':
            return '-----------'
        return text

    # remove whitespaces to simplify alignment
    bpe_tokens = []
    for o in bert_tokens:
        if o not in {'<s>', '</s>'}:
            bpe_tokens.append(clean(str(o)))
    other_tokens = [clean_stanford(str(o)) for o in other_tokens]
    try:
        assert ''.join(bpe_tokens) == ''.join(other_tokens)
    except AssertionError:
        if (len(''.join(bpe_tokens))+1==len(''.join(other_tokens))) and (other_tokens[-1]=='.'):
            bpe_tokens[-1]+='.'
        assert ''.join(bpe_tokens) == ''.join(other_tokens)

    # create alignment
    alignment = []
    bpe_toks = filter(lambda item: item[1] != '', enumerate(bpe_tokens, start=1))
    j, bpe_tok = next(bpe_toks)
    for other_tok in other_tokens:
        bpe_indices = []
        while True:
            if other_tok.startswith(bpe_tok):
                bpe_indices.append(j)
                other_tok = other_tok[len(bpe_tok):]
                try:
                    j, bpe_tok = next(bpe_toks)
                except StopIteration:
                    j, bpe_tok = None, None
            elif bpe_tok.startswith(other_tok):
                # other_tok spans multiple BPE tokens
                bpe_indices.append(j)
                bpe_tok = bpe_tok[len(other_tok):]
                other_tok = ''
            else:
                raise Exception('Cannot align "{}" and "{}"'.format(other_tok, bpe_tok))
            if other_tok == '':
                break
        assert len(bpe_indices) > 0
        alignment.append(bpe_indices)
    assert len(alignment) == len(other_tokens)
    return alignment

def align_features_to_words(features, alignment):
    assert features.dim() == 2
    bpe_counts = Counter(j for bpe_indices in alignment for j in bpe_indices)
    assert bpe_counts[0] == 0  # <s> shouldn't be aligned
    denom = features.new([bpe_counts.get(j, 1) for j in range(len(features))])
    weighted_features = features / denom.unsqueeze(-1)
    #output = [weighted_features[0]] # <s>
    output = []
    #largest_j = -1
    for bpe_indices in alignment:
        output.append(weighted_features[bpe_indices].sum(dim=0))
        #largest_j = max(largest_j, *bpe_indices)
    #for j in range(largest_j + 1, len(features)):
    #    output.append(weighted_features[j])
    output = torch.stack(output)
    return output

def spacy_nlp():
    if getattr(spacy_nlp, '_nlp', None) is None:
        try:
            from spacy.lang.en import English
            spacy_nlp._nlp = English()
        except ImportError:
            raise ImportError('Please install spacy with: pip install spacy')
    return spacy_nlp._nlp

def spacy_tokenizer():
    if getattr(spacy_tokenizer, '_tokenizer', None) is None:
        try:
            nlp = spacy_nlp()
            spacy_tokenizer._tokenizer = nlp.Defaults.create_tokenizer(nlp)
        except ImportError:
            raise ImportError('Please install spacy with: pip install spacy')
    return spacy_tokenizer._tokenizer

def correct_unmatch(tokens, new_tok, features):
    tok_aln, new_tok_aln, tok_id, new_tok_id = minEditMatching(tokens, new_tok)
    assert len(new_tok_id)==0
    new_fea = list()
    for idx, tid in enumerate(tok_id[::-1]):
        assert new_tok[tid]+new_tok[tid+1] == tokens[tid-idx]
    idx = 0
    lens = len(features)
    while(idx < lens):
        if idx not in tok_id:
            new_fea.append(features[idx])
            idx += 1
        else:
            new_fea.append(np.mean(features[idx:idx+2], axis=0))
            idx += 2
    assert len(new_fea)==len(tokens)
    for n in new_fea:
        assert n.size==1024
    return new_fea

def minEditMatching(target, source):
    ''' Return a pair of aligned target and source'''
    n = len(target)
    m = len(source)
    distance = [[0 for i in range(m+1)] for j in range(n+1)]
    for i in range(1,n+1):
        #distance[i][0] = distance[i-1][0] + insertCost(target[i-1])
        distance[i][0] = distance[i-1][0] + 1

    for j in range(1,m+1):
        #distance[0][j] = distance[0][j-1] + deleteCost(source[j-1])
        distance[0][j] = distance[0][j-1] + 1

    for i in range(1,n+1):
        for j in range(1,m+1):
            distance[i][j] = min(distance[i-1][j-1]+substCostSen(source[j-1],target[i-1]),
                                 distance[i-1][j]+1,
                                 distance[i][j-1]+1)
    ii = n
    jj = m

    target_aln = []
    source_aln = []
    target_id = []
    source_id = []
    while (ii > 0) or (jj > 0):
        if distance[ii][jj]-substCostSen(source[jj-1],target[ii-1]) == distance[ii-1][jj-1]:
            target_aln.append(target[ii-1])
            source_aln.append(source[jj-1])
            ii -= 1
            jj -= 1
        elif distance[ii][jj] - 1 == distance[ii][jj-1]:
            source_aln.append(source[jj-1])
            target_aln.append("___")
            jj -= 1
            target_id.append(jj)
        elif distance[ii][jj] - 1 == distance[ii-1][jj]:
            source_aln.append("___")
            target_aln.append(target[ii-1])
            ii -= 1
            source_id.append(ii)
        else:
            print ("error!")

    target_aln = target_aln[::-1]
    source_aln = source_aln[::-1]
    return (target_aln,source_aln,target_id, source_id)

def substCostSen(x,y):
    if x==y:
        return 0
    else:
        return 1


