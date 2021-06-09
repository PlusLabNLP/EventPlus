# A fair portion of this code ('align_bpe_to_words') is taken from: https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/alignment_utils.py
# Author: sidvash
# Created: 10/28/2019
# Last modified: 11/19/2019

'''
The purpose of this code is to extract RoBERTa embeddings for a sentence whose gold tokens are known.

Usage:
from roberta_extract import aligned_roberta
embeddings = aligned_roberta(sentence, tokens, roberta='large')

where sentence is a string, and tokens are the tokens of the sentence.
'''
from collections import Counter
from typing import List

import torch
import fairseq #not importing this causes line 94 assertion to fail -- why?

##### Load Roberta model
roberta_large = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta_large.eval()
print("Large Model loaded")

roberta_base = torch.hub.load('pytorch/fairseq', 'roberta.base')
roberta_base.eval()
print("Base Model loaded")

def aligned_roberta(sentence: str, 
                            tokens: List[str], 
                            roberta='large',
                            return_all_hiddens=False,
                            border_tokens=False):
    '''
    Code inspired from: https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/hub_interface.py
    
    Aligns roberta embeddings for an input tokenization of words for a sentence
    
    Inputs:
    1. sentence: sentence in string
    2. tokens: tokens of the sentence in which the alignment is to be done
    3. roberta: 'large' or 'base'
    4. border_tokens: Boolean for whether to include special token embeddings <s> and </s>

    Outputs:    
    Roberta embeddings aligned with the input tokens 
    '''

    # tokenize both with GPT-2 BPE and get alignment with given tokens
    if roberta=='large':
        roberta_model = roberta_large
    else:
        roberta_model = roberta_base

    bpe_toks = roberta_model.encode(sentence)
    alignment = align_bpe_to_words(roberta_model, bpe_toks, tokens)
    
    
    # extract features and align them
    features = roberta_model.extract_features(bpe_toks, return_all_hiddens=return_all_hiddens)
    features = features.squeeze(0)   #Batch-size = 1
    aligned_feats = align_features_to_words(roberta_model, features, alignment)
    
    if border_tokens:
        return aligned_feats
    else:
        return aligned_feats[1:-1]  #exclude <s> and </s> tokens


def align_bpe_to_words(roberta, bpe_tokens: torch.LongTensor, other_tokens: List[str]):
    """
    Helper to align GPT-2 BPE to other tokenization formats (e.g., spaCy).

    Args:
        roberta (RobertaHubInterface): RoBERTa instance
        bpe_tokens (torch.LongTensor): GPT-2 BPE tokens of shape `(T_bpe)`
        other_tokens (List[str]): other tokens of shape `(T_words)`

    Returns:
        List[str]: mapping from *other_tokens* to corresponding *bpe_tokens*.
    """
    assert bpe_tokens.dim() == 1
    assert bpe_tokens[0] == 0. ##added after revision in alignment utils from fairseq (Feb11, 2020)

    def clean(text):
        return text.strip()

    # remove whitespaces to simplify alignment
    bpe_tokens = [roberta.task.source_dictionary.string([x]) for x in bpe_tokens]
    bpe_tokens = [clean(roberta.bpe.decode(x) if x not in {'<s>', ''} else x) for x in bpe_tokens]
    other_tokens = [clean(str(o)) for o in other_tokens]

    # strip leading <s>
    
    bpe_tokens = bpe_tokens[1:]
    assert ''.join(bpe_tokens) == ''.join(other_tokens)

    # create alignment from every word to a list of BPE tokens
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


def align_features_to_words(roberta, features, alignment):
    """
    Align given features to words.

    Args:
        roberta (RobertaHubInterface): RoBERTa instance
        features (torch.Tensor): features to align of shape `(T_bpe x C)`
        alignment: alignment between BPE tokens and words returned by
            func:`align_bpe_to_words`.
    """
    assert features.dim() == 2

    bpe_counts = Counter(j for bpe_indices in alignment for j in bpe_indices)
    assert bpe_counts[0] == 0  # <s> shouldn't be aligned
    denom = features.new([bpe_counts.get(j, 1) for j in range(len(features))])
    weighted_features = features / denom.unsqueeze(-1)

    output = [weighted_features[0]]
    largest_j = -1
    for bpe_indices in alignment:
        output.append(weighted_features[bpe_indices].sum(dim=0))
        largest_j = max(largest_j, *bpe_indices)
    for j in range(largest_j + 1, len(features)):
        output.append(weighted_features[j])
    output = torch.stack(output)
    #assert torch.all(torch.abs(output.sum(dim=0) - features.sum(dim=0)) < 1e-4)
    return output



