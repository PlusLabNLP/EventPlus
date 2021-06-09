# Assert the torchcrf version is 0.7.2
# allennlp version is 0.9.1
import torch
import heapq
import numpy as np
import time
from torch.nn.utils.rnn import pad_sequence
from allennlp.nn.util import viterbi_decode

def calculate_prob_byObser(crf_obj, emissions, observation, mask):
    '''
    Given padded sequence of crf_path, calculate corresponding score for path
    Args:
        crf_obj : torchcrf object
        emissions (`~torch.Tensor`): Emission score tensor of size
            ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
            ``(batch_size, seq_length, num_tags)`` otherwise.
        observation (`~torch.Tensor`): ``size (seq_length, batch_size)`` if ``batch_first is ``False``,
            ``(batch_size, seq_length)`` otherwise.
        mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
            if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
    Returns:
        torch.FloatTensor in size (batch) # log prob.
    '''
    if mask is None:
        mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

    if crf_obj.batch_first:
        emissions = emissions.transpose(0, 1)
        mask = mask.transpose(0, 1)
        obser = observation.transpose(0, 1)
    numerator = crf_obj._compute_score(emissions, obser, mask)
    denominator = crf_obj._compute_normalizer(emissions, mask)
    return numerator - denominator

def pad_seq(best_path, seq_length, batch_first=True, padding_value=0):
    assert batch_first
    batch = []
    for path in best_path:
        ori_len = len(path)
        pads = [padding_value]*(seq_length-ori_len)
        batch.append(path+pads)
    return torch.LongTensor(batch)

def kViterbi(crf_obj, emissions, topK, mask):
    """
    Find the k-best tag sequence using modified Viterbi algorithm.
    Args:
        crf_obj : torchcrf object
        emissions (`~torch.Tensor`): Emission score tensor of size
            ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
            ``(batch_size, seq_length, num_tags)`` otherwise.
        topK (int): How many path want to consider
        mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
            if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
    Returns:
        List of list containing the best tag sequence for each batch.
    """
    assert topK >=1
    if topK == 1:
        seq_length = emissions.size(1)
        best_path = crf_obj.decode(emissions, mask)
        observation = pad_seq(best_path, seq_length, crf_obj.batch_first, 0)
        best_probs = calculate_prob_byObser(crf_obj, emissions, observation, mask)
        return best_path, best_probs.squeeze()

    crf_obj._validate(emissions, mask=mask)
    if mask is None:
        mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)
    if crf_obj.batch_first:
        emissions = emissions.transpose(0, 1)
        mask = mask.transpose(0, 1)
    normalizer = crf_obj._compute_normalizer(emissions, mask)
    # ===============start main part========================
    # emissions: (seq_length, batch_size, num_tags)
    # mask: (seq_length, batch_size)
    assert emissions.dim() == 3 and mask.dim() == 2
    assert emissions.shape[:2] == mask.shape
    assert emissions.size(2) == crf_obj.num_tags
    assert mask[0].all()

    seq_length, batch_size = mask.shape

    # Start transition and first emission
    # score is a tensor of size(batch_size, num_tags, topK) where for each
    # batch, value at tags i and top j stores the scores of the j-th best tag
    # sequence so far that ends with tag i
    #
    # pre_states saves the previous tag where the j-th best path that ends with tag i currently
    score = emissions.new_zeros((seq_length, batch_size, crf_obj.num_tags, topK))
    score[0,:,:,0] = crf_obj.start_transitions + emissions[0] # batch x num_tags

    pre_states = np.zeros((seq_length, batch_size, crf_obj.num_tags, topK), int)
    for i in range(crf_obj.num_tags):
        for b in range(batch_size):
            for k in range(topK):
                pre_states[0,b,i,k] = i # should be start transition

    # The ranking of multiple paths through same state
    rank = np.zeros((seq_length, batch_size, crf_obj.num_tags, topK), int)
    for t in range(1, seq_length):
        next_score_list = []
        for k in range(topK):
            broadcast_score = score[t-1,:,:,k].unsqueeze(2) #(batch_size, num_tags, 1)
            broadcast_emissions = emissions[t].unsqueeze(1) #(batch_size, 1, num_tags)

            # Compute the score tensor of size (batch_size, num_tags, num_tags)
            # where for each sample, entry at row i and column j stores
            # the sum of scores of all possible tag sequences so far that end
            # with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + crf_obj.transitions + broadcast_emissions
            next_score_list.append(next_score)

        for b in range(batch_size):
            if mask[t,b]:
                for cur_state in range(crf_obj.num_tags):
                    h = []
                    for pre_state in range(crf_obj.num_tags):
                        for k in range(topK):
                            heapq.heappush(h, (-1*next_score_list[k][b, pre_state, cur_state], pre_state))

                    # Get the sorted list
                    h_sorted = [heapq.heappop(h) for _ in range(topK)] #get topK path into cur_state
                    # We need to keep a ranking if a path crosses a state more than once
                    rankDict = dict()
                    # Retain the topK scoring paths
                    for k in range(topK):
                        score[t, b, cur_state, k] = score[t, b, cur_state, k] + (h_sorted[k][0].data * -1)
                        pre_states[t, b, cur_state, k] = h_sorted[k][1]
                        state = h_sorted[k][1]
                        if state in rankDict:
                            rankDict[state] = rankDict[state]+1
                        else:
                            rankDict[state] = 0
                        rank[t, b, cur_state, k] = rankDict[state]
            else:
                for cur_state in range(crf_obj.num_tags):
                    for k in range(topK):
                        score[t, b, cur_state, k]=score[t-1, b, cur_state, k]


    batch_path = []
    batch_path_prob = []
    seq_ends = mask.long().sum(dim=0) - 1 # seq_len x batch # assume seq_ends=8, seq_len=9
    for b in range(batch_size):
        h = []
        for cur_state in range(crf_obj.num_tags):
            for k in range(topK):
                heapq.heappush(h, ( -1 * (score[seq_ends[b], b, cur_state, k]+crf_obj.end_transitions[cur_state]),
                                   cur_state, k))
        h_sorted = [heapq.heappop(h) for _ in range(topK)]
        k_list = np.zeros((topK, seq_ends[b]+1), int) # k x 9
        k_list_probs = list()
        for k in range(topK):
            prob = h_sorted[k][0]
            state = h_sorted[k][1]
            rankK = h_sorted[k][2]

            k_list_probs.append((prob*-1)-(normalizer[b]))
            k_list[k][seq_ends[b]] = state # assign index 8 == last one
            for t in range(seq_ends[b]-1, -1, -1): # t = 7,6,5,4,3,2,1,0
                nextState = k_list[k][t+1]
                preState = pre_states[t+1, b, nextState, rankK]
                k_list[k][t] = preState
                rankK = rank[t+1,b,nextState,rankK]
        batch_path.append(k_list.tolist())
        batch_path_prob.append(k_list_probs)
    if crf_obj.batch_first:
        batch_probs = recalculate_probs(crf_obj, batch_path, emissions.transpose(0,1), mask.transpose(0,1), topK)
    else:
        batch_probs = recalculate_probs(crf_obj, batch_path, emissions, mask, topK)
    return batch_path, batch_probs

def recalculate_probs(crf_obj, batch_path, emissions, mask, topK):
    '''
    batch_path: List(batch) of List(k) of int
    emissions' and mask's batch_first should align with crf_obj
    '''
    if crf_obj.batch_first:
        batch_size = emissions.size(0)
    else:
        batch_size = emissions.size(1)

    batch_probs = []
    for k in range(topK):
        candidate = []
        for b in range(batch_size):
            candidate.append(batch_path[b][k])
        observation = pad_sequence([torch.LongTensor(s) for s in candidate],
                                   batch_first=crf_obj.batch_first,
                                   padding_value=0)
        batch_probs.append(calculate_prob_byObser(crf_obj, emissions, observation, mask))

    return torch.stack(batch_probs, dim=0).transpose(0,1)
