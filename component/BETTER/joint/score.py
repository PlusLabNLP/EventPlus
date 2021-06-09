from split_event import split_tri_output
from CRF_util import calculate_prob_byObser
import torch
import math
import pdb
import time

def predict_score_module(trig_topk_preds, trig_scores, argu_module, system_input, args):
    '''
    A wrapper that given beamed trigger predictions, generate argument predictions
    and calculate the corresponding scores for each tri_argu pair

    Need to call    split_event, score_util, calculate_prob_byObser

    Argument:
    trig_topk_preds, trig_scores is the output of kViterbi
        trig_topk_preds is a list of batch of a list of k(topk's k) of sequence list
        trig_scores is a list of batch of a sequence of scores (length k)
    argu_module is the argument module, which is a neural network
    system_input is a dict:
        {
            'sents': ..., in size (batch, sequence_length, emb_size)
            'sent_ids': ...,
            'poss': ...,
            'lengths': ...
            .
        }

    Output:
    output_data_dict
    '''
    if len(trig_scores.size()) == 1:
        trig_topk_preds = [[i] for i in trig_topk_preds]
        trig_scores = trig_scores.unsqueeze(1)
    # pdb.set_trace()
    #start_time = time.time()
    batch_size = len(trig_topk_preds)
    k_size = len(trig_topk_preds[0])
    max_len = system_input['lengths'][0]
    batch_output = list()
    for b, tri_pred in enumerate(trig_topk_preds):
        input_sents = system_input['sents'][b]
        assert input_sents.dim()==2
        sent_id = system_input['sent_ids'][b]
        pos_tag = system_input['poss'][b]
        assert pos_tag.dim()==1
        seq_lens = system_input['lengths'][b]
        trig_score_list = list()
        num_events = list()
        sent_list = list()
        pos_list = list()
        len_list = list()
        trig_idx_list = list()
        trig_type_list = list()
        for k, trigger in enumerate(tri_pred):
            # prepare input for argument module
            trig_score_list.append(trig_scores[b][k])
            trig_idx, trig_type = split_tri_output(trigger, args.B2I_trigger)# list(size k) of list
            n_event = len(trig_idx)
            num_events.append(n_event)
            if n_event == 0:
                sent_list.append(input_sents.unsqueeze(0))
                pos_list.append(pos_tag.unsqueeze(0))
                len_list.extend([seq_lens])
                trig_idx_list.append([])
                trig_type_list.append([])
            else:
                sent_list.append(input_sents.repeat(n_event, 1, 1))
                pos_list.append(pos_tag.repeat(n_event, 1))
                len_list.extend([seq_lens]*n_event)
                trig_idx_list.extend(trig_idx)
                trig_type_list.extend(trig_type)
        # crf_mask = torch.arange(max_len).expand(len(len_list), max_len) < torch.LongTensor(len_list).unsqueeze(1)
        assert args.use_crf
        assert args.use_att
        # tensorlize list
        tensor_sent = torch.cat(sent_list,0)
        tensor_pos = torch.cat(pos_list,0)
        if args.cuda:
            tensor_sent = tensor_sent.cuda()
            tensor_pos = tensor_pos.cuda()
            # crf_mask = crf_mask.cuda()
        # feed into argument module
        argu_pred, _, _, logits  = argu_module(tensor_sent, tensor_pos, len_list,
                                           task='argument',tri_idxs=trig_idx_list,
                                           att_pool=args.att_pool,att_mthd=args.att_mthd,
                                           crf=args.use_crf, seq_tags=None,
                                           crf_mask=None, use_att=args.use_att,
                                           k_tri=args.k_tri, k_arg=args.k_arg)
        assert len(trig_idx_list) == len(trig_type_list)
        total_process = len(trig_idx_list) #sum(num_events)
        assert len(argu_pred) == total_process

        #calculate scores & accumulate arguments predictions
        output_dict = dict()
        output_dict['sent_id']=sent_id
        output_dict['beam']=list()
        counter = 0
        # pdb.set_trace()
        for k, trigger in enumerate(tri_pred):
            tri_arg_pair = []
            if num_events[k] > 0:
                pred_arguments = argu_pred[counter:(counter+num_events[k])]
                pred_logits = logits[counter:(counter+num_events[k])]
                pred_trigger_prob = trig_score_list[k]
                pred_arguments = torch.LongTensor(pred_arguments)
                if args.cuda:
                    pred_arguments = pred_arguments.cuda()
                argu_seq_probs = calculate_prob_byObser(argu_module.crf_e, pred_logits, pred_arguments,None)
                beam_score = score_util(pred_trigger_prob, argu_seq_probs)
                # generate event-level tri-arg data
                for i in range(num_events[k]):
                    trigger_event = construct_event_trigger_seq(trigger, trig_idx_list[counter+i], trig_type_list[counter+i])
                    tri_arg_pair.append(([args._id_to_label_t[x] for x in trigger_event], [args._id_to_label_e[x] for x in pred_arguments[i].tolist()]))
                counter+=num_events[k]
            else:
                # no trigger in beam, just retrieve the predicted arguments
                pred_arguments = argu_pred[counter]
                pred_logits = logits[counter].unsqueeze(0)
                pred_trigger_prob = trig_score_list[k]
                pred_arguments = torch.LongTensor(pred_arguments).unsqueeze(0)
                if args.cuda:
                    pred_arguments = pred_arguments.cuda()
                argu_seq_probs = calculate_prob_byObser(argu_module.crf_e, pred_logits, pred_arguments, None)
                beam_score = score_util(pred_trigger_prob, argu_seq_probs)
                # generate event-level tri-arg data
                trigger_event = construct_event_trigger_seq(trigger, trig_idx_list[counter], trig_type_list[counter])
                tri_arg_pair = [([args._id_to_label_t[x] for x in trigger_event], [args._id_to_label_e[x] for x in pred_arguments[0].tolist()])]
                counter+=1
            output_dict['beam'].append({
                'beam_id':k,
                'pred_trigger_full': [args._id_to_label_t[x] for x in trigger],
                'pred_tri_arg_pair': tri_arg_pair,
                'pred_trigger_prob': pred_trigger_prob,
                'pred_arg_prob': argu_seq_probs,
                'beam_score': beam_score
            })

        batch_output.append(output_dict)
    #print("predict_score_module", time.time()-start_time)
    return batch_output

def construct_event_trigger_seq(sent_trigger_seq, tri_idx, tri_type):
    '''
    from sent-level trigger sequence to event level trigger sequence (single-event)
    '''
    seq_len = len(sent_trigger_seq)
    out_seq = [1] * seq_len
    if len(tri_idx) == 0:
        return out_seq
    elif len(tri_idx) == 1:
        # single-token trigger
        out_seq[tri_idx[0]] = tri_type
    else:
        # multi-token trigger
        l_idx = tri_idx[0]
        r_idx = tri_idx[-1]
        out_seq[l_idx] = tri_type
        out_seq[l_idx+1:r_idx+1] = [tri_type + 1] * (r_idx - l_idx)
    return out_seq

def gold_score_module(gold_trig, gold_arg, n_events, model, system_input, args):
    '''
    gold_trig, gold_arg are gold trig and arg label in batch, respectively
    directly feed in entire batch
    '''
    assert len(n_events) == gold_trig.size(0) #batch size
    sents_ext = system_input['sents_ext']
    poss_ext = system_input['poss_ext']
    lengths_ext = system_input['lengths_ext']
    sents = system_input['sents']
    poss = system_input['poss']
    lengths = system_input['lengths']
    tri_idxs = system_input['tri_idxs']
    tri_types = system_input['tri_types']
    sent_ids = system_input['sent_ids']

    max_len = lengths_ext[0]
    crf_mask_tri = torch.arange(max_len).expand(len(lengths), max_len) < torch.LongTensor(lengths).unsqueeze(1)
    crf_mask_arg = torch.arange(max_len).expand(len(lengths_ext), max_len) < torch.LongTensor(lengths_ext).unsqueeze(1)
    if args.cuda:
        crf_mask_tri = crf_mask_tri.cuda()
        crf_mask_arg = crf_mask_arg.cuda()
    # run trigger
    _, _, _, tri_logits  = model(sents, poss, lengths, task='trigger',
                        crf=args.use_crf, seq_tags=None,
                        crf_mask=crf_mask_tri, use_att=args.use_att,
                        k_tri=1, k_arg=1)
    # run argument
    _, _, _, arg_logits  = model(sents_ext, poss_ext, lengths_ext,
                        task='argument',tri_idxs=tri_idxs,
                        att_pool=args.att_pool,att_mthd=args.att_mthd,
                        crf=args.use_crf, seq_tags=None,
                        crf_mask=crf_mask_arg, use_att=args.use_att,
                        k_tri=1, k_arg=1)

    batch_output = list()
    counter = 0
    beam_id = 0
    for k in range(len(n_events)):  # batch size
        output_dict = dict()
        output_dict['sent_id']=sent_ids[k]
        output_dict['beam']=list()
        tri_arg_pair = []
        # retreive trigger seq and logit
        gold_triggers = gold_trig[k, :lengths[k]].unsqueeze(0)
        gold_tri_logits = tri_logits[k, :lengths[k]].unsqueeze(0)
        # tri_mask = crf_mask_tri[k].unsqueeze(0)
        # retreive arg seqs and logits
        if n_events[k] > 0:
            gold_arguments = gold_arg[counter:(counter+n_events[k]), :lengths[k]]
            gold_arg_logits = arg_logits[counter:(counter+n_events[k]), :lengths[k]]
            # arg_mask = crf_mask_arg[counter:(counter+n_events[k])]
            # generate event-level tri-arg data
            for i in range(n_events[k]):
                trigger_event = construct_event_trigger_seq(gold_triggers[0].tolist(), tri_idxs[counter+i], tri_types[counter+i])
                tri_arg_pair.append(([args._id_to_label_t[x] for x in trigger_event], [args._id_to_label_e[x] for x in gold_arguments[i].tolist()]))
            counter += n_events[k]
        else:
            gold_arguments = gold_arg[counter, :lengths[k]].unsqueeze(0)
            gold_arg_logits = arg_logits[counter, :lengths[k]].unsqueeze(0)
            # arg_mask = crf_mask_arg[counter].unsqueeze(0)
            # generate event-level tri-arg data
            trigger_event = construct_event_trigger_seq(gold_triggers[0].tolist(), tri_idxs[counter], tri_types[counter])
            tri_arg_pair=[([args._id_to_label_t[x] for x in trigger_event], [args._id_to_label_e[x] for x in gold_arguments[0].tolist()])]
            counter += 1
        # compute beam score
        assert gold_triggers.size(1) == gold_arguments.size(1)  # length should be equal
        gold_tri_prob = calculate_prob_byObser(model.crf_t, gold_tri_logits, gold_triggers, None)
        gold_arg_probs = calculate_prob_byObser(model.crf_e, gold_arg_logits, gold_arguments, None)
        beam_score = score_util(gold_tri_prob, gold_arg_probs)

        output_dict['beam'].append({
            'beam_id':beam_id,
            'gold_trigger_full': [args._id_to_label_t[x] for x in gold_triggers[0].tolist()],
            'gold_tri_arg_pair': tri_arg_pair,
            'gold_trigger_prob': gold_tri_prob,
            'gold_arg_prob': gold_arg_probs,
            'beam_score': beam_score
        })

        batch_output.append(output_dict)
    return batch_output

def score_util(tri_seq_prob, argu_seq_probs):
    '''
    The util function that calculate the score for each event pairs
    Current Idea is that perform averge of argument "probability" than multiply with trigger probability.
    Thus, in log space it's a logsumexp on argument score and an addition with trigger's score
    Args:
        tri_seq_prob is the a torch.FloatTensor that is the score of the trigger sequence (in log space)
        argu_seq_prob is a tensor of torch.FloatTensor that each of them is the score of the argument sequence (in log space)
    '''
    argu_scores = torch.logsumexp(argu_seq_probs,0)-math.log(argu_seq_probs.detach().size(0)+1e-9)
    # argu_scores = torch.mean(argu_seq_probs)
    return tri_seq_prob+1.0*argu_scores
