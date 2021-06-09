from dataclasses import dataclass
import torch
from torch.autograd import Variable
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
import time
from eval import *
from train import get_tri_idx_from_mix_sent, get_expand_trigger_seqs_from_idxs,\
        get_loss_mlp,get_prediction_crf,get_prediction_mlp
from score import predict_score_module, gold_score_module, construct_event_trigger_seq

class NNClassifier(nn.Module):
    def __init__(self, args, model):
        super(NNClassifier, self).__init__()
        self.args = args
        self.model = model
        self.logger = Logger(args.log_dir)
        self.best_model = model
        self.dev_cnt = 0

    def predict(self, data, test=False, use_gold=True):
        self.model.eval()
        if test:
            self.best_model.eval()
        if self.args.cuda:
            self.model.cuda()
            if test:
                self.best_model.cuda()
        count = 1
        y_trues_beam, y_preds_beam = [], []

        for sent_ids, sents, poss, triggers, arguments, lengths, seq_pairs, all_pairs in data:
            if self.args.use_crf:
                max_len = lengths[0]
                crf_mask = torch.arange(max_len).expand(len(lengths), max_len) < torch.LongTensor(lengths).unsqueeze(1)

            if self.args.cuda:
                sents = sents.cuda()
                poss = poss.cuda()
                arguments = arguments.cuda()
                triggers = triggers.cuda()
                if self.args.use_crf:
                    crf_mask = crf_mask.cuda()


            if use_gold:
                if self.args.use_att:
                    sent_ids_ext, sents_ext, poss_ext, args_ext, lengths_ext, tri_idxs, tri_types, n_events = self.expand_sents(sent_ids, sents, poss, lengths, seq_pairs)
                y_true_beam = self.get_y_trues_beam(triggers, args_ext, n_events, tri_idxs, tri_types, sent_ids, lengths, self.args)
            else:
                y_true_beam = []
            ##### inference y_hat and compute S(y_hat;x)
            ## first predict triggers
            if self.args.use_crf:
                if test:
                    out_t_k, out_t_k_prob, _, _ = self.best_model(sents, poss, lengths, task='trigger',
                                                   crf=True, seq_tags=None,
                                                   crf_mask=crf_mask, use_att=self.args.use_att,
                                                   k_tri=self.args.k_tri, k_arg=self.args.k_arg)
                else:
                    out_t_k, out_t_k_prob, _, _ = self.model(sents, poss, lengths, task='trigger',
                                                   crf=True, seq_tags=None,
                                                   crf_mask=crf_mask, use_att=self.args.use_att,
                                                   k_tri=self.args.k_tri, k_arg=self.args.k_arg)
            else:
                out_t, prob_t = self.model(sents, poss, lengths, task='trigger', use_att=self.args.use_att)
                loss_t = get_loss_mlp(lengths, triggers, out_t, criterion_t)

            ## next predict arguments
            system_input = {
                    'sents': sents,
                    'sent_ids': sent_ids,
                    'poss': poss,
                    'lengths': lengths
                    }
            if test:
                pred_output = predict_score_module(out_t_k, out_t_k_prob, self.best_model, system_input, self.args)
            else:
                pred_output = predict_score_module(out_t_k, out_t_k_prob, self.model, system_input, self.args)
            y_pred_beam, _ = self.inference(pred_output)
            # pdb.set_trace()

            # have to do this to release the tensors for prob scores, which are unused during evaluation
            # --> avoid CUDA OOM
            for i in y_pred_beam:
                del i['pred_trigger_prob']
                del i['pred_arg_prob']
                del i['beam_score']

            #####
            y_trues_beam.extend(y_true_beam)
            y_preds_beam.extend(y_pred_beam)

            # print('predict {} batch'.format(count))
            count += 1

        return y_trues_beam, y_preds_beam

    def get_y_trues_beam(self, gold_trig, gold_arg, n_events, tri_idxs, tri_types, sent_ids, lengths, args):
        '''
        same as the gold_score_module in `score.py`
        difference is just we don't pass the model --> directly get the trigger and arg labels
        '''
        batch_output = list()
        counter = 0
        beam_id = 0
        for k in range(len(n_events)):  # batch size
            output_dict = dict()
            output_dict['sent_id']=sent_ids[k]
            tri_arg_pair = []
            # retreive trigger seq
            gold_triggers = gold_trig[k, :lengths[k]].unsqueeze(0)
            # retreive arg seqs
            if n_events[k] > 0:
                gold_arguments = gold_arg[counter:(counter+n_events[k]), :lengths[k]]
                # generate event-level tri-arg data
                for i in range(n_events[k]):
                    trigger_event = construct_event_trigger_seq(gold_triggers[0].tolist(), tri_idxs[counter+i], tri_types[counter+i])
                    tri_arg_pair.append(([args._id_to_label_t[x] for x in trigger_event], [args._id_to_label_e[x] for x in gold_arguments[i].tolist()]))
                counter += n_events[k]
            else:
                gold_arguments = gold_arg[counter, :lengths[k]].unsqueeze(0)
                # generate event-level tri-arg data
                trigger_event = construct_event_trigger_seq(gold_triggers[0].tolist(), tri_idxs[counter], tri_types[counter])
                tri_arg_pair=[([args._id_to_label_t[x] for x in trigger_event], [args._id_to_label_e[x] for x in gold_arguments[0].tolist()])]
                counter += 1
            assert gold_triggers.size(1) == gold_arguments.size(1)  # length should be equal

            output_dict['beam_id'] = beam_id
            output_dict['gold_trigger_full'] = [args._id_to_label_t[x] for x in gold_triggers[0].tolist()]
            output_dict['gold_tri_arg_pair'] = tri_arg_pair

            batch_output.append(output_dict)
        return batch_output

    def expand_sents(self, sent_ids, sents, poss, lengths, seq_pairs=None):
        '''
        expand the sents, poss and arguments label tensors, from sentence level
        instancse to event(trigger) level instances for Attention usage

        NOTE: WARNING: current not support
        when developing settings for using predicted triggers, the `out_t` should be used, when using gold triggers
              the `out_t` can be set to None
        input:
            sents and poss: tensor of shape (batch_size, max_seq_len)
            triggers: tensor of shape (batch_size, max_seq_len)
            seq_pairs: list of length batch_size, each item is a list of (seq, seq) pairs, length == n_event in this sent
        '''
        sents_ext, poss_ext, args_ext, sent_ids_ext = [], [], [], []
        n_events = []
        lengths_ext = []
        tri_idxs = []
        tri_types = []

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
                for j in range(n_event):
                    # tri_idx = get_tri_idx(seq_pairs[i][j][0])
                    # tri_idxs.append(tri_idx)
                    # tri_types.append(tri_type)
                    tri_idx, tri_type = get_tri_idx_from_mix_sent(seq_pairs[i][j][0], self.args.B2I_trigger)
                    if len(tri_idx) == 0:
                        # tri_idx can be empty because the their might be a trigger word not captured in the `seq_pairs`
                        # ignore this instance
                        # NOTE: under current `json_to_pkl.py` this case will never happen
                        continue
                    else:
                        tri_idxs.append(tri_idx[0]) # tri_idx is a list of idxs, we know this `tri_idxs` is constructed from a SINGLE event
                        tri_types.append(tri_type[0])
                args_ext.extend([x[1] for x in seq_pairs[i]])

        sents_ext = pad_sequence([s for s in sents_ext], batch_first=True, padding_value=TOKEN_PAD_ID)
        poss_ext = pad_sequence([s for s in poss_ext], batch_first=True, padding_value=POS_PAD_ID)
        args_ext = pad_sequence([torch.LongTensor(s) for s in args_ext], batch_first=True, padding_value=ARGU_PAD_ID)

        assert args_ext.size(0) == poss_ext.size(0)
        assert args_ext.size(0) == sents_ext.size(0)
        assert args_ext.size(0) == len(tri_idxs)
        assert args_ext.size(0) == len(lengths_ext)
        assert len(tri_idxs) == len(tri_types), pdb.set_trace()

        return sent_ids_ext, sents_ext, poss_ext, args_ext, lengths_ext, tri_idxs, tri_types, n_events

    def _train(self, train_data, eval_data, test_data, train_task):
        if self.args.cuda:
            print("using cuda device: %s" % torch.cuda.current_device())
            assert torch.cuda.is_available()
            self.model.cuda()

        # criterion_t = nn.CrossEntropyLoss() #trigger
        # criterion_e = nn.CrossEntropyLoss() #argument

        if self.args.opt == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr)
        elif self.args.opt == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr, momentum=self.args.momentum)

        best_eval_f1 = 0.0
        best_epoch = 0
        patience = 0
        step = 0
        print(self.model)
        for epoch in range(self.args.epochs):
            print()
            print("*"*10+"Training Epoch #%s..." % epoch+"*"*10)
            epoch_start_time = time.time()
            self.model.train()
            epoch_loss = 0
            # start running batch
            if self.args.do_ssvm_train:
                for sent_ids, sents, poss, triggers, arguments, lengths, seq_pairs, all_pairs in train_data:
                    if self.args.use_crf:
                        max_len = lengths[0]
                        crf_mask = torch.arange(max_len).expand(len(lengths), max_len) < torch.LongTensor(lengths).unsqueeze(1)
                    if self.args.cuda:
                        sents = sents.cuda()
                        poss = poss.cuda()
                        triggers = triggers.cuda()
                        arguments = arguments.cuda()
                        if self.args.use_crf:
                            crf_mask = crf_mask.cuda()

                    self.model.zero_grad()
                    ##### inference for y and compute S(y;x)
                    if self.args.use_att:
                        sent_ids_ext, sents_ext, poss_ext, args_ext, lengths_ext, tri_idxs, tri_types, n_events = self.expand_sents(sent_ids, sents, poss, lengths, seq_pairs)
                        system_input = {
                                'sents_ext': sents_ext,
                                'poss_ext': poss_ext,
                                'sent_ids_ext': sent_ids_ext,
                                'lengths_ext': lengths_ext,
                                'sents': sents,
                                'poss': poss,
                                'sent_ids': sent_ids,
                                'lengths': lengths,
                                'tri_idxs': tri_idxs,
                                'tri_types': tri_types
                                }
                        # `triggers` is the UN-splited trigger, `args_ext` is arg seqs coresponds to splited triggers
                        gold_output = gold_score_module(triggers, args_ext, n_events, self.model, system_input, self.args)
                        y_golds, score_golds = self.get_gold_y_score(gold_output)
                    #####

                    ###### inference y_hat and compute S(y_hat;x)
                    ### first predict triggers
                    assert train_task in ['multitask_pipe', 'multitask_joint']
                    if self.args.use_crf:
                        out_t_k, out_t_k_prob, _, _, = self.model(sents, poss, lengths, task='trigger',
                                                       crf=True, seq_tags=None,
                                                       crf_mask=crf_mask, use_att=self.args.use_att,
                                                       k_tri=self.args.k_tri, k_arg=self.args.k_arg)
                    else:
                        out_t, prob_t = self.model(sents, poss, lengths, task='trigger', use_att=self.args.use_att)
                        loss_t = get_loss_mlp(lengths, triggers, out_t, criterion_t)

                    ## next predict arguments
                    system_input = {
                        'sents': sents,
                        'sent_ids': sent_ids,
                        'poss': poss,
                        'lengths': lengths
                    }

                    pred_output = predict_score_module(out_t_k, out_t_k_prob, self.model, system_input, self.args)
                    #start_time = time.time()
                    y_preds, score_preds = self.inference(pred_output)
                    #print("infererence", time.time()-start_time)
                    ######

                    # print('score_preds', score_preds)
                    # print('score_golds', score_golds)

                    loss = self.ssvm_loss(y_preds, score_preds, y_golds, score_golds, self.args.margin, self.args)
                    # losses = torch.cat([x['gold_arg_prob'] for x in y_golds])
                    # loss = torch.mean(losses)
                    # loss = -1 * loss

                    #print("score_preds", score_preds)
                    #print("score_golds", score_golds)
                    epoch_loss += loss
                    #start_time = time.time()
                    loss.backward()
                    self.logger.scalar_summary(tag='train/ssvm_loss',
                                               value=loss.item(),
                                               step=step)
                    #print('backward', time.time()-start_time)
                    optimizer.step()
                    if step % 50 == 0:
                        print('finished {} step'.format(step))
                    # print('loss {}'.format(loss))
                    step += 1
                print('epoch loss {}'.format(epoch_loss))
                print('{} epoch lasting for {}'.format(epoch, time.time()-epoch_start_time))
            # Evaluate at the end of each epoch
            print("*"*10+'Evaluating on Dev Set.....'+'*'*10)
            if len(eval_data) > 0:
                y_trues_beam, y_preds_beam = self.predict(eval_data, test=False, use_gold=True)
                out_pkl_dir = self.args.out_pkl_dir
                if self.args.trigger_type:
                    pkl_name = 'dev_ssvm_type.pkl'
                else:
                    pkl_name = 'dev_ssvm.pkl'
                if not os.path.isdir(out_pkl_dir):
                    os.makedirs(out_pkl_dir)
                self.write_pkl(y_preds_beam, out_pkl_dir, pkl_name)
                if len(y_trues_beam) > 0:
                    f1_t, acc_t = self.internal_trigger_eval(y_trues_beam, y_preds_beam)
                    print('trigger f1: {:.4f}, trigger acc: {:.4f}'.format(f1_t, acc_t))
                    eval_f1 = f1_t
                    self.logger.scalar_summary(tag='dev/trigger_f1',
                                               value=f1_t,
                                               step=epoch)
                print("====end2end eval====")
                _, f1_e2e = self.internal_e2e_arg_cls_eval(y_trues_beam, y_preds_beam, self.args)
                print('end2end eval f1 {}'.format(f1_e2e))

                # eval_f1 = (f1_t + f1_e2e) / 2
                eval_f1 = f1_e2e

                if eval_f1  > best_eval_f1:
                    print('Update eval f1: {}'.format(eval_f1))
                    best_eval_f1 = eval_f1
                    self.best_model = copy.deepcopy(self.model)
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
        y_trues_beam, y_preds_beam = self.predict(test_data, test=True, use_gold=True)
        test_f1_e = 0
        test_f1_t = 0
        if len(y_trues_beam) > 0:
            f1_t, acc_t = self.internal_trigger_eval(y_trues_beam, y_preds_beam)
            self.write_pkl(y_preds_beam, out_pkl_dir, self.args.out_pkl_name)
            print('trigger f1: {:.4f}, trigger acc: {:.4f}'.format(f1_t, acc_t))
            test_f1 = f1_t
            self.logger.scalar_summary(tag='test/trigger_f1',
                                       value=f1_t,
                                       step=0)

        print("====end2end eval====")
        _, test_f1_e2e = self.internal_e2e_arg_cls_eval(y_trues_beam, y_preds_beam, self.args)
        print('end2end eval f1 {}'.format(test_f1_e2e))

        return best_eval_f1, best_epoch, test_f1_e2e, test_f1_t


    def inference(self, score_data):

        y_preds = [] # each y_pred in y_preds is a list of dict, representing a beam
        score_preds = []

        sent_ids_uniq = get_uniq_id_list([i['sent_id'] for i in score_data])
        for sent_id in sent_ids_uniq:
            # gather all beams for the sentence
            preds = [x['beam'] for x in score_data if x['sent_id'] == sent_id]
            beams = [x for y in preds for x in y]  # flattern `beams` is now a list of all beams of the sentence
            beam_scores = torch.cat([x['beam_score'].unsqueeze(0) for x in beams])
            # select the beam that has the highest beam_score
            sel_score, sel_idx = torch.max(beam_scores, 0)
            sel_beam = beams[sel_idx]
            sel_beam['sent_id'] = sent_id
            y_preds.append(sel_beam)
            score_preds.append(sel_score)

        score_preds = torch.cat([x.unsqueeze(0) for x in score_preds])
        return y_preds, score_preds

    def get_gold_y_score(self, score_data_gold):

        # true label scores
        y_golds = []
        score_golds = []
        sent_ids_uniq = get_uniq_id_list([i['sent_id'] for i in score_data_gold])
        for sent_id in sent_ids_uniq:
            # gather y_true the sentence
            golds = [x['beam'] for x in score_data_gold if x['sent_id'] == sent_id]
            # y_true is the beam (only one beam)
            y_gold = [x for y in golds for x in y]  # `beams` is now a list of all beams of the sentence
            assert len(y_gold) == 1 # should only have ONE beam
            y_gold[0]['sent_id'] = sent_id  # add the sent_id key for evaluation purpose
            score_gold = y_gold[0]['beam_score']
            y_golds.append(y_gold[0])
            score_golds.append(score_gold)

        score_golds = torch.cat([x for x in score_golds], 0)
        return y_golds, score_golds


    def ssvm_loss(self, y_preds, score_preds, y_golds, score_golds, margin='e2e_eval', args=None):
        '''
        y_preds: the inference results from score_data of y_pred, holding one beam for each sentence
        score_data_true: score_data generated by the y_true
        margin: the delta term in SSVM loss, choices: ['const', 'e2e_eval']
        '''
        losses = []

        if margin == 'const':
            diff = score_preds - score_golds
        elif margin == 'e2e_eval':
            B2I_trigger = {'B-ANCHOR': 'I-ANCHOR'}
            B2I_arg = {'B-AGENT': 'I-AGENT', 'B-PATIENT': 'I-PATIENT'}
            f1s, _ = self.internal_e2e_arg_cls_eval(y_golds, y_preds, args)
            f1s = torch.FloatTensor(f1s)
            f1s = f1s.cuda(score_preds.get_device()) if score_preds.is_cuda else f1s
            deltas = 1.0 - f1s     # (1-f1) to make the delta larger when f1 is lower
            diff = score_preds - score_golds + deltas
        else:
            print('unsupported margin!!!')
            assert False

        # construct the final ssvm loss
        # for i in range(diff.size()[0]):
        #     if diff[i].cpu().data.numpy() <= 0.0:
        #         if args.cuda:
        #             losses.append(torch.FloatTensor([0.0]).cuda())
        #         else:
        #             losses.append(torch.FloatTensor([0.0]))
        #     else:
        #         losses.append(diff[i].reshape(1,))
        # loss = torch.mean(torch.cat(losses))
        if diff.is_cuda:
            loss_mask = diff > torch.zeros(diff.size()[0]).cuda()
            loss_mask = loss_mask.type(torch.ByteTensor).cuda()
        else:
            loss_mask = diff > torch.zeros(diff.size()[0])
            loss_mask = loss_mask.type(torch.ByteTensor)
        losses = torch.masked_select(diff, loss_mask)
        loss = torch.mean(losses)
        return loss

    def internal_e2e_arg_cls_eval(self, data_gold, data_pred, args):

        sent_ids_uniq = list(OrderedDict.fromkeys([i['sent_id'] for i in data_gold]))
        f1s = []
        n_corr_accum, n_true_accum, n_pred_accum = 0.0, 0.0, 0.0
        for sent_id in sent_ids_uniq:

            golds = [i for i in data_gold if i['sent_id'] == sent_id]
            preds = [i for i in data_pred if i['sent_id'] == sent_id]
            assert len(golds) == 1
            assert len(preds) == 1
            golds = golds[0]
            preds = preds[0]

            gold_triggers = [i[0] for i in golds['gold_tri_arg_pair']]
            gold_args = [i[1] for i in golds['gold_tri_arg_pair']]
            gold_trigger_objs = [iob_to_obj(i, args.B2I_trigger_string) for i in gold_triggers]
            gold_arg_objs = [iob_to_obj(i, args.B2I_arg_string) for i in gold_args]
            # sort the order of args of each event(trigger), according to the appearance order of text
            gold_arg_objs = [sorted(i, key=lambda x: x[1]) for i in gold_arg_objs]
            n_true = len([x for y in gold_arg_objs for x in y])
            n_true_accum += n_true

            pred_triggers = [i[0] for i in preds['pred_tri_arg_pair']]
            pred_args = [i[1] for i in preds['pred_tri_arg_pair']]
            pred_trigger_objs = [iob_to_obj(i, args.B2I_trigger_string) for i in pred_triggers]
            pred_arg_objs = [iob_to_obj(i, args.B2I_arg_string) for i in pred_args]
            # sort the order of args of each event(trigger), according to the appearance order of text
            pred_arg_objs = [sorted(i, key=lambda x: x[1]) for i in pred_arg_objs]
            n_pred = len([x for y in pred_arg_objs for x in y])
            n_pred_accum += n_pred

            n_match = 0.0
            for p_idx, p_tri in enumerate(pred_trigger_objs):
                # first, trigger should be correct
                if p_tri == []:
                    continue
                if (p_tri[0][1], p_tri[0][2]) in [(i[0][1], i[0][2]) for i in gold_trigger_objs if i != []]:  # strict match for only span, not type
                    g_idx = [(i[0][1], i[0][2]) for i in gold_trigger_objs if i != []].index((p_tri[0][1], p_tri[0][2]))
                    # g_idx = gold_trigger_objs.index(p_tri)
                    gold_args = gold_arg_objs[g_idx]
                    pred_args = pred_arg_objs[p_idx]
                    # second, retrieve how many argument matches are there given the correctly predicted trigger
                    n_match += score_match(gold_args, pred_args)
            n_corr_accum += n_match

            prec, recall, f1 = cal_prec_recall_f1(n_match, n_pred, n_true)
            f1s.append(f1)
        # print('n_corr {}, n_pred {}, n_true {}'.format(n_corr_accum, n_pred_accum, n_true_accum))
        prec_accum, recall_accum, f1_accum = cal_prec_recall_f1(n_corr_accum, n_pred_accum, n_true_accum)
        return f1s, f1_accum
    def internal_trigger_eval(self, y_trues_beam, y_preds_beam):
        y_trues_t = [x['gold_trigger_full'] for x in y_trues_beam]
        y_preds_t = [x['pred_trigger_full'] for x in y_preds_beam]
        f1 = f1_score(y_trues_t, y_preds_t)
        acc = accuracy_score(y_trues_t, y_preds_t)
        # print(classification_report(y_trues_t, y_preds_t))
        return f1, acc
    def write_pkl(self, y_preds_beam, out_pkl_dir, pkl_name):
        with open('{}/{}'.format(out_pkl_dir, pkl_name), 'wb') as f:
            pickle.dump(y_preds_beam, f)
        print('pkl saved at {}/{}'.format(out_pkl_dir, pkl_name))

    def train_epoch(self, train_data, dev_data, test_data, task):
        best_f1, best_epoch, test_f1_e, test_f1_t = self._train(train_data, dev_data, test_data, task)
        print("Final Dev F1: %.4f" % best_f1)
        return best_f1, best_epoch, test_f1_e, test_f1_t

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.args
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed ... continuing anyway.]")

    def load(self, filename=None, filename_t=None, filename_e=None):
        try:
            if filename:
                # multitask model
                if self.args.cuda:
                    device = torch.device('cuda')
                else:
                    device = torch.device('cpu')
                checkpoint = torch.load(filename, map_location=device)
                self.model.load_state_dict(checkpoint['model'])
                self.best_model.load_state_dict(checkpoint['model'])
                print("multitask model load from {}".format(filename))
        except BaseException as e:
            print(e)
            print("Cannot load model from {}".format(filename))
