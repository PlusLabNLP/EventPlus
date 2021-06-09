import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch.nn import MSELoss, L1Loss, SmoothL1Loss, CrossEntropyLoss
from sklearn.metrics import precision_score, f1_score, recall_score
from scipy.stats import spearmanr

from preprocess_udst import UDS_T_Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Multitask_Baseline(nn.Module):
    def __init__(self, 
                freeze_bert=False,
                num_duration_classes=11,
                num_relation_output=4,
                span_based=True,
                root_based=False,
                alpha_for_dur_loss=2,
                beta_for_rel_loss=1,
                dur_loss_weight = torch.ones(11)
            ):
        super(Multitask_Baseline, self).__init__()

        self.span_based = span_based
        self.root_based = root_based
        self.beta_for_rel_loss = beta_for_rel_loss
        self.alpha_for_dur_loss = alpha_for_dur_loss
        self.dur_loss_weight = dur_loss_weight
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # should we do max pool if averages are too small (after masking) + unfair for different length of word pieces? yes
        self.max_pool = nn.AdaptiveMaxPool2d((1, self.bert.config.hidden_size))
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, self.bert.config.hidden_size))
        self.attn_layer = nn.Linear(self.bert.config.hidden_size * 2, self.bert.config.hidden_size)
        
        self.duration_mlp = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.Dropout(0.5),
            nn.Linear(self.bert.config.hidden_size, num_duration_classes), # do softmax
        )
        
        self.relation_mlp = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size * 2, self.bert.config.hidden_size),
            nn.Dropout(0.5),
            nn.Linear(self.bert.config.hidden_size, num_relation_output), # normalize to 0-1? (sigmoid?)
        )

        # for loss
        self.cross_entropy_loss = CrossEntropyLoss(weight = self.dur_loss_weight)
        self.l1_loss = L1Loss(reduction='sum')

    def forward(self, tokens, span1_masks, span2_masks, root1_masks, root2_masks):
        """
        inputs:
            tokens: [B, L]
            span1_masks: [B, L]
            span2_masks: [B, L]
            root1_masks: [B, L]
            root2_masks: [B, L]
        """
        bert_hidden_states = self.bert(tokens)[0] # last hidden states, [B, L, D]
        sequence_pooled = self.avg_pool(bert_hidden_states).squeeze(1) # [B, D]
        
        if self.root_based:
            pred1_masked_hidden_states = bert_hidden_states * root1_masks.unsqueeze(-1) # [B, L, D]
            pred2_masked_hidden_states = bert_hidden_states * root2_masks.unsqueeze(-1) # [B, L, D]
        elif self.span_based:
            pred1_masked_hidden_states = bert_hidden_states * span1_masks.unsqueeze(-1) # [B, L, D]
            pred2_masked_hidden_states = bert_hidden_states * span2_masks.unsqueeze(-1) # [B, L, D]

        pred1_pooled = self.max_pool(pred1_masked_hidden_states).squeeze(1) # [B, D]
        pred2_pooled = self.max_pool(pred2_masked_hidden_states).squeeze(1) # [B, D]
        
        # additional attention layer
        pred1_contextualized = self.attn_layer(torch.cat((pred1_pooled, sequence_pooled), dim=1))
        pred2_contextualized = self.attn_layer(torch.cat((pred2_pooled, sequence_pooled), dim=1))

        dur1_final = self.duration_mlp(pred1_contextualized) # [B, 11]
        dur2_final = self.duration_mlp(pred2_contextualized) # [B, 11]
        
        rel_final = torch.sigmoid(self.relation_mlp(torch.cat((pred1_contextualized, pred2_contextualized), dim=1))) # [B, 4]
        
        return dur1_final, dur2_final, rel_final

    def get_losses(self, predictions, ground_truth):
        '''
        Calculate L1 to L6 as described in the paper
        '''
        p1_dur_hat, p2_dur_hat, b_p1_hat, e_p1_hat, b_p2_hat, e_p2_hat = predictions
        # print("\n\nall predictions:")
        # print("p1_dur_hat:", p1_dur_hat)
        # print("p2_dur_hat:", p2_dur_hat)
        # print("b_p1_hat:", b_p1_hat)
        # print("e_p1_hat:", e_p1_hat)
        # print("b_p2_hat:", b_p2_hat)
        # print("e_p2_hat:", e_p2_hat)

        p1_durs, p2_durs, b_p1, e_p1, b_p2, e_p2 = ground_truth
        # print("p1_dur:", p1_durs)
        # print("p2_dur:", p2_durs)
        # print("b_p1:", b_p1)
        # print("e_p1:", e_p1)
        # print("b_p2:", b_p2)
        # print("e_p2:", e_p2)
        
        # Duration Losses
        
        L1_p1 = self.cross_entropy_loss(p1_dur_hat, p1_durs)
        L1_p2 = self.cross_entropy_loss(p2_dur_hat, p2_durs)
        
        dur_loss = ((L1_p1 + L1_p2) / 2)

        # Relation Losses

        L2 = self.l1_loss(e_p1_hat - b_p2_hat, e_p1 - b_p2)
        L3 = self.l1_loss(e_p2_hat - b_p1_hat, e_p2 - b_p1)
        L4 = self.l1_loss(b_p2_hat - b_p1_hat, b_p2 - b_p1)
        L5 = self.l1_loss(e_p2_hat - e_p1_hat, e_p2 - e_p1)
        # print("L2:", L2)
        # print("L3:", L3)
        # print("L4:", L4)
        # print("L5:", L5)

        rel_loss = sum([L2, L3, L4, L5])

        total_loss = sum([self.alpha_for_dur_loss * dur_loss, self.beta_for_rel_loss * rel_loss])
        
        return total_loss, dur_loss, rel_loss

def get_relation_metrics(p1_rel_beg_yhat, p1_rel_end_yhat, p2_rel_beg_yhat, p2_rel_end_yhat, 
                        p1_rel_beg, p1_rel_end, p2_rel_beg, p2_rel_end):
    """
        "Spearman Correlation": closer to +/-1: better; 0: no correlation
        "P-values": the probability of an uncorrelated system producing datasets that have a Spearman correlation
                    at least as extreme as the one computed from these datasets: 
                    This should mean: lower p-value means the two provided datasets are less correlated
    """
    pred_reals = np.concatenate((p1_rel_beg.cpu(), p1_rel_end.cpu(), p2_rel_beg.cpu(), p2_rel_end.cpu()))
    pred_reals_yhat = np.concatenate((p1_rel_beg_yhat.cpu(), p1_rel_end_yhat.cpu(), p2_rel_beg_yhat.cpu(), p2_rel_end_yhat.cpu()))
    
    p_corr = spearmanr(pred_reals.astype('float64'), pred_reals_yhat.astype('float64'))

    return {
        "Spearman Correlation": p_corr[0] if not np.isnan(p_corr[0]) else 0, # 0: no correlation
        "P-value": p_corr[1] if not np.isnan(p_corr[1]) else 0 # [0-1] lower p-value means the two provided datasets are less correlated
    }

def get_duration_metrics(p1_dur_yhat, p2_dur_yhat, p1_dur, p2_dur): # input: tensors of dim (dataset length, )
    """
        "Spearman Correlation": closer to +/-1: better; 0: no correlation
        "P-values": the probability of an uncorrelated system producing datasets that have a Spearman correlation
                    at least as extreme as the one computed from these datasets: 
                    This should mean: lower p-value means the two provided datasets are less correlated
        "Exact-match Accuracy": normal accuracy,
        "At-most-one-off Accuracy": coarser accuracy,
    """
    pred_durs = np.append(p1_dur.cpu(), p2_dur.cpu())
    print("pred_durs[:10]:", pred_durs[:10])
    pred_durs_yhat = np.append(p1_dur_yhat.cpu(), p2_dur_yhat.cpu())
    print("pred_durs_yhat[:10]:", pred_durs_yhat[:10])
    dataset_size = len(pred_durs)
    
    ## RuntimeWarning: invalid value encountered in true_divide i.e. divided by zero -> nan
    # console output:
    #   val batch ground truth: [2 6 8 0 4 3 7 0] prediction: [8 8 8 8 8 8 8 8]
    #   score: nan p: nan
    # fix: change nparray to .astype('float64')
    # STILL NOT FULLY WORKING... (p-value)
    # source: https://github.com/wasade/scikit-bio/commit/33c73520ab81fd7ad47c88187a7526d94604762b
    sp_corr = spearmanr(pred_durs.astype('float64'), pred_durs_yhat.astype('float64'))

    exact_agree_count = np.count_nonzero(pred_durs == pred_durs_yhat)
    exact_accu = exact_agree_count / dataset_size
    print("exact_match:", pred_durs[:10] == pred_durs_yhat[:10])
    dur_at_most_one_off = (pred_durs == pred_durs_yhat) \
                            + ((pred_durs + 1)== pred_durs_yhat) \
                            + ((pred_durs - 1)== pred_durs_yhat)

    at_most_one_off_count = np.count_nonzero(dur_at_most_one_off)
    at_most_one_off_accu = at_most_one_off_count / dataset_size

    return {
        "Spearman Correlation": sp_corr[0] if not np.isnan(sp_corr[0]) else 0, # 0: no correlation
        "P-value": sp_corr[1] if not np.isnan(sp_corr[1]) else 0, # [0-1] lower p-value means the two provided datasets are less correlated
        "Exact-match Accuracy": exact_accu,
        "At-most-one-off Accuracy": at_most_one_off_accu,
    }


if __name__ == "__main__":

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    train = True
    val = True
    test = False
    n_epochs = 1000
    b_z = 8
    lr = 2e-5
    dur_label_weights = torch.ones(11)

    writer = SummaryWriter("logs/no_duration_class_weight/")

    # dataloaders
    if train:
        train_dataset = UDS_T_Dataset(tsv_filename=None, 
                                processed_jsonl_filename='UDS_T_data/time_eng_ud_v1.2_2015_10_30_preprocessed.jsonl',
                                dataset_split='train',
                                get_pairs=True,
                                max_input_size=256)
        train_dataloader = DataLoader(train_dataset, batch_size=b_z, shuffle=True, drop_last=False)
        print("training dataset size:", len(train_dataset))
        # dur_label_weights = train_dataset.get_dur_label_loss_weight()
        
    if val:
        val_dataset = UDS_T_Dataset(tsv_filename=None, 
                                processed_jsonl_filename='UDS_T_data/time_eng_ud_v1.2_2015_10_30_preprocessed.jsonl',
                                dataset_split='dev',
                                get_pairs=True,
                                max_input_size=256)
        val_dataloader = DataLoader(val_dataset, batch_size=b_z, shuffle=True, drop_last=True)
        print("validation dataset size:", len(val_dataset))
    
    model = Multitask_Baseline(freeze_bert=False,
                                num_duration_classes=11,
                                num_relation_output=4,
                                span_based=True,
                                root_based=False,
                                alpha_for_dur_loss=3,
                                beta_for_rel_loss=0.5,
                                dur_loss_weight = dur_label_weights
                            )
    model.to(device)

    # get Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # training
    if train:
        
        num_steps = len(train_dataloader) * n_epochs
        current_step = 0
        batch_size = train_dataloader.batch_size
        best_val_durs_accu_fine = 0
        best_val_durs_accu_coarse = 0
        best_val_rels_coor = 0

        for epoch in range(1, n_epochs + 1):
            print("training epoch: {}...".format(epoch))
            running_total_loss = 0.0
            running_dur_loss = 0.0
            running_rel_loss = 0.0

            for i, batch in enumerate(train_dataloader):
                X = {key: value.to(device) for key, value in batch[0][0].items()} # not sure why X needs double [0]..
                Y = {key: value.to(device) for key, value in batch[1].items()}
                
                ground_truth = (
                    Y['Event_1_dur'],
                    Y['Event_2_dur'],
                    Y['Event_1_begin'],
                    Y['Event_1_end'],
                    Y['Event_2_begin'],
                    Y['Event_2_end']
                )
                optimizer.zero_grad()
                dur1_final, dur2_final, rel_final = model(
                                            tokens = X['Token_ids'],
                                            span1_masks = X['Span1_only_masks'],
                                            span2_masks = X['Span2_only_masks'],
                                            root1_masks = X['Root1_only_masks'],
                                            root2_masks = X['Root2_only_masks']
                                        )
                # print(dur1_final.size())
                # print(Y['Event_1_dur'].size())
                # print(rel_final[:, 0].size())
                # print(Y['Event_2_end'].size())

                prediction = (
                    dur1_final,
                    dur2_final,
                    rel_final[:,0],
                    rel_final[:,1],
                    rel_final[:,2],
                    rel_final[:,3],
                )

                total_loss, dur_loss, rel_loss = model.get_losses(prediction, ground_truth)
                running_total_loss += total_loss.item() / batch_size
                running_dur_loss += dur_loss.item() / batch_size
                running_rel_loss += rel_loss.item() / batch_size
                
                total_loss.backward()
                optimizer.step()
                current_step += 1


                if i % 100 == 99:
                # if i % 100 == 1: # debugging
                    print('[Epoch %2d, Step %5d] --- training total_loss: %.5f, duration_loss: %.5f, relation_loss: %.5f' %
                        (epoch, i + 1, running_total_loss / 100, running_dur_loss / 100, running_rel_loss / 100))
                    
                    writer.add_scalar('Train/Total_Loss', running_total_loss / 100, current_step)
                    writer.add_scalar('Train/Duration_CrossEntropyLoss', running_dur_loss / 100, current_step)
                    writer.add_scalar('Train/Relation_L1loss', running_rel_loss / 100, current_step)

                    running_total_loss = 0.0
                    running_dur_loss = 0.0
                    running_rel_loss = 0.0

                if eval and current_step % 1000 == 999:
                # if eval and current_step % 200 == 1: # debugging
                    model.eval()
                    with torch.no_grad():
                        
                        val_total_loss = 0.0
                        val_dur_loss = 0.0
                        val_rel_loss = 0.0
                        
                        size = len(val_dataset)
                        val_dur1_pred = torch.zeros(size)
                        val_dur1_true = torch.zeros(size)
                        
                        val_dur2_pred = torch.zeros(size)
                        val_dur2_true = torch.zeros(size)

                        val_rel1_beg_pred = torch.zeros(size)
                        val_rel1_beg_true = torch.zeros(size)

                        val_rel1_end_pred = torch.zeros(size)
                        val_rel1_end_true = torch.zeros(size)

                        val_rel2_beg_pred = torch.zeros(size)
                        val_rel2_beg_true = torch.zeros(size)

                        val_rel2_end_pred = torch.zeros(size)
                        val_rel2_end_true = torch.zeros(size)
                        
                        for j, batch in enumerate(val_dataloader):
                            X = {key: value.to(device) for key, value in batch[0][0].items()} # not sure why X needs double [0]..
                            Y = {key: value.to(device) for key, value in batch[1].items()}

                            dur1_final, dur2_final, rel_final = model(
                                                        tokens = X['Token_ids'],
                                                        span1_masks = X['Span1_only_masks'],
                                                        span2_masks = X['Span2_only_masks'],
                                                        root1_masks = X['Root1_only_masks'],
                                                        root2_masks = X['Root2_only_masks']
                                                    )

                            prediction = (
                                dur1_final,
                                dur2_final,
                                rel_final[:,0],
                                rel_final[:,1],
                                rel_final[:,2],
                                rel_final[:,3],
                            )
                            ground_truth = (
                                Y['Event_1_dur'],
                                Y['Event_2_dur'],
                                Y['Event_1_begin'],
                                Y['Event_1_end'],
                                Y['Event_2_begin'],
                                Y['Event_2_end']
                            )

                            curr_total_loss, curr_dur_loss, curr_rel_loss = model.get_losses(prediction, ground_truth)
                            val_total_loss += curr_total_loss
                            val_dur_loss   += curr_dur_loss
                            val_rel_loss   += curr_rel_loss
                            
                            val_dur1_true[j * batch_size : (j+1) * batch_size]     = Y['Event_1_dur']
                            val_dur2_true[j * batch_size : (j+1) * batch_size]     = Y['Event_2_dur']
                            val_rel1_beg_true[j * batch_size : (j+1) * batch_size] = Y['Event_1_begin']
                            val_rel1_end_true[j * batch_size : (j+1) * batch_size] = Y['Event_1_end']
                            val_rel2_beg_true[j * batch_size : (j+1) * batch_size] = Y['Event_2_begin']
                            val_rel2_end_true[j * batch_size : (j+1) * batch_size] = Y['Event_2_end']

                            val_dur1_pred[j * batch_size : (j+1) * batch_size]     = torch.argmax(dur1_final, dim=1)
                            val_dur2_pred[j * batch_size : (j+1) * batch_size]     = torch.argmax(dur2_final, dim=1)
                            val_rel1_beg_pred[j * batch_size : (j+1) * batch_size] = rel_final[:,0]
                            val_rel1_end_pred[j * batch_size : (j+1) * batch_size] = rel_final[:,1]
                            val_rel2_beg_pred[j * batch_size : (j+1) * batch_size] = rel_final[:,2]
                            val_rel2_end_pred[j * batch_size : (j+1) * batch_size] = rel_final[:,3]

                        dur_metrics_dict = get_duration_metrics(val_dur1_pred, val_dur2_pred, val_dur2_true, val_dur2_true)
                        rel_metrics_dict = get_relation_metrics(
                            val_rel1_beg_pred, val_rel1_end_pred, val_rel2_beg_pred, val_rel2_end_pred, 
                            val_rel1_beg_true, val_rel1_end_true, val_rel2_beg_true, val_rel2_end_true
                        )

                        val_dur_sp_corr     = dur_metrics_dict["Spearman Correlation"]
                        val_dur_p_value     = dur_metrics_dict["P-value"]
                        val_dur_accu_fine   = dur_metrics_dict["Exact-match Accuracy"]
                        val_dur_accu_coarse = dur_metrics_dict["At-most-one-off Accuracy"]
                        
                        writer.add_scalar('Dev/Spearman Correlation of Duration', val_dur_sp_corr, current_step)
                        writer.add_scalar('Dev/P-values of Duration', val_dur_p_value, current_step)
                        writer.add_scalar('Dev/Exact-match Accuracy of Duration', val_dur_accu_fine, current_step)
                        writer.add_scalar('Dev/At-most-one-off Accuracy of Duration', val_dur_accu_coarse, current_step)

                        val_rel_sp_corr  = rel_metrics_dict["Spearman Correlation"]
                        val_rel_p_value  = rel_metrics_dict["P-value"]
                        
                        writer.add_scalar('Dev/Spearman Correlation of Relation', val_rel_sp_corr, current_step)
                        writer.add_scalar('Dev/P-values of Relation', val_rel_p_value, current_step)
                        
                        writer.add_scalar('Dev/Total Loss', val_total_loss / len(val_dataset), current_step)
                        writer.add_scalar('Dev/Duration Loss', val_dur_loss / len(val_dataset), current_step)
                        writer.add_scalar('Dev/Relation Loss', val_rel_loss / len(val_dataset), current_step)
                        

                        print('[Epoch %2d, Step %5d] --- validation total_loss: %.5f, duration_loss: %.5f, relation_loss: %.5f' %
                            (epoch, i + 1, val_total_loss / len(val_dataset), val_dur_loss / len(val_dataset), val_rel_loss / len(val_dataset)))
                        print('                       --- duration correlation: %.5f, relation correlation: %.5f' % (val_dur_sp_corr, val_rel_sp_corr))
                        print('                       --- duration exact accuracy: %.5f, duration <= one off accu: %.5f' % (val_dur_accu_fine, val_dur_accu_coarse))
                        
                        if val_dur_accu_fine > best_val_durs_accu_fine:
                            # save as best duration
                            torch.save(model.state_dict(), "model_ckpt/no_duration_class_weight/best_fine_duration_epoch{}_step{}_bert_base_accuracy_{}.ckpt".format(epoch, i, val_dur_accu_fine))
                            best_val_durs_accu_fine = val_dur_accu_fine
                        
                        if val_dur_accu_coarse > best_val_durs_accu_coarse:
                            # save as best duration
                            torch.save(model.state_dict(), "model_ckpt/no_duration_class_weight/best_coarse_duration_epoch{}_step{}_bert_base_accuracy_{}.ckpt".format(epoch, i, val_dur_accu_coarse))
                            best_val_durs_accu_coarse = val_dur_accu_coarse
                        
                    model.train()