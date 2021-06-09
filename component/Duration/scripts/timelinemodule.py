import allennlp
import torch
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import pickle
from torch.distributions.binomial import Binomial
from torch.nn import MSELoss, L1Loss, SmoothL1Loss, CrossEntropyLoss

import torch
from torch import nn
#from torchviz import make_dot, make_dot_from_trace
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm_n


class TimelineModel(torch.nn.Module):
    '''
     A class to extract a simple timeline model from a
     given document's predicate-pair data
    '''
    def __init__(self,
                 data = None,
                 num_preds = None, 
                 mlp_activation='relu',
                 mlp_dropout=0.0,
                 optimizer_class = torch.optim.Adam,
                  dur_output_size = 11, fine_output_size = 4,
                device=torch.device(type="cpu"),
                **kwargs):
        super().__init__()

        self.device = device
        self.linear_maps = nn.ModuleDict()
        self.mlp_activation = mlp_activation
        self.mlp_dropout =  nn.Dropout(mlp_dropout) 
        self.dur_output_size = dur_output_size
        
        ## Parameters
            # Hidden predicate representations
        self.pred_tensor = torch.nn.Parameter(torch.randn(num_preds,2).to(self.device), requires_grad=True)
            # Binomial parameter
        self.k = torch.nn.Parameter(torch.randn(1).to(self.device), requires_grad=True)
        
        self.params = nn.ParameterList()
        self.params.extend([self.pred_tensor, self.k])
        
        self._optimizer_class = optimizer_class
        
        ## Losses Initialization
        self.fine_loss = L1Loss().to(self.device)
        self.duration_loss = CrossEntropyLoss().to(self.device)

        
    def _init_MLP(self, input_size, hidden_sizes, output_size, param=None):
        '''
        Initialise MLP or regression parameters
        '''
        self.linear_maps[param] = nn.ModuleList()

        for h in hidden_sizes:
            linmap = torch.nn.Linear(input_size, h)
            linmap = linmap.to(self.device)
            self.linear_maps[param].append(linmap)
            input_size = h

        linmap = torch.nn.Linear(input_size, output_size)
        linmap = linmap.to(self.device)
        self.linear_maps[param].append(linmap)
        
    def forward(self, local_data, **kwargs):
        '''
        INput: dataframe with cols:
                b1, e1, b2, e2, pred1_dict_idx, pred2_dict_idx
                
        Output: 
        '''
        t_sq = self.pred_tensor**2 
        num_preds= t_sq.size()[0]
        anchored_tensor = torch.zeros(num_preds,2).to(self.device)
        
        anchored_tensor[:,0] = t_sq[:,0] - t_sq[:,0].min()
        anchored_tensor[:,1] = t_sq[:,1]
        
        #Predicted fine-grained values for the given document
        b1 = anchored_tensor[local_data.pred1_dict_idx.values][:,0]
        dur1 = anchored_tensor[local_data.pred1_dict_idx.values][:,1]
        b2 = anchored_tensor[local_data.pred2_dict_idx.values][:,0]
        dur2 = anchored_tensor[local_data.pred2_dict_idx.values][:,1]
        
        batch_size = b1.size()[0]
        #print(batch_size)
                
        pred1_dur = self._binomial_dist(dur1)
        pred2_dur = self._binomial_dist(dur2)
        
        yhat = (b1, dur1, b2, dur2, pred1_dur, pred2_dur,
                anchored_tensor)
        
        return yhat
    
    def fit(self, local_data, epochs=5000, **kwargs):
        losses = [10000]
        
        # print("#### Model Parameters ####")
        # for name,param in self.named_parameters():     
        #     if param.requires_grad:
        #         print(name, param.shape) 
        # print("##########################") 
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = self._optimizer_class(parameters)
        
        #Actual ground truth values
        b1_lst = local_data.b1.values
        e1_lst = local_data.e1.values
        b2_lst = local_data.b2.values
        e2_lst = local_data.e2.values
        durations = [local_data.pred1_duration.values,
                     local_data.pred2_duration.values]

        
        # pbar = tqdm(total = total_obs//self.train_batch_size)
        
        for epoch in tqdm(range(epochs)):
            preds = self(local_data)
            #zero_grad
            optimizer.zero_grad()
            curr_loss = self._custom_loss(preds,
                                         b1_lst,
                                         e1_lst,
                                         b2_lst,
                                         e2_lst,
                                         durations)
            
            curr_loss.backward()
            optimizer.step()
            
            if epoch==0:
                tqdm.write("Epoch: {}, Loss: {}".format(epoch+1, curr_loss))
            
            #print("Epoch: {}, Loss: {}".format(epoch+1, curr_loss))
               
            ## Stop training when loss converges
            if abs(curr_loss.detach() - losses[-1]) < 0.00001:
                #print("Epoch: {}, Converging-Loss: {}".format(epoch+1, curr_loss))
                break
                
            #pbar.update(1)
                
            losses.append(curr_loss.detach())
        #pbar.close()
        tqdm.write("Epoch: {}, Converging-Loss: {}".format(epoch+1, curr_loss))
                
        return self.predict(preds)
        
    def _custom_loss(self, preds, b1_lst, e1_lst, b2_lst,
                            e2_lst,durations):
        ## Predictions
        b1_pred, dur1_pred, b2_pred, dur2_pred = preds[0], preds[1], preds[2], preds[3]
        out_p1_d, out_p2_d, anchored_tensor = preds[4], preds[5], preds[6]
#         out_coarse, out_coarser = preds[7], preds[8]
        
        ## Ground truth values:
        b1_act, e1_act, b2_act, e2_act = self._lsts_to_tensors(b1_lst, e1_lst, b2_lst, e2_lst,
                                        param="float")
        ## Store actual_y into tensors
        pred1_durs, pred2_durs = durations

        pred1_durs, pred2_durs = self._lsts_to_tensors(pred1_durs,pred2_durs)
        
        ## Duration Losses
        L5_p1 = self.duration_loss(out_p1_d, pred1_durs)
        L5_p2 = self.duration_loss(out_p2_d, pred2_durs)
        #print("L5_p1 {},  L5_p2: {}".format(L5_p1, L5_p2))
            
        ## Normalize predicted fine-grained values:
        num_pairs = b1_pred.size()[0]
        t = torch.zeros(num_pairs,4).to(self.device)
        t[:,0] = b1_pred
        t[:,1] = b1_pred + dur1_pred
        t[:,2] = b2_pred
        t[:,3] = b2_pred + dur2_pred
        
    
        t_min, _ = torch.min(t,dim=1)
        t_min = t_min.unsqueeze(1).repeat(1,4)  #add extra dimension
        t_adj = t - t_min
        t_adj_max, _ = torch.max(t_adj,dim=1)
        t_adj_max = t_adj_max.unsqueeze(1).repeat(1,4)
        t_normalized = t_adj/t_adj_max
        
        ## Fine-grained Losses
        l1 = self.fine_loss(t_normalized[:,0]-t_normalized[:,2], b1_act-b2_act)
        l2 = self.fine_loss(t_normalized[:,1]-t_normalized[:,2], e1_act-b2_act)
        l3 = self.fine_loss(t_normalized[:,3]-t_normalized[:,0], e2_act-b1_act)
        l4 = self.fine_loss(t_normalized[:,1]-t_normalized[:,3], e1_act-e2_act)
        
        L1to4 = sum([l1, l2, l3, l4])/4 
           
        #L5_p1, L5_p2 = 0,0 
        
        #print("L1to4: {}".format(L1to4))
        
        dur = (L5_p1+L5_p2)/2
        fine = L1to4
        beta=2.0
        
        total_loss = (sum([dur, beta*fine])/2)
        
        return total_loss
            
    def _lsts_to_tensors(self, *args, param=None):
        '''
        Input: list1, list2,......

        Output: [Tensor(list1), tensor(list2),....]

        '''
        if param=="float":
            return [torch.from_numpy(np.array(arg)).float().to(self.device) for arg in args]
        else:
            return [torch.from_numpy(np.array(arg, dtype="int64")).to(self.device) for arg in args]
        
    def predict(self, preds):
        b1_pred, dur1_pred, b2_pred, dur2_pred = preds[0], preds[1], preds[2], preds[3]
        pred_timeline =  preds[6]
        
        ## Normalize predicted values:
        num_pairs = b1_pred.size()[0]
        t = torch.zeros(num_pairs,4).to(self.device)
        t[:,0] = b1_pred
        t[:,1] = b1_pred + dur1_pred
        t[:,2] = b2_pred
        t[:,3] = b2_pred + dur2_pred
        
        t_min, _ = torch.min(t,dim=1)
        t_min = t_min.unsqueeze(1).repeat(1,4)  #add extra dimension
        t_adj = t - t_min
        t_adj_max, _ = torch.max(t_adj,dim=1)
        t_adj_max = t_adj_max.unsqueeze(1).repeat(1,4)
        t_normalized = t_adj/t_adj_max
        t_normalized = t_normalized.detach().cpu().numpy()
        
        return t_normalized[:,0],t_normalized[:,1], t_normalized[:,2], t_normalized[:,3], pred_timeline.detach().cpu().numpy()
    
    def _binomial_dist(self, pred_dur):
        '''
        *** Vectorized implementation ***
        Input: A tensor with dimension: batch_size x 1
        Output: A tensor with dimension: batch_size x 11 
        Binomial Prob distribution for a given duration value 
        '''
        pred_dur = torch.sigmoid((self.k)*(torch.log(pred_dur)))
    
        bin_class = Binomial(total_count=self.dur_output_size-1, probs=pred_dur)
        durations = torch.tensor(range(self.dur_output_size), dtype=torch.float).to(self.device)
        
        return self._log_prob_vectorized(bin_class, durations)
        
    def _log_prob_vectorized(self, bin_class, value):
        '''
        1. bin_class: Pytorch Binomial distribution class 
        2. Value is a tensor with size: [total_count+1]
        '''
        batch_size = bin_class.total_count.size()[0]

        value = value.repeat(batch_size,1)
        #print(value.size())

        bin_class.logits = bin_class.logits.repeat(11,1).permute(1,0)
        #print(bin_class.logits.size())

        bin_class.total_count = bin_class.total_count.repeat(11,1).permute(1,0)
        #print(bin_class.total_count.size())

        log_factorial_n = torch.lgamma(bin_class.total_count + 1)
        log_factorial_k = torch.lgamma(value + 1)
        log_factorial_nmk = torch.lgamma(bin_class.total_count - value + 1)
        max_val = (-bin_class.logits).clamp(min=0.0)
        # Note that: torch.log1p(-bin_class.probs)) = max_val - torch.log1p((bin_class.logits + 2 * max_val).exp()))

        return (log_factorial_n - log_factorial_k - log_factorial_nmk +
                value * bin_class.logits + bin_class.total_count * max_val -
                bin_class.total_count * torch.log1p((bin_class.logits + 2 * max_val).exp()))