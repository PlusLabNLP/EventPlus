import torch
from torch.nn import Module, Linear, ModuleList, Dropout, L1Loss, BCELoss, KLDivLoss
import numpy as np
import pandas as pd
import pickle
import argparse
import random
import os
import sys
from sklearn.metrics import mean_absolute_error as mae
from itertools import product
from ast import literal_eval
from factslab.utility.utility import print_metrics as print_metrics
from factslab.utility.elmo import get_elmo_rt
from factslab.utility.bert import get_bert
from factslab.utility.hand_engineered_features import get_type_feats
from factslab.utility.hand_engineered_features import get_token_feats
from tqdm import tqdm
import copy


class simpleMLP(Module):
    def __init__(self, activation, input_size, layers, device, dropout, \
                 embedders, prot, token_col):

        super(simpleMLP, self).__init__()
        self.device = device
        self.prot = prot
        self.token_col = token_col
        self._linmaps = ModuleList([])

        last_size = input_size
        output_size = 3
        for j in layers:
            self._linmaps.append(Linear(last_size, j))
            last_size = j
        self._linmaps.append(Linear(last_size, output_size))

        self._dropout = Dropout(p=dropout)
        self._activation = activation

        self.embedders = embedders


    def _embed_inputs(self, input_df):
        '''
            Initialize the different embedding modules.

            Arguments:
            ---------
            input_df: A minibatched pandas dataframe

            Returns:
            --------
            embedded_inputs: A pyTorch tensor containing embeddings for root
            tokens in each sentence of the minibatch. It will have dimensions
            (batch_size x embedded_dimensions) where embedded_dimensions
            depends on the feature ablation chosen for training
        '''

        sentences = [sen.split() for sen in input_df['Sentence'].tolist()]
        tokens = torch.tensor(input_df[self.token_col].values,dtype=torch.long,\
                              device=self.device).unsqueeze(1) - 1

        embedded_inputs = None

        if self.embedders['token'] != {}:
            token_embeds = pd.DataFrame([get_token_feats(row=row,
                                    token_feats=self.embedders['token'].copy(),
                                    token_col=self.token_col)
                                    for ix, row in input_df.iterrows()])
            token_embeds = torch.tensor(token_embeds.values, dtype=torch.float,
                                        device=self.device)
            embedded_inputs = token_embeds

        if self.embedders['type']['embedder'] != {}:
            type_embeds = pd.DataFrame([get_type_feats(row=row,
                          type_feats=self.embedders['type']['embedder'].copy(),
                          concreteness=self.embedders['type']['conc'],
                          lcs=self.embedders['type']['lcs'],
                          lem2frame=self.embedders['type']['lem2frame'],
                          prot=self.prot, token_col=self.token_col) \
                          for ix, row in input_df.iterrows()])
            type_embeds = torch.tensor(type_embeds.values, dtype=torch.float,
                                       device=self.device)
            if embedded_inputs is not None:
                embedded_inputs = torch.cat((embedded_inputs, type_embeds),
                                            dim=1)
            else:
                embedded_inputs = type_embeds

        if self.embedders['elmo']:
            elmo_embeds = get_elmo_rt(sentences=sentences, tokens=tokens,
                                      elmo_embedder=self.embedders['elmo'])
            if embedded_inputs is not None:
                embedded_inputs = torch.cat((embedded_inputs, elmo_embeds),
                                            dim=1)
            else:
                embedded_inputs = elmo_embeds

        if self.embedders['glove'] is not None:
            root_words = [sentence[token].lower() for sentence, token in \
                          zip(sentences, tokens)]
            glove_embeds = \
            np.array([self.embedders['glove'].loc[word] if word in \
                      self.embedders['glove'].index else \
                      self.embedders['glove'].loc['_UNK'] for word in \
                      root_words])
            glove_embeds = torch.tensor(glove_embeds, dtype=torch.float, \
                                        device=self.device)
            if embedded_inputs is not None:
                embedded_inputs = torch.cat((embedded_inputs, glove_embeds),
                                            dim=1)
            else:
                embedded_inputs = glove_embeds

        if self.embedders['bert'][0] is not None:
            bert_embeds = get_bert(sentences=sentences, tokens=tokens,
                                      bert_embedder=self.embedders['bert'][0],
                                      bert_tokenizer=self.embedders['bert'][1],
                                      device=self.device)
            if embedded_inputs is not None:
                embedded_inputs = torch.cat((embedded_inputs, bert_embeds),
                                            dim=1)
            else:
                embedded_inputs = bert_embeds

        return embedded_inputs


    def nonlinearity(self, x):
        '''Applies relu or tanh activation on tensor.'''

        if self._activation == 'relu':
            return torch.nn.functional.relu(x)
        elif self._activation == 'tanh':
            return torch.tanh(x)

    def forward(self, inputs, return_hidden=False):
        '''
            Runs forward pass on neural network.

            Arguments:
            ----------
            inputs: A minibatched pandas dataframe
            return_hidden: if true return a list of hidden state activations

            Returns:
            ------
            x: Final layer activation
            hidden: Hidden layer activations(if return_hidden is True)
        '''
        hidden = []

        # Get embedded inputs from dataframe
        x = self._embed_inputs(inputs)

        # Run through linear layers
        for i, linmap in enumerate(self._linmaps):
            if i:
                x = self.nonlinearity(x)
                x = self._dropout(x)
            x = linmap(x)
            if return_hidden:
                hidden.append(x.detach().cpu().numpy())

        if return_hidden:
            return x, hidden
        else:
            return x


class simpleMLPTrainer:
    '''Trainer class to perform regression using simple MLP'''

    def __init__(self, device,attributes_norm, batch_size, epochs, alpha,\
                 optimizer_class=torch.optim.Adam,lr=0.001, **kwargs):

        self._optimizer_class = optimizer_class
        self._init_kwargs = kwargs
        self.device = device
        self.lr = lr
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs=epochs
        self.attributes = attributes_norm
        self._initialize_trainer_regression()

    def _initialize_trainer_regression(self):
        '''Initialize regression module and loss function'''

        self._regression = \
            simpleMLP(device=self.device, **self._init_kwargs).to(self.device)
        self._loss_function = L1Loss().to(self.device)


    def fit(self, data, data_dev, verbosity):
        '''
            Perform training with early stopping.

            Arguments:
            ---------
            data: Training data - a pandas DataFrame.
            data_dev: Validation data (used for early stopping) - a pandas
                      DataFrame.
            verbosity: Parameter to control whether tqdm progress bar is
                       displayed. It is False when performing a hyperparameter
                       search.

            Returns:
            --------
            None
        '''

        self._regression = self._regression.train()

        parameters = [p for p in self._regression.parameters() if \
                                                        p.requires_grad]
        optimizer = self._optimizer_class(parameters, weight_decay=self.alpha)

        self.early_stopping = [1000]

        for epoch in range(self.epochs):
            for mb_idx in tqdm(range(0, data.shape[0], self.batch_size),
                               disable=verbosity):
                optimizer.zero_grad()
                minibatch = data[mb_idx: mb_idx + self.batch_size]

                y = np.array([minibatch[k].values for k in self.attributes])
                # Take transpose to make it 128x3 rather than 3x128
                y = torch.tensor(y, dtype=torch.float, device=self.device).t()

                # Forward pass
                y_pred = self._regression(inputs=minibatch)

                # Calculate loss
                loss = self._loss_function(y_pred, y)
                loss.backward()
                optimizer.step()

            # Early stopping
            df_pred_dev = self.predict(data_dev)
            y_pred_dev = df_pred_dev.loc[:, [(a + ".Pred") for a in \
                                             self.attributes]].values
            y_dev = df_pred_dev.loc[:, self.attributes].values
            score = mae(y_dev, y_pred_dev)
            if score - self.early_stopping[-1] > 0:
                self._regression.load_state_dict(
                                               torch.load('tmp/checkpoint.pt'))
                break
            else:
                torch.save(self._regression.state_dict(), 'tmp/checkpoint.pt')
                self.early_stopping.append(score)


    def predict(self, input_df, batch_size=256):
        '''
            Run predictions on an input and return predictions

            Arguments:
            ----------
            input_df: Data to perform predictions on - a pandas DataFrame.
            batch_size: Batch size to split input_df into for prediction.

            Returns:
            --------
            output_df: A copy of input_df with extra columns for each attribute
                       being predicted.
        '''
        output_df = input_df.copy()
        self._regression = self._regression.eval()

        for mb_idx in range(0, input_df.shape[0], batch_size):
            minibatch = input_df[mb_idx: mb_idx + batch_size]

            y_pred = self._regression(inputs=minibatch)

            for i, attr in enumerate(self.attributes):
                output_df.loc[mb_idx:mb_idx + batch_size - 1, attr+".Pred"] = \
                        y_pred[:, i].detach().cpu().numpy()

        return output_df


    def write_preds_to_file(self, data, output_path, prot):
        '''
            Write predictions to file in a nice tsv format for analysis

            Arguments:
            ----------
            data: Data to write to file - a pandas DataFrame.
            output_path: Path on filesystem to write to - str.
            prot: Protocol being run (arg or pred) - str.

            Returns:
            --------
            None
        '''
        if prot == "arg":
            columns = ['Unique.ID', 'Sentence', 'Arg.Token', 'Arg.Span', 'POS',
                       'DEPREL', 'Is.Particular.Norm', 'Is.Particular.Pred',
                       'Is.Kind.Norm', 'Is.Kind.Pred', 'Is.Abstract.Norm',
                       'Is.Abstract.Pred']
            data['POS'] = data.apply(lambda x: \
                        x['PredPatt'].tokens[x['Arg.Token'] - 1].tag, axis=1)
            data['DEPREL'] = data.apply(lambda x: \
                       x['PredPatt'].tokens[x['Arg.Token'] - 1].gov_rel, axis=1)
        else:
            columns = ['Unique.ID', 'Sentence', 'Pred.Token', 'Pred.Span',\
                       'POS', 'DEPREL', 'Is.Particular.Norm',\
                       'Is.Particular.Pred', 'Is.Hypothetical.Norm',\
                       'Is.Hypothetical.Pred', 'Is.Dynamic.Norm',\
                       'Is.Dynamic.Pred']
            data['POS'] = data.apply(lambda x: x['PredPatt'].tokens[x['Pred.Token'] - 1].tag, axis=1)
            data['DEPREL'] = data.apply(lambda x: x['PredPatt'].tokens[x['Pred.Token'] - 1].gov_rel, axis=1)

        data = data.loc[:, columns]
        data.to_csv(output_path, sep='\t', index=False)

    def save_model(self, path):
        '''
            Saves pyTorch state dict model of _regression
        '''
        torch.save(self._regression.state_dict(), path)

    def load_model(self, path):
        '''
            Loads pyTorch state dict model into _regression
        '''
        self._regression.load_state_dict(torch.load(path))

    def do_riemann(h, y):
        '''
        Perform Riemann analysis using UMAP package and save figure.

        Arguments:
        ----------
        h: hidden state values - np.array
        y: final predictions - np.array

        Returns:
        --------
        None
        '''
        import umap
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        reducer = umap.UMAP(random_state=1)
        reducer.fit(h)
        embedding = reducer.transform(h)
        # target_dict = {'[0 0 0]': 0, '[1 0 0]': 1, '[0 1 0]': 2, '[0 0 1]': 3, '[1 1 0]': 4, '[0 1 1]': 5, '[1 0 1]': 6, '[1 1 1]': 7}
        # targets = np.array([i for i in range(0, 8)])
        plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='Spectral')
        # plt.gca().set_aspect('equal', 'datalim')
        # plt.legend()
        # plt.colorbar(boundaries=np.arange(8) - 0.5).set_ticks(np.arange(8))
        plt.savefig('umap.png')
