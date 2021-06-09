import torch
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss, Linear, Dropout
from transformers import AutoModel


class TransformerRegressionModel(torch.nn.Module):

    def __init__(self, transformer, dropout, num_labels,cls_token=0,
                 activation='relu'):
        '''
        Setup the modules in the model - a transformer, followed by a GRU for
        the CLS hidden states/taking the mean of all tokens, followed by Linear
        layers that outputs one number, followed by softmax
        '''
        super(TransformerRegressionModel, self).__init__()

        # Setup the transformer model
        self.transformer = AutoModel.from_pretrained(transformer)


        # For now CLS pooling is the only pooling supported
        self.tr_output_size = self.transformer.config.hidden_size
        self.num_labels = num_labels
        self.cls_token = cls_token

        # Setup the linear layers on top of the transformer
        self.dense = Linear(self.tr_output_size, self.tr_output_size)
        self.dropout = Dropout(p=dropout)
        self.classifier = Linear(self.tr_output_size, self.num_labels)
        self._activation = activation

        self._loss_fn = MSELoss()


    def _nonlinearity(self, x):
        '''Applies relu or tanh activation on tensor.'''

        if self._activation == 'relu':
            return torch.nn.functional.relu(x)
        elif self._activation == 'tanh':
            return torch.tanh(x)


    def forward(self, input_ids, input_mask, tokens, labels=None):
        '''
        Runs forward pass on neural network

        Arguments:
        ---------
        input_ids: the tokenized, bert wordpiece IDs. (batch_size, MAX_LEN)
        input_masks: the masking to be done on input_ids due to padding.
        (batch_size, MAX_LEN)
        labels: target against which to computer the loss. DEFAULT: None
        max_seq_len: The length to which to pad the output of the rnn

        Returns:
        -------

        Object of type Tuple of form (loss, logits)

        loss: Cross Entropy loss calculated in loss_fn which implements masking
        logits: logsoftmaxed probabilities of classifier output

        '''


        # Forward pass through transformer
        # other values returned are pooler_output, hidden_states, and attentions
        outputs = self.transformer(input_ids,
                                   token_type_ids=None,
                                   attention_mask=input_mask)

        last_hidden_states = outputs[0]

        # Get the hidden states based on token indices
        token_hidden_states = torch.cat([h.index_select(0,tok) for \
                                    h, tok in zip(last_hidden_states, tokens)])


        # Then run it through linear layers
        x = self.dropout(token_hidden_states)
        x = self.dense(x)
        x = self._nonlinearity(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        outputs  = (logits,)

        if labels is not None:
            loss = self._loss_fn(logits, labels)
            outputs = (loss,) + outputs

        return outputs