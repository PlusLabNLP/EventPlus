import sys
import torch
import torch.nn.functional as F
import pdb
from abc import ABCMeta, abstractmethod
from factslab.datastructures import ConstituencyTree
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.rnn import RNNBase

if sys.version_info.major == 3:
    from functools import lru_cache
else:
    from functools32 import lru_cache


class ChildSumTreeLSTM(RNNBase):
    """A bidirectional extension of child-sum tree LSTMs

    This class cannot be instantiated directly. Instead, use one of
    its subclasses:
      - ChildSumDependencyTreeLSTM
      - ChildSumConstituencyTreeLSTM
    """

    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        super(ChildSumTreeLSTM, self).__init__('LSTM', *args, **kwargs)

        # lru_cache is normally used as a decorator, but that usage
        # leads to a global cache, where we need an instance specific
        # cache
        self._get_parameters = lru_cache()(self._get_parameters)

    @staticmethod
    def nonlinearity(x):
        return F.tanh(x)

    def forward(self, inputs, tree):
        """
        Parameters
        ----------
        inputs : torch.Tensor
            a 2D (steps x embedding dimension) or a 3D tensor (steps x
            batch dimension x embedding dimension); the batch
            dimension must always have size == 1, since this module
            does not support minibatching
        tree : nltk.DependencyGraph
            must implement the following instance methods
            - root_idx: all root indices in the tree
            - children_idx: indices of children of a particular index
            - parents_idx: indices of parents of a particular index
            - is_terminal_idx: whether the node is terminal

        Returns
        -------
        hidden_all : torch.Tensor
        hidden_final : torch.Tensor
            the hidden state of the trees root node; if there are two
            or more such nodes, the average of their hidden states is
            returned
        """

        self._validate_inputs(inputs)

        ridx = tree.root_idx()

        self.hidden_state = {}
        self.cell_state = {}

        for layer in range(self.num_layers):
            self.hidden_state[layer] = {'up': {}, 'down': {}}
            self.cell_state[layer] = {'up': {}, 'down': {}}

            for i in ridx:
                self._upward_downward(layer, 'up', inputs, tree, i)

        # sort the indices; only really matters for constituency trees,
        # since dependency trees can be linearized in the same order
        # as the original inputs, while constituency trees will have
        # more hidden states than there are inputs
        indices = tree.positions

        hidden_up = self.hidden_state[self.num_layers - 1]['up']

        if self.bidirectional:
            hidden_down = self.hidden_state[self.num_layers - 1]['down']
            hidden_all = [torch.cat([hidden_up[i], hidden_down[i]])
                          for i in indices]
        else:
            hidden_all = [hidden_up[i] for i in indices]

        hidden_final = [hidden_all[k]
                        for k, i in enumerate(indices)
                        if i in ridx]

        hidden_all = torch.stack(hidden_all)
        hidden_final = torch.mean(torch.stack(hidden_final), 0, keepdim=False)

        if self._has_batch_dimension:
            if self.batch_first:
                return hidden_all[None, :, :], hidden_final[None, :]
            else:
                return hidden_all[:, None, :], hidden_final[None, :]
        else:
            return hidden_all, hidden_final

    def _upward_downward(self, layer, direction, inputs, tree, idx):
        # check to see whether this node has been computed on this
        # layer in this direction, if so short circuit the rest of
        # this function and return that result
        if idx in self.hidden_state[layer][direction]:
            h_t = self.hidden_state[layer][direction][idx]
            c_t = self.cell_state[layer][direction][idx]

            return h_t, c_t

        x_t = self._construct_x_t(layer, inputs, idx, tree)

        oidx, (h_prev, c_prev) = self._construct_previous(layer, direction,
                                                          inputs, tree, idx)

        if self.bias:
            Wih, Whh, bih, bhh = self._get_parameters(layer, direction)

            # print(Wih.size())
            # print(Whh.size())
            # print(bih.size())
            # print(bhh.size())

            # print(x_t.size())
            # print(h_prev.size())

            fcio_t_raw = torch.matmul(Whh, h_prev) +\
                torch.matmul(Wih, x_t[:, None]) +\
                bhh[:, None] + bih[:, None]

        else:
            Wih, Whh = self._get_parameters(layer, direction)

            fcio_t_raw = torch.matmul(Whh, h_prev) +\
                torch.matmul(Wih, x_t[:, None])

        f_t_raw, c_hat_t_raw, i_t_raw, o_t_raw = torch.split(fcio_t_raw,
                                                             self.hidden_size,
                                                             dim=0)

        f_t = F.sigmoid(f_t_raw)

        gated_children = torch.mul(f_t, c_prev)
        gated_children = torch.sum(gated_children, 1, keepdim=False)

        c_hat_t_raw = torch.sum(c_hat_t_raw, 1, keepdim=False)
        i_t_raw = torch.sum(i_t_raw, 1, keepdim=False)
        o_t_raw = torch.sum(o_t_raw, 1, keepdim=False)

        c_hat_t = self.__class__.nonlinearity(c_hat_t_raw)
        i_t = F.sigmoid(i_t_raw)
        o_t = F.sigmoid(o_t_raw)

        c_t = gated_children + torch.mul(i_t, c_hat_t)
        h_t = torch.mul(o_t, self.__class__.nonlinearity(c_t))

        if self.dropout:
            dropout = Dropout(p=self.dropout)
            h_t = dropout(h_t)
            c_t = dropout(c_t)

        self.hidden_state[layer][direction][idx] = h_t
        self.cell_state[layer][direction][idx] = c_t

        if direction == 'up' and self.bidirectional:
            self._upward_downward(layer, 'down', inputs, tree, idx)

        return h_t, c_t

    def _validate_inputs(self, inputs):
        if len(inputs.size()) == 3:
            self._has_batch_dimension = True
            try:
                assert inputs.size()[1] == 1
            except AssertionError:
                msg = 'ChildSumTreeLSTM assumes that dimension 1 of'
                msg += 'inputs is a batch dimension and, because it'
                msg += 'does not support minibatching, this dimension'
                msg += 'must always have size == 1'
                raise ValueError(msg)
        elif len(inputs.size()) == 2:
            self._has_batch_dimension = False
        else:
            msg = 'inputs must be 2D or 3D (with dimension 1 being'
            msg += 'a batch dimension)'
            raise ValueError(msg)

    def _get_parameters(self, layer, direction):
        dirtag = '' if direction == 'up' else '_reverse'

        Wihattrname = 'weight_ih_l{}{}'.format(str(layer), dirtag)
        Whhattrname = 'weight_hh_l{}{}'.format(str(layer), dirtag)

        Wih, Whh = getattr(self, Wihattrname), getattr(self, Whhattrname)

        if self.bias:
            bhhattrname = 'bias_hh_l{}{}'.format(str(layer), dirtag)
            bihattrname = 'bias_ih_l{}{}'.format(str(layer), dirtag)

            bih, bhh = getattr(self, bihattrname), getattr(self, bhhattrname)

            return Wih, Whh, bih, bhh

        else:
            return Wih, Whh

    @abstractmethod
    def _construct_x_t(self, layer, inputs, idx):
        raise NotImplementedError

    def _construct_previous(self, layer, direction, inputs, tree, idx):
        if direction == 'up':
            oidx = tree.children_idx(idx)
        else:
            oidx = tree.parents_idx(idx)

        if oidx:
            h_prev, c_prev = [], []

            for i in oidx:
                h_prev_i, c_prev_i = self._upward_downward(layer,
                                                           direction,
                                                           inputs,
                                                           tree, i)

                h_prev.append(h_prev_i)
                c_prev.append(c_prev_i)

            h_prev = torch.stack(h_prev, 1)
            c_prev = torch.stack(c_prev, 1)

        elif inputs.is_cuda:
            h_prev = torch.zeros(self.hidden_size, 1).cuda()
            c_prev = torch.zeros(self.hidden_size, 1).cuda()

        else:
            h_prev = torch.zeros(self.hidden_size, 1)
            c_prev = torch.zeros(self.hidden_size, 1)

        return oidx, (h_prev, c_prev)


class ChildSumDependencyTreeLSTM(ChildSumTreeLSTM):
    """A bidirectional extension of child-sum dependency tree LSTMs

    This module is constructed so as to be a drop-in replacement for
    the stock LSTM implemented in pytorch.nn.modules.rnn. It
    implements both bidirectional and unidirectional child-sum tree
    LSTMs for dependency trees. As such, it aims to minimally change
    that implementation's interface to allow for nontrivial tree
    topologies, and it exposes the parameters of the LSTM in the same
    way - i.e. the attribute names for the LSTM weights and biases are
    exactly the same as for the linear chain LSTM. The main difference
    between the linear chain version and this version is that
    forward() requires an nltk dependency graph representing a
    dependency tree for the input embeddings and does not require
    initial values for the hidden and cell states.

    """

    def _construct_x_t(self, layer, inputs, idx, tree):
        if layer > 0 and self.bidirectional:
            x_t = torch.cat([self.hidden_state[layer - 1]['up'][idx],
                             self.hidden_state[layer - 1]['down'][idx]])
        elif layer > 0:
            x_t = self.hidden_state[layer - 1]['up'][idx]
        elif self._has_batch_dimension:
            word_idx = tree.word_index(idx)
            x_t = inputs[word_idx, 0]
        else:
            word_idx = tree.word_index(idx)
            x_t = inputs[word_idx]

        return x_t


class ChildSumConstituencyTreeLSTM(ChildSumTreeLSTM):
    """A bidirectional extension of child-sum constituency tree LSTMs

    The main difference between this subclass of ChildSumTreeLSTM and
    the dependency tree subclass (ChildSumDependencyTreeLSTM) is that
    we don't necessarily have a word embedding for each node, and so
    we need to alter how we construct the inputs at a node to reflect
    this.

    Unlike ChildSumDependencyTreeLSTM, this subclass is harder to use
    as a drop-in replace for LSTM or GRU because its output hidden
    states will be larger than its inputs.
    """

    def _construct_x_t(self, layer, inputs, idx, tree):
        if layer > 0 and self.bidirectional:
            x_t = torch.cat([self.hidden_state[layer - 1]['up'][idx],
                             self.hidden_state[layer - 1]['down'][idx]])
        elif layer > 0:
            x_t = self.hidden_state[layer - 1]['up'][idx]
        else:
            if idx in tree.terminal_indices:
                string_idx = tree.terminal_indices.index(idx)

                if self._has_batch_dimension:
                    x_t = inputs[string_idx, 0]
                else:
                    x_t = inputs[string_idx]
            else:
                if self._has_batch_dimension:
                    x_t_raw = torch.zeros(self.input_size, 1)
                else:
                    x_t_raw = torch.zeros(self.input_size)

                if inputs.is_cuda:
                    x_t = x_t_raw.cuda()

                else:
                    x_t = x_t_raw

        return x_t
