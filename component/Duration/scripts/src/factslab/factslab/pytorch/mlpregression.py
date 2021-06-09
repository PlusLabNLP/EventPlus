from torch.nn import Module, Linear, ModuleDict, ModuleList
from torch.nn import MSELoss, L1Loss, SmoothL1Loss, CrossEntropyLoss, Dropout, BCELoss
import numpy as np
from sklearn.metrics import accuracy_score as acc, f1_score as f1, precision_score as prec, recall_score as rec, r2_score as r2, mean_squared_error as mse
import torch
import torch.nn.functional as F
from scipy.stats import mode
from collections import defaultdict
from functools import partial
from allennlp.commands.elmo import ElmoEmbedder
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from factslab.utility.bert import CustomBertTokenizer
import sys
# from allennlp.modules.elmo import Elmo, batch_to_ids
from os.path import expanduser
from tqdm import tqdm
import random

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

class MLPRegression(Module):
    def __init__(self, embed_params, attention_type, all_attributes,
                 output_size, layers, type_feats, token_feats, elmo, bert, glove, device="cpu"):
        '''
            Super class for training
        '''
        super(MLPRegression, self).__init__()

        # Set model constants and embeddings
        self.device = device
        self.layers = layers
        self.output_size = output_size
        self.attention_type = attention_type
        self.all_attributes = all_attributes
        self.type_feats = type_feats
        self.token_feats = token_feats
        self.glove = glove
        self.elmo = elmo
        self.bert = bert

        # Initialise embedding parameters
        self.embedding_dim = 0
        self._init_embeddings()

        # Initialise regression layers and parameters
        self._init_regression()

        # Initialise attention parameters
        self._init_attention()

    def _init_embeddings(self):
        '''
            Initialise embeddings
        '''
        if self.elmo:
            self.embedding_dim += 1024*3
            options_file = "/data/models/pytorch/elmo/options/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
            weight_file = "/data/models/pytorch/elmo/weights/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
            self.elmo_embedder = ElmoEmbedder(options_file, weight_file, cuda_device=0)
        if self.glove:
            self.embedding_dim += 300
            self.vocab = glove_embeddings.index.tolist()
            self.num_embeddings = glove_embeddings.shape[0]
            self.glove_embedder = torch.nn.Embedding(self.num_embeddings, self.embedding_dim, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
            self.glove_embedder.weight.data.copy_(torch.from_numpy(glove_embeddings.values))
            self.glove_embedder.weight.requires_grad = False
            self.vocab_hash = {w: i for i, w in enumerate(self.vocab)}
        if self.type:
            self.embedding_dim += type_dim
        if self.token:
            self.embedding_dim += token_dim
        if self.bert:
            self.bert_tokenizer = CustomBertTokenizer.from_pretrained("bert-large-cased", do_lower_case=False)
            self.bert_embedder = BertModel.from_pretrained("bert-large-cased").to(self.device)

    def _init_regression(self):
        '''
            Define the linear maps
        '''

        # Output regression parameters
        self.linmaps = ModuleDict({prot: ModuleList([]) for prot in self.all_attributes.keys()})

        for prot in self.all_attributes.keys():
            last_size = self.reduced_embedding_dim
            # Handle varying size of dimension depending on representation
            if self.attention_type[prot]['repr'] == "root":
                if self.attention_type[prot]['context'] != "none":
                    last_size *= 2
            else:
                if self.attention_type[prot]['context'] == "none":
                    last_size *= 2
                else:
                    last_size *= 3
            # self.layer_norm[prot] = torch.nn.LayerNorm(last_size)
            last_size += self.hand_feat_dim
            for out_size in self.layers:
                linmap = Linear(last_size, out_size)
                self.linmaps[prot].append(linmap)
                last_size = out_size
            final_linmap = Linear(last_size, self.output_size)
            self.linmaps[prot].append(final_linmap)

        # Dropout layer
        self.dropout = Dropout()

    def _regression_nonlinearity(self, x):
        return F.relu(x)

    def _init_attention(self):
        '''
            Initialises the attention map vector/matrix

            Takes attention_type-Span, Sentence, Span-param, Sentence-param
            as a parameter to decide the size of the attention matrix
        '''

        self.att_map_repr = ModuleDict({})
        self.att_map_W = ModuleDict({})
        self.att_map_V = ModuleDict({})
        self.att_map_context = ModuleDict({})
        for prot in self.attention_type.keys():
            # Token representation
            if self.attention_type[prot]['repr'] == "span":
                repr_dim = 2 * self.reduced_embedding_dim
                self.att_map_repr[prot] = Linear(self.reduced_embedding_dim, 1, bias=False)
                self.att_map_W[prot] = Linear(self.reduced_embedding_dim, self.reduced_embedding_dim)
                self.att_map_V[prot] = Linear(self.reduced_embedding_dim, 1, bias=False)
            elif self.attention_type[prot]['repr'] == "param":
                repr_dim = 2 * self.reduced_embedding_dim
                self.att_map_repr[prot] = Linear(self.reduced_embedding_dim, self.reduced_embedding_dim, bias=False)
                self.att_map_W[prot] = Linear(2 * self.reduced_embedding_dim, self.reduced_embedding_dim)
                self.att_map_V[prot] = Linear(self.reduced_embedding_dim, 1, bias=False)
            else:
                repr_dim = self.reduced_embedding_dim

            # Context representation
            # There is no attention for argument davidsonian
            if self.attention_type[prot]['context'] == 'param':
                self.att_map_context[prot] = Linear(repr_dim, self.reduced_embedding_dim, bias=False)
            elif self.attention_type[prot]['context'] == 'david' and prot == 'arg':
                self.att_map_context[prot] = Linear(repr_dim, self.reduced_embedding_dim, bias=False)

    def _choose_tokens(self, batch, lengths):
        '''
            Extracts tokens from a batch at specified position(lengths)
            batch - batch_size x max_sent_length x embed_dim
            lengths - batch_size x max_span_length x embed_dim
        '''
        idx = (lengths).unsqueeze(2).expand(-1, -1, batch.shape[2])
        return batch.gather(1, idx).squeeze()

    def _get_inputs(self, words):
        '''
           Return ELMO embeddings as root, span or param span
        '''
        if not self.vocab:
            raw_embeds, masks = self.embeddings.batch_to_embeddings(words)
            # raw_ = self.embeddings(batch_to_ids(words).to(self.device))
            # raw_embeds, masks = torch.cat([x.unsqueeze(1) for x in raw_['elmo_representations']], dim=1), raw_['mask']
            masks = masks.unsqueeze(2).repeat(1, 1, self.reduced_embedding_dim).byte()
            embedded_inputs = (
                self.embed_linmap_argpred_lower(raw_embeds[:, 0, :, :].squeeze()) +
                self.embed_linmap_argpred_mid(raw_embeds[:, 1, :, :].squeeze()) +
                self.embed_linmap_argpred_top(raw_embeds[:, 2, :, :].squeeze()))
            masked_embedded_inputs = embedded_inputs * masks.float()
            return masked_embedded_inputs, masks
        else:
            # Glove embeddings
            indices = [[self.vocab_hash[word] for word in sent] for sent in words]
            indices = torch.tensor(indices, dtype=torch.long, device=self.device)
            embeddings = self.embeddings(indices)
            masks = (embeddings != 0)[:, :, :self.reduced_embedding_dim].byte()
            # reduced_embeddings = self.embed_linmap(embeddings) * masks.float()
            return embeddings, masks

    def _get_representation(self, prot, embeddings, roots, spans,
                            context=False):
        '''
            returns the representation required from arguments passed by
            running attention based on arguments passed
        '''

        # Get token(pred/arg) representation
        rep_type = self.attention_type[prot]['repr']

        roots_rep_raw = self._choose_tokens(embeddings, roots)
        if len(roots_rep_raw.shape) == 1:
            roots_rep_raw = roots_rep_raw.unsqueeze(0)

        if rep_type == "root":
            token_rep = roots_rep_raw
        else:
            masks_spans = (spans == -1)
            spans[spans == -1] = 0
            spans_rep_raw = self._choose_tokens(embeddings, spans)

            if len(spans_rep_raw.shape) == 1:
                spans_rep_raw = spans_rep_raw.unsqueeze(0).unsqueeze(1)
            elif len(spans_rep_raw.shape) == 2:
                if spans.shape[0] == 1:
                    spans_rep_raw = spans_rep_raw.unsqueeze(0)
                elif spans.shape[1] == 1:
                    spans_rep_raw = spans_rep_raw.unsqueeze(1)

            if rep_type == "span":
                att_raw = self.att_map_repr[prot](spans_rep_raw).squeeze()
                # additive attention
                # att_raw_w = torch.relu(self.att_map_W[prot](for_att))
                # att_raw = self.att_map_V[prot](att_raw_w).squeeze()
            elif rep_type == "param":
                # att_param = torch.relu(self.att_map_repr[prot](roots_rep_raw)).unsqueeze(2)
                # att_raw = torch.matmul(spans_rep_raw, att_param).squeeze()
                # additive attention
                for_att = torch.cat((spans_rep_raw, roots_rep_raw.unsqueeze(1).repeat(1, spans_rep_raw.shape[1], 1)), dim=2)
                att_raw_w = torch.relu(self.att_map_W[prot](for_att))
                att_raw = self.att_map_V[prot](att_raw_w).squeeze()

            att_raw = att_raw.masked_fill(masks_spans, -1e9)
            att = F.softmax(att_raw, dim=1)
            att = self.dropout(att)
            pure_token_rep = torch.matmul(att.unsqueeze(2).permute(0, 2, 1),
                                     spans_rep_raw).squeeze()
            if not context:
                token_rep = torch.cat((roots_rep_raw, pure_token_rep), dim=1)
            else:
                token_rep = pure_token_rep

        return token_rep

    def _run_attention(self, prot, embeddings, roots, spans, context_roots, context_spans, masks):
        '''
            Various attention mechanisms implemented
        '''

        # Get the required representation for pred/arg
        token_rep = self._get_representation(prot=prot,
                                             embeddings=embeddings,
                                             roots=roots,
                                             spans=spans)

        # Get the required representation for context of pred/arg
        context_type = self.attention_type[prot]['context']

        if context_type == "none":
            context_rep = None

        elif context_type == "param":
            # Sentence level attention
            att_param = torch.relu(self.att_map_context[prot](token_rep)).unsqueeze(1)
            att_raw = torch.matmul(embeddings, att_param.permute(0, 2, 1))
            att_raw = att_raw.masked_fill(masks[:, :, 0:1] == 0, -1e9)
            att = F.softmax(att_raw, dim=1)
            att = self.dropout(att)
            context_rep = torch.matmul(att.permute(0, 2, 1), embeddings).squeeze()

        elif context_type == "david":
            if prot == "arg":
                prot_context = 'pred'
                context_roots = torch.tensor(context_roots, dtype=torch.long, device=self.device).unsqueeze(1)
                max_span = max([len(a) for a in context_spans])
                context_spans = torch.tensor([a + [-1 for i in range(max_span - len(a))] for a in context_spans], dtype=torch.long, device=self.device)
                context_rep = self._get_representation(context=True,
                    prot=prot_context, embeddings=embeddings,
                    roots=context_roots, spans=context_spans)
            else:
                prot_context = 'arg'
                context_rep = None
                for i, ctx_root in enumerate(context_roots):
                    ctx_root = torch.tensor(ctx_root, dtype=torch.long, device=self.device).unsqueeze(1)
                    max_span = max([len(a) for a in context_spans[i]])
                    ctx_span = torch.tensor([a + [-1 for i in range(max_span - len(a))] for a in context_spans[i]], dtype=torch.long, device=self.device)
                    sentence = embeddings[i, :, :].unsqueeze(0).repeat(len(ctx_span), 1, 1)
                    ctx_reps = self._get_representation(context=True,
                            prot=prot_context, embeddings=sentence,
                            roots=ctx_root, spans=ctx_span)

                    if len(ctx_reps.shape) == 1:
                        ctx_reps = ctx_reps.unsqueeze(0)
                    # Attention over arguments
                    att_nd_param = torch.relu(self.att_map_context[prot](token_rep[i, :].unsqueeze(0)))
                    att_raw = torch.matmul(att_nd_param, ctx_reps.permute(1, 0))
                    att = F.softmax(att_raw, dim=1)
                    ctx_rep_final = torch.matmul(att, ctx_reps)
                    if i:
                        context_rep = torch.cat((context_rep, ctx_rep_final), dim=0).squeeze()
                    else:
                        context_rep = ctx_rep_final

        if context_rep is not None:
            inputs_for_regression = torch.cat((token_rep, context_rep), dim=1)
        else:
            inputs_for_regression = token_rep

        return inputs_for_regression

    def _run_regression(self, prot, x):
        '''
            Run regression to get 3 attribute vector
        '''
        for i, lin_map in enumerate(self.linmaps[prot]):
            if i:
                x = self._regression_nonlinearity(x)
                x = self.dropout(x)

            x = lin_map(x)

        return torch.sigmoid(x)

    def forward(self, prot, words, roots, spans, context_roots,
                context_spans, hand_feats):
        """
            Forward propagation of activations
        """

        if self.is_embeds_on:
            inputs_for_attention, masks = self._get_inputs(words)
            inputs_for_regression = self._run_attention(prot=prot,
                                        embeddings=inputs_for_attention,
                                        roots=roots, spans=spans,
                                        context_roots=context_roots,
                                        context_spans=context_spans,
                                        masks=masks)
            if self.is_hand_feats_on:
                inputs_for_regression = torch.cat((inputs_for_regression, hand_feats), dim=1)
        elif self.is_hand_feats_on:
            inputs_for_regression = hand_feats
        else:
            sys.exit('You need some word representation!!')

        outputs = self._run_regression(prot=prot, x=inputs_for_regression)
        return outputs


class MLPTrainer:

    loss_function_map = {"linear": MSELoss,
                         "robust": L1Loss,
                         "robust_smooth": SmoothL1Loss,
                         "multinomial": BCELoss}

    def __init__(self, attention_type, all_attributes,
                 regressiontype="multinomial",
                 optimizer_class=torch.optim.Adam, device="cpu",
                 lr=0.001, weight_decay=0, **kwargs):
        '''

        '''
        self._regressiontype = regressiontype
        self._optimizer_class = optimizer_class
        self._init_kwargs = kwargs
        self._continuous = regressiontype != "multinomial"
        self.device = device
        self.att_type = attention_type
        self.all_attributes = all_attributes
        self.lr = lr
        self.weight_decay = weight_decay

    def _initialize_trainer_regression(self):
        '''

        '''
        lf_class = self.__class__.loss_function_map[self._regressiontype]
        if self._continuous:
            output_size = 1
            self._regression = MLPRegression(output_size=output_size,
                                            device=self.device,
                                            attention_type=self.att_type,
                                            all_attributes=self.all_attributes,
                                             **self._init_kwargs)
            self._loss_function = lf_class()
        else:
            output_size = 3
            self._regression = MLPRegression(output_size=output_size,
                                            device=self.device,
                                            attention_type=self.att_type,
                                            all_attributes=self.all_attributes,
                                             **self._init_kwargs)
            # self._loss_function = lf_class(reduction="none")
            self._loss_function = lf_class()

        self._regression = self._regression.to(self.device)
        self._loss_function = self._loss_function.to(self.device)

    def fit(self, X, Y, loss_wts, roots, spans, context_roots, context_spans,
            hand_feats, dev, epochs, prot):
        '''
            Fit X
        '''

        # Load the dev_data
        dev_x, dev_y, dev_roots, dev_spans, dev_context_roots, dev_context_spans, dev_wts, dev_hand_feats = dev

        self._initialize_trainer_regression()

        y_ = []
        loss_trace = []
        early_stop_acc = [0]

        parameters = [p for p in self._regression.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, weight_decay=self.weight_decay,
                                     lr=self.lr)
        epoch = 0

        while epoch < epochs:
            epoch += 1
            print("Epoch", epoch, "of", epochs)
            for x, y, rts, sps, croots, csps, wts, hfts in tqdm(zip(X, Y, roots, spans, context_roots, context_spans, loss_wts, hand_feats), total=len(X)):

                optimizer.zero_grad()

                rts = torch.tensor(rts, dtype=torch.long, device=self.device).unsqueeze(1)
                hfts = torch.tensor(hfts, dtype=torch.float, device=self.device)
                max_span = max([len(a) for a in sps])
                sps = torch.tensor([a + [-1 for i in range(max_span - len(a))] for a in sps], dtype=torch.long, device=self.device)

                # for attr in self.all_attributes[prot]:
                if self._continuous:
                    y = torch.tensor(y, dtype=torch.float, device=self.device)
                else:
                    y = torch.tensor(y, dtype=torch.float, device=self.device)
                wts = torch.tensor(wts, dtype=torch.float, device=self.device)

                y_ = self._regression(prot=prot, words=x, roots=rts,
                                      spans=sps, context_roots=croots,
                                      context_spans=csps, hand_feats=hfts)

                loss = self._loss_function(y_, y)
                # loss = torch.sum(loss * wts)
                loss.backward()
                optimizer.step()
                loss_trace.append(float(loss.data))

            # EARLY STOPPING
            dev_preds = {}
            self._regression = self._regression.eval()
            dev_attributes = dev_y.keys()
            dev_preds = self.predict(prot=prot,
                                     attributes=dev_attributes,
                                     X=dev_x,
                                     roots=dev_roots,
                                     spans=dev_spans,
                                     context_roots=dev_context_roots,
                                     context_spans=dev_context_spans,
                                     hand_feats=dev_hand_feats)

            print("Dev Metrics(Unweighted, Weighted)")
            early_stop_acc.append(self._print_metric(prot=prot,
                                                     loss_trace=loss_trace,
                                                     dev_preds=dev_preds,
                                                     dev_y=dev_y,
                                                     dev_wts=dev_wts))
            y_ = []
            loss_trace = []
            self._regression = self._regression.train()
            if early_stop_acc[-1] - early_stop_acc[-2] < -0.01:
                break
            else:
                name_of_model = (prot + str(self._regression.layers) +
                                 self.att_type['arg']['repr'] + "_" +
                                 self.att_type['arg']['context'] + "_" +
                                 self.att_type['pred']['repr'] + "_" +
                                 self.att_type['pred']['context'] + "_" +
                                 str(epoch))
                Path = expanduser('~') + "/Desktop/saved_models/" + name_of_model
                torch.save(self._regression.state_dict(), Path)

    def predict(self, prot, attributes, X, roots, spans, context_roots, context_spans, hand_feats):
        """Predict using the MLP regression

        Parameters
        ----------
        X : iterable(iterable(object))
            a matrix of structures (independent variables) with rows
            corresponding to a particular kind of RNN
        """
        self._regression = self._regression.eval()
        predictions = defaultdict(partial(np.ndarray, 0))
        for x, rts, sps, ctx_root, ctx_span, hfts in zip(X, roots, spans, context_roots, context_spans, hand_feats):

            rts = torch.tensor(rts, dtype=torch.long, device=self.device).unsqueeze(1)
            hfts = torch.tensor(hfts, dtype=torch.float, device=self.device)
            max_span = max([len(a) for a in sps])
            sps = torch.tensor([a + [-1 for i in range(max_span - len(a))] for a in sps], dtype=torch.long, device=self.device)

            y_dev = self._regression(prot=prot, words=x, roots=rts,
                                     spans=sps, context_roots=ctx_root,
                                     context_spans=ctx_span,
                                     hand_feats=hfts)
            y_dev = y_dev > 0.5
            for ind, attr in enumerate(attributes):
                if self._continuous:
                    predictions[attr] = np.concatenate([predictions[attr], y_dev[:, ind].detach().cpu().numpy()])
                else:
                    predictions[attr] = np.concatenate([predictions[attr], y_dev[:, ind].detach().cpu().numpy()])
            del y_dev
        return predictions
