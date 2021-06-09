import os, re, torch, html, tempfile, copy, json, math, shutil, tarfile, tempfile, sys, random, pickle
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, ReLU
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from transformers import RobertaTokenizer, BertForTokenClassification, BertTokenizer, BertConfig, BertModel, WordpieceTokenizer, XLNetTokenizer
# new
from transformers import XLNetForTokenClassification
from transformers.file_utils import cached_path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
# %matplotlib inline
from scipy import stats
from knockknock import email_sender, telegram_sender
# from data import *
# from metric import *
# from model import *

MAX_LEN = 128
bs = 8
EPOCHS = 60
PATIENCE = 6
INITIAL_LEARNING_RATE = 3e-5
NUM_RUNS = 1 #Number of times to run the training and evaluation code

CUE_MODEL = 'bert-base-uncased'
SCOPE_MODEL = 'xlnet-base-cased'
SCOPE_METHOD = 'augment' # Options: augment, replace
F1_METHOD = 'average' # Options: average, first_token
TASK = 'speculation' # Options: negation, speculation
SUBTASK = 'cue_detection' # Options: cue_detection, scope_resolution
TRAIN_DATASETS = ['sfu']
# TEST_DATASETS = ['bioscope_full_papers','bioscope_abstracts','sfu']
TEST_DATASETS = ['sfu']

# TELEGRAM_CHAT_ID = #Replace with chat ID for telegram notifications
# TELEGRAM_TOKEN = #Replace with token for telegram notifications

BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json"
}

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin"
}

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin"
}

ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json"
}

XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'xlnet-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-config.json"
}

XLNET_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'xlnet-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-pytorch_model.bin"
}

TF_WEIGHTS_NAME = 'model.ckpt'
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

device = torch.device("cuda")
n_gpu = torch.cuda.device_count()

class Cues:
    def __init__(self, data):
        self.sentences = data[0]
        self.cues = data[1]
        self.num_sentences = len(data[0])
class Scopes:
    def __init__(self, data):
        self.sentences = data[0]
        self.cues = data[1]
        self.scopes = data[2]
        self.num_sentences = len(data[0])

class Data:
    def __init__(self, file, dataset_name = 'sfu', frac_no_cue_sents = 1.0):
        '''
        file: The path of the data file.
        dataset_name: The name of the dataset to be preprocessed. Values supported: sfu, bioscope, starsem.
        frac_no_cue_sents: The fraction of sentences to be included in the data object which have no negation/speculation cues.
        '''
        def starsem(f_path, cue_sents_only=False, frac_no_cue_sents = 1.0):
            raw_data = open(f_path)
            sentence = []
            labels = []
            label = []
            scope_sents = []
            data_scope = []
            scope = []
            scope_cues = []
            data = []
            cue_only_data = []
            
            for line in raw_data:
                label = []
                sentence = []
                tokens = line.strip().split()
                if len(tokens)==8: #This line has no cues
                        sentence.append(tokens[3])
                        label.append(3) #Not a cue
                        for line in raw_data:
                            tokens = line.strip().split()
                            if len(tokens)==0:
                                break
                            else:
                                sentence.append(tokens[3])
                                label.append(3)
                        cue_only_data.append([sentence, label])
                        
                    
                else: #The line has 1 or more cues
                    num_cues = (len(tokens)-7)//3
                    #cue_count+=num_cues
                    scope = [[] for i in range(num_cues)]
                    label = [[],[]] #First list is the real labels, second list is to modify if it is a multi-word cue.
                    label[0].append(3) #Generally not a cue, if it is will be set ahead.
                    label[1].append(-1) #Since not a cue, for now.
                    for i in range(num_cues):
                        if tokens[7+3*i] != '_': #Cue field is active
                            if tokens[8+3*i] != '_': #Check for affix
                                label[0][-1] = 0 #Affix
                                affix_list.append(tokens[7+3*i])
                                label[1][-1] = i #Cue number
                                #sentence.append(tokens[7+3*i])
                                #new_word = '##'+tokens[8+3*i]
                            else:
                                label[0][-1] = 1 #Maybe a normal or multiword cue. The next few words will determine which.
                                label[1][-1] = i #Which cue field, for multiword cue altering.
                                
                        if tokens[8+3*i] != '_':
                            scope[i].append(1)
                        else:
                            scope[i].append(0)
                    sentence.append(tokens[3])
                    for line in raw_data:
                        tokens = line.strip().split()
                        if len(tokens)==0:
                            break
                        else:
                            sentence.append(tokens[3])
                            label[0].append(3) #Generally not a cue, if it is will be set ahead.
                            label[1].append(-1) #Since not a cue, for now.   
                            for i in range(num_cues):
                                if tokens[7+3*i] != '_': #Cue field is active
                                    if tokens[8+3*i] != '_': #Check for affix
                                        label[0][-1] = 0 #Affix
                                        label[1][-1] = i #Cue number
                                    else:
                                        label[0][-1] = 1 #Maybe a normal or multiword cue. The next few words will determine which.
                                        label[1][-1] = i #Which cue field, for multiword cue altering.
                                if tokens[8+3*i] != '_':
                                    scope[i].append(1)
                                else:
                                    scope[i].append(0)
                    for i in range(num_cues):
                        indices = [index for index,j in enumerate(label[1]) if i==j]
                        count = len(indices)
                        if count>1:
                            for j in indices:
                                label[0][j] = 2
                    for i in range(num_cues):
                        sc = []
                        for a,b in zip(label[0],label[1]):
                            if i==b:
                                sc.append(a)
                            else:
                                sc.append(3)
                        scope_cues.append(sc)
                        scope_sents.append(sentence)
                        data_scope.append(scope[i])
                    labels.append(label[0])
                    data.append(sentence)
            cue_only_samples = random.sample(cue_only_data, k=int(frac_no_cue_sents*len(cue_only_data)))
            cue_only_sents = [i[0] for i in cue_only_samples]
            cue_only_cues = [i[1] for i in cue_only_samples]
            starsem_cues = (data+cue_only_sents,labels+cue_only_cues)
            starsem_scopes = (scope_sents, scope_cues, data_scope)
            return [starsem_cues, starsem_scopes]
            
        def bioscope(f_path, cue_sents_only=False, frac_no_cue_sents = 1.0):
            file = open(f_path, encoding = 'utf-8')
            sentences = []
            for s in file:
                sentences+=re.split("(<.*?>)", html.unescape(s))
            cue_sentence = []
            cue_cues = []
            cue_only_data = []
            scope_cues = []
            scope_scopes = []
            scope_sentence = []
            sentence = []
            cue = {}
            scope = {}
            in_scope = []
            in_cue = []
            word_num = 0
            c_idx = []
            s_idx = []
            in_sentence = 0
            for token in sentences:
                if token == '':
                    continue
                elif '<sentence' in token:
                    in_sentence = 1
                elif '<cue' in token:
                    if TASK in token:
                        in_cue.append(str(re.split('(ref=".*?")',token)[1][4:]))
                        c_idx.append(str(re.split('(ref=".*?")',token)[1][4:]))
                        if c_idx[-1] not in cue.keys():
                            cue[c_idx[-1]] = []
                elif '</cue' in token:
                    in_cue = in_cue[:-1]
                elif '<xcope' in token:
                    #print(re.split('(id=".*?")',token)[1][3:])
                    in_scope.append(str(re.split('(id=".*?")',token)[1][3:]))
                    s_idx.append(str(re.split('(id=".*?")',token)[1][3:]))
                    scope[s_idx[-1]] = []
                elif '</xcope' in token:
                    in_scope = in_scope[:-1]
                elif '</sentence' in token:
                    #print(cue, scope)
                    if len(cue.keys())==0:
                        cue_only_data.append([sentence, [3]*len(sentence)])
                    else:
                        cue_sentence.append(sentence)
                        cue_cues.append([3]*len(sentence))
                        for i in cue.keys():
                            scope_sentence.append(sentence)
                            scope_cues.append([3]*len(sentence))
                            if len(cue[i])==1:
                                cue_cues[-1][cue[i][0]] = 1
                                scope_cues[-1][cue[i][0]] = 1
                            else:
                                for c in cue[i]:
                                    cue_cues[-1][c] = 2
                                    scope_cues[-1][c] = 2
                            scope_scopes.append([0]*len(sentence))

                            if i in scope.keys():
                                for s in scope[i]:
                                    scope_scopes[-1][s] = 1

                    sentence = []
                    cue = {}
                    scope = {}
                    in_scope = []
                    in_cue = []
                    word_num = 0
                    in_sentence = 0
                    c_idx = []
                    s_idx = []
                elif '<' not in token:
                    if in_sentence==1:
                        words = token.split()
                        sentence+=words
                        if len(in_cue)!=0:
                            for i in in_cue:
                                cue[i]+=[word_num+i for i in range(len(words))]
                        elif len(in_scope)!=0:
                            for i in in_scope:
                                scope[i]+=[word_num+i for i in range(len(words))]
                        word_num+=len(words)
            cue_only_samples = random.sample(cue_only_data, k=int(frac_no_cue_sents*len(cue_only_data)))
            cue_only_sents = [i[0] for i in cue_only_samples]
            cue_only_cues = [i[1] for i in cue_only_samples]
            return [(cue_sentence+cue_only_sents, cue_cues+cue_only_cues),(scope_sentence, scope_cues, scope_scopes)]
        
        def sfu_review(f_path, cue_sents_only=False, frac_no_cue_sents = 1.0):
            file = open(f_path, encoding = 'utf-8')
            sentences = []
            for s in file:
                sentences+=re.split("(<.*?>)", html.unescape(s))
            cue_sentence = []
            cue_cues = []
            scope_cues = []
            scope_scopes = []
            scope_sentence = []
            sentence = []
            cue = {}
            scope = {}
            in_scope = []
            in_cue = []
            word_num = 0
            c_idx = []
            cue_only_data = []
            s_idx = []
            in_word = 0
            for token in sentences:
                if token == '':
                    continue
                elif token == '<W>':
                    in_word = 1
                elif token == '</W>':
                    in_word = 0
                    word_num += 1
                elif '<cue' in token:
                    if TASK in token:
                        in_cue.append(int(re.split('(ID=".*?")',token)[1][4:-1]))
                        c_idx.append(int(re.split('(ID=".*?")',token)[1][4:-1]))
                        if c_idx[-1] not in cue.keys():
                            cue[c_idx[-1]] = []
                elif '</cue' in token:
                    in_cue = in_cue[:-1]
                elif '<xcope' in token:
                    continue
                elif '</xcope' in token:
                    in_scope = in_scope[:-1]
                elif '<ref' in token:
                    in_scope.append([int(i) for i in re.split('(SRC=".*?")',token)[1][5:-1].split(' ')])
                    s_idx.append([int(i) for i in re.split('(SRC=".*?")',token)[1][5:-1].split(' ')])
                    for i in s_idx[-1]:
                        scope[i] = []
                elif '</SENTENCE' in token:
                    if len(cue.keys())==0:
                        cue_only_data.append([sentence, [3]*len(sentence)])
                    else:
                        cue_sentence.append(sentence)
                        cue_cues.append([3]*len(sentence))
                        for i in cue.keys():
                            scope_sentence.append(sentence)
                            scope_cues.append([3]*len(sentence))
                            if len(cue[i])==1:
                                cue_cues[-1][cue[i][0]] = 1
                                scope_cues[-1][cue[i][0]] = 1
                            else:
                                for c in cue[i]:
                                    cue_cues[-1][c] = 2
                                    scope_cues[-1][c] = 2
                            scope_scopes.append([0]*len(sentence))
                            if i in scope.keys():
                                for s in scope[i]:
                                    scope_scopes[-1][s] = 1
                    sentence = []
                    cue = {}
                    scope = {}
                    in_scope = []
                    in_cue = []
                    word_num = 0
                    in_word = 0
                    c_idx = []
                    s_idx = []
                elif '<' not in token:
                    if in_word == 1:
                        if len(in_cue)!=0:
                            for i in in_cue:
                                cue[i].append(word_num)
                        if len(in_scope)!=0:
                            for i in in_scope:
                                for j in i:
                                    scope[j].append(word_num)
                        sentence.append(token)
            cue_only_samples = random.sample(cue_only_data, k=int(frac_no_cue_sents*len(cue_only_data)))
            cue_only_sents = [i[0] for i in cue_only_samples]
            cue_only_cues = [i[1] for i in cue_only_samples]
            return [(cue_sentence+cue_only_sents, cue_cues+cue_only_cues),(scope_sentence, scope_cues, scope_scopes)]
        
        
        if dataset_name == 'bioscope':
            ret_val = bioscope(file, frac_no_cue_sents=frac_no_cue_sents)
            self.cue_data = Cues(ret_val[0])
            self.scope_data = Scopes(ret_val[1])
        elif dataset_name == 'sfu':
            sfu_cues = [[], []]
            sfu_scopes = [[], [], []]
            for dir_name in os.listdir(file):
                if '.' not in dir_name:
                    for f_name in os.listdir(file+"//"+dir_name):
                        r_val = sfu_review(file+"//"+dir_name+'//'+f_name, frac_no_cue_sents=frac_no_cue_sents)
                        sfu_cues = [a+b for a,b in zip(sfu_cues, r_val[0])]
                        sfu_scopes = [a+b for a,b in zip(sfu_scopes, r_val[1])]
            self.cue_data = Cues(sfu_cues)
            self.scope_data = Scopes(sfu_scopes)
        elif dataset_name == 'starsem':
            if TASK == 'negation':
                ret_val = starsem(file, frac_no_cue_sents=frac_no_cue_sents)
                self.cue_data = Cues(ret_val[0])
                self.scope_data = Scopes(ret_val[1])
            else:
                raise ValueError("Starsem 2012 dataset only supports negation annotations")
        else:
            raise ValueError("Supported Dataset types are:\n\tbioscope\n\tsfu\n\tconll_cue")
    
    def get_cue_dataloader(self, val_size = 0.15, test_size = 0.15, other_datasets = []):
        '''
        This function returns the dataloader for the cue detection.
        val_size: The size of the validation dataset (Fraction between 0 to 1)
        test_size: The size of the test dataset (Fraction between 0 to 1)
        other_datasets: Other datasets to use to get one combined train dataloader
        Returns: train_dataloader, list of validation dataloaders, list of test dataloaders
        '''
        do_lower_case = True
        if 'uncased' not in CUE_MODEL:
            do_lower_case = False
        if 'xlnet' in CUE_MODEL:
            tokenizer = XLNetTokenizer.from_pretrained(CUE_MODEL, do_lower_case=do_lower_case, cache_dir='xlnet_tokenizer')
        elif 'roberta' in CUE_MODEL:
            tokenizer = RobertaTokenizer.from_pretrained(CUE_MODEL, do_lower_case=do_lower_case, cache_dir='roberta_tokenizer')
        elif 'bert' in CUE_MODEL:
            tokenizer = BertTokenizer.from_pretrained(CUE_MODEL, do_lower_case=do_lower_case, cache_dir='bert_tokenizer')
        def preprocess_data(obj, tokenizer):
            dl_sents = obj.cue_data.sentences
            dl_cues = obj.cue_data.cues
                
            sentences = [" ".join(sent) for sent in dl_sents]

            mytexts = []
            mylabels = []
            mymasks = []
            if do_lower_case == True:
                sentences_clean = [sent.lower() for sent in sentences]
            else:
                sentences_clean = sentences
            for sent, tags in zip(sentences_clean,dl_cues):
                new_tags = []
                new_text = []
                new_masks = []
                for word, tag in zip(sent.split(),tags):
                    sub_words = tokenizer._tokenize(word)
                    for count, sub_word in enumerate(sub_words):
                        if type(tag)!=int:
                            raise ValueError(tag)
                        mask = 1
                        if count > 0:
                            mask = 0
                        new_masks.append(mask)
                        new_tags.append(tag)
                        new_text.append(sub_word)
                mymasks.append(new_masks)
                mytexts.append(new_text)
                mylabels.append(new_tags)

            
            input_ids = pad_sequences([[tokenizer._convert_token_to_id(word) for word in txt] for txt in mytexts],
                                      maxlen=MAX_LEN, dtype="long", truncating="post", padding="post").tolist()

            tags = pad_sequences(mylabels,
                                maxlen=MAX_LEN, value=4, padding="post",
                                dtype="long", truncating="post").tolist()
            
            mymasks = pad_sequences(mymasks, maxlen=MAX_LEN, value=0, padding='post', dtype='long', truncating='post').tolist()
            
            attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
            
            random_state = np.random.randint(1,2019)

            tra_inputs, test_inputs, tra_tags, test_tags = train_test_split(input_ids, tags, test_size=test_size, random_state = random_state)
            tra_masks, test_masks, _, _ = train_test_split(attention_masks, input_ids, test_size=test_size, random_state = random_state)
            tra_mymasks, test_mymasks, _, _ = train_test_split(mymasks, input_ids, test_size=test_size, random_state = random_state)
            
            random_state_2 = np.random.randint(1,2019)

            tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(tra_inputs, tra_tags, test_size=(val_size/(1-test_size)), random_state = random_state_2)
            tr_masks, val_masks, _, _ = train_test_split(tra_masks, tra_inputs, test_size=(val_size/(1-test_size)), random_state = random_state_2)
            tr_mymasks, val_mymasks, _, _ = train_test_split(tra_mymasks, tra_inputs, test_size=(val_size/(1-test_size)), random_state = random_state_2)
            return [tr_inputs, tr_tags, tr_masks, tr_mymasks], [val_inputs, val_tags, val_masks, val_mymasks], [test_inputs, test_tags, test_masks, test_mymasks]

        tr_inputs = []
        tr_tags = []
        tr_masks = []
        tr_mymasks = []
        val_inputs = [[] for i in range(len(other_datasets)+1)]
        test_inputs = [[] for i in range(len(other_datasets)+1)]

        train_ret_val, val_ret_val, test_ret_val = preprocess_data(self, tokenizer)
        tr_inputs+=train_ret_val[0]
        tr_tags+=train_ret_val[1]
        tr_masks+=train_ret_val[2]
        tr_mymasks+=train_ret_val[3]
        val_inputs[0].append(val_ret_val[0])
        val_inputs[0].append(val_ret_val[1])
        val_inputs[0].append(val_ret_val[2])
        val_inputs[0].append(val_ret_val[3])
        test_inputs[0].append(test_ret_val[0])
        test_inputs[0].append(test_ret_val[1])
        test_inputs[0].append(test_ret_val[2])
        test_inputs[0].append(test_ret_val[3])
        
        for idx, arg in enumerate(other_datasets, 1):
            train_ret_val, val_ret_val, test_ret_val = preprocess_data(arg, tokenizer)
            tr_inputs+=train_ret_val[0]
            tr_tags+=train_ret_val[1]
            tr_masks+=train_ret_val[2]
            tr_mymasks+=train_ret_val[3]
            val_inputs[idx].append(val_ret_val[0])
            val_inputs[idx].append(val_ret_val[1])
            val_inputs[idx].append(val_ret_val[2])
            val_inputs[idx].append(val_ret_val[3])
            test_inputs[idx].append(test_ret_val[0])
            test_inputs[idx].append(test_ret_val[1])
            test_inputs[idx].append(test_ret_val[2])
            test_inputs[idx].append(test_ret_val[3])
        
        tr_inputs = torch.LongTensor(tr_inputs)
        tr_tags = torch.LongTensor(tr_tags)
        tr_masks = torch.LongTensor(tr_masks)
        tr_mymasks = torch.LongTensor(tr_mymasks)
        val_inputs = [[torch.LongTensor(i) for i in j] for j in val_inputs]
        test_inputs = [[torch.LongTensor(i) for i in j] for j in test_inputs]

        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags, tr_mymasks)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

        val_dataloaders = []
        for i,j,k,l in val_inputs:
            val_data = TensorDataset(i, k, j, l)
            val_sampler = RandomSampler(val_data)
            val_dataloaders.append(DataLoader(val_data, sampler=val_sampler, batch_size=bs))

        test_dataloaders = []
        for i,j,k,l in test_inputs:
            test_data = TensorDataset(i, k, j, l)
            test_sampler = RandomSampler(test_data)
            test_dataloaders.append(DataLoader(test_data, sampler=test_sampler, batch_size=bs))

        return train_dataloader, val_dataloaders, test_dataloaders

    def get_scope_dataloader(self, val_size = 0.15, test_size=0.15, other_datasets = []):
        '''
        This function returns the dataloader for the cue detection.
        val_size: The size of the validation dataset (Fraction between 0 to 1)
        test_size: The size of the test dataset (Fraction between 0 to 1)
        other_datasets: Other datasets to use to get one combined train dataloader
        Returns: train_dataloader, list of validation dataloaders, list of test dataloaders
        '''
        method = SCOPE_METHOD
        do_lower_case = True
        if 'uncased' not in SCOPE_MODEL:
            do_lower_case = False
        if 'xlnet' in SCOPE_MODEL:
            tokenizer = XLNetTokenizer.from_pretrained(SCOPE_MODEL, do_lower_case=do_lower_case, cache_dir='xlnet_tokenizer')
        elif 'roberta' in SCOPE_MODEL:
            tokenizer = RobertaTokenizer.from_pretrained(SCOPE_MODEL, do_lower_case=do_lower_case, cache_dir='roberta_tokenizer')
        elif 'bert' in SCOPE_MODEL:
            tokenizer = BertTokenizer.from_pretrained(SCOPE_MODEL, do_lower_case=do_lower_case, cache_dir='bert_tokenizer')
        def preprocess_data(obj, tokenizer_obj):
            dl_sents = obj.scope_data.sentences
            dl_cues = obj.scope_data.cues
            dl_scopes = obj.scope_data.scopes
            
            sentences = [" ".join([s for s in sent]) for sent in dl_sents]
            mytexts = []
            mylabels = []
            mycues = []
            mymasks = []
            if do_lower_case == True:
                sentences_clean = [sent.lower() for sent in sentences]
            else:
                sentences_clean = sentences
            
            for sent, tags, cues in zip(sentences_clean,dl_scopes, dl_cues):
                new_tags = []
                new_text = []
                new_cues = []
                new_masks = []
                for word, tag, cue in zip(sent.split(),tags,cues):
                    sub_words = tokenizer_obj._tokenize(word)
                    for count, sub_word in enumerate(sub_words):
                        mask = 1
                        if count > 0:
                            mask = 0
                        new_masks.append(mask)
                        new_tags.append(tag)
                        new_cues.append(cue)
                        new_text.append(sub_word)
                mymasks.append(new_masks)
                mytexts.append(new_text)
                mylabels.append(new_tags)
                mycues.append(new_cues)
            final_sentences = []
            final_labels = []
            final_masks = []
            if method == 'replace':
                for sent,cues in zip(mytexts, mycues):
                    temp_sent = []
                    for token,cue in zip(sent,cues):
                        if cue==3:
                            temp_sent.append(token)
                        else:
                            temp_sent.append(f'[unused{cue+1}]')
                    final_sentences.append(temp_sent)
                final_labels = mylabels
                final_masks = mymasks
            elif method == 'augment':
                for sent,cues,labels,masks in zip(mytexts, mycues, mylabels, mymasks):
                    temp_sent = []
                    temp_label = []
                    temp_masks = []
                    first_part = 0
                    for token,cue,label,mask in zip(sent,cues,labels,masks):
                        if cue!=3:
                            if first_part == 0:
                                first_part = 1
                                temp_sent.append(f'[unused{cue+1}]')
                                temp_masks.append(1)
                                temp_label.append(label)
                                temp_sent.append(token)
                                temp_masks.append(0)
                                temp_label.append(label)
                                continue
                            temp_sent.append(f'[unused{cue+1}]')
                            temp_masks.append(0)
                            temp_label.append(label)
                        else:
                            first_part = 0
                        temp_masks.append(mask)
                        temp_sent.append(token)
                        temp_label.append(label)
                    final_sentences.append(temp_sent)
                    final_labels.append(temp_label)
                    final_masks.append(temp_masks)
            else:
                raise ValueError("Supported methods for scope detection are:\nreplace\naugment")
            input_ids = pad_sequences([[tokenizer_obj._convert_token_to_id(word) for word in txt] for txt in final_sentences],
                                      maxlen=MAX_LEN, dtype="long", truncating="post", padding="post").tolist()

            tags = pad_sequences(final_labels,
                                maxlen=MAX_LEN, value=0, padding="post",
                                dtype="long", truncating="post").tolist()
            
            final_masks = pad_sequences(final_masks,
                                maxlen=MAX_LEN, value=0, padding="post",
                                dtype="long", truncating="post").tolist()

            attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
            
            random_state = np.random.randint(1,2019)

            tra_inputs, test_inputs, tra_tags, test_tags = train_test_split(input_ids, tags, test_size=test_size, random_state = random_state)
            tra_masks, test_masks, _, _ = train_test_split(attention_masks, input_ids, test_size=test_size, random_state = random_state)
            tra_mymasks, test_mymasks, _, _ = train_test_split(final_masks, input_ids, test_size=test_size, random_state = random_state)
            
            random_state_2 = np.random.randint(1,2019)

            tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(tra_inputs, tra_tags, test_size=(val_size/(1-test_size)), random_state = random_state_2)
            tr_masks, val_masks, _, _ = train_test_split(tra_masks, tra_inputs, test_size=(val_size/(1-test_size)), random_state = random_state_2)
            tr_mymasks, val_mymasks, _, _ = train_test_split(tra_mymasks, tra_inputs, test_size=(val_size/(1-test_size)), random_state = random_state_2)

            return [tr_inputs, tr_tags, tr_masks, tr_mymasks], [val_inputs, val_tags, val_masks, val_mymasks], [test_inputs, test_tags, test_masks, test_mymasks]

        tr_inputs = []
        tr_tags = []
        tr_masks = []
        tr_mymasks = []
        val_inputs = [[] for i in range(len(other_datasets)+1)]
        test_inputs = [[] for i in range(len(other_datasets)+1)]

        train_ret_val, val_ret_val, test_ret_val = preprocess_data(self, tokenizer)
        tr_inputs+=train_ret_val[0]
        tr_tags+=train_ret_val[1]
        tr_masks+=train_ret_val[2]
        tr_mymasks+=train_ret_val[3]
        val_inputs[0].append(val_ret_val[0])
        val_inputs[0].append(val_ret_val[1])
        val_inputs[0].append(val_ret_val[2])
        val_inputs[0].append(val_ret_val[3])
        test_inputs[0].append(test_ret_val[0])
        test_inputs[0].append(test_ret_val[1])
        test_inputs[0].append(test_ret_val[2])
        test_inputs[0].append(test_ret_val[3])
        
        for idx, arg in enumerate(other_datasets, 1):
            train_ret_val, val_ret_val, test_ret_val = preprocess_data(arg, tokenizer)
            tr_inputs+=train_ret_val[0]
            tr_tags+=train_ret_val[1]
            tr_masks+=train_ret_val[2]
            tr_mymasks+=train_ret_val[3]
            val_inputs[idx].append(val_ret_val[0])
            val_inputs[idx].append(val_ret_val[1])
            val_inputs[idx].append(val_ret_val[2])
            val_inputs[idx].append(val_ret_val[3])
            test_inputs[idx].append(test_ret_val[0])
            test_inputs[idx].append(test_ret_val[1])
            test_inputs[idx].append(test_ret_val[2])
            test_inputs[idx].append(test_ret_val[3])

        tr_inputs = torch.LongTensor(tr_inputs)
        tr_tags = torch.LongTensor(tr_tags)
        tr_masks = torch.LongTensor(tr_masks)
        tr_mymasks = torch.LongTensor(tr_mymasks)
        val_inputs = [[torch.LongTensor(i) for i in j] for j in val_inputs]
        test_inputs = [[torch.LongTensor(i) for i in j] for j in test_inputs]

        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags, tr_mymasks)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

        val_dataloaders = []
        for i,j,k,l in val_inputs:
            val_data = TensorDataset(i, k, j, l)
            val_sampler = RandomSampler(val_data)
            val_dataloaders.append(DataLoader(val_data, sampler=val_sampler, batch_size=bs))

        test_dataloaders = []
        for i,j,k,l in test_inputs:
            test_data = TensorDataset(i, k, j, l)
            test_sampler = RandomSampler(test_data)
            test_dataloaders.append(DataLoader(test_data, sampler=test_sampler, batch_size=bs))

        return train_dataloader, val_dataloaders, test_dataloaders

class CustomData:
    def __init__(self, sentences, cues = None):
        self.sentences = sentences
        self.cues = cues
    def get_cue_dataloader(self):
        do_lower_case = True
        if 'uncased' not in CUE_MODEL:
            do_lower_case = False
        if 'xlnet' in CUE_MODEL:
            tokenizer = XLNetTokenizer.from_pretrained(CUE_MODEL, do_lower_case=do_lower_case, cache_dir='xlnet_tokenizer')
        elif 'roberta' in CUE_MODEL:
            tokenizer = RobertaTokenizer.from_pretrained(CUE_MODEL, do_lower_case=do_lower_case, cache_dir='roberta_tokenizer')
        elif 'bert' in CUE_MODEL:
            tokenizer = BertTokenizer.from_pretrained(CUE_MODEL, do_lower_case=do_lower_case, cache_dir='bert_tokenizer')
        
        dl_sents = self.sentences    
        sentences = dl_sents # sentences = [" ".join(sent) for sent in dl_sents]

        mytexts = []
        mylabels = []
        mymasks = []
        if do_lower_case == True:
            sentences_clean = [sent.lower() for sent in sentences]
        else:
            sentences_clean = sentences
        for sent in sentences_clean:
            new_text = []
            new_masks = []
            for word in sent.split():
                sub_words = tokenizer._tokenize(word)
                for count, sub_word in enumerate(sub_words):
                    mask = 1
                    if count > 0:
                        mask = 0
                    new_masks.append(mask)
                    new_text.append(sub_word)
            mymasks.append(new_masks)
            mytexts.append(new_text)
            
        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in mytexts],
                                  maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        mymasks = pad_sequences(mymasks, maxlen=MAX_LEN, value=0, padding='post', dtype='long', truncating='post').tolist()

        attention_masks = [[float(i>0) for i in ii] for ii in input_ids]

        inputs = torch.LongTensor(input_ids)
        masks = torch.LongTensor(attention_masks)
        mymasks = torch.LongTensor(mymasks)

        data = TensorDataset(inputs, masks, mymasks)
        dataloader = DataLoader(data, batch_size=bs)

        return dataloader
    
    def get_scope_dataloader(self, cues = None):
        if cues != None:
            self.cues = cues
        if self.cues == None:
            raise ValueError("Need Cues Data to Generate the Scope Dataloader")
        method = SCOPE_METHOD
        do_lower_case = True
        if 'uncased' not in SCOPE_MODEL:
            do_lower_case = False
        if 'xlnet' in SCOPE_MODEL:
            tokenizer = XLNetTokenizer.from_pretrained(SCOPE_MODEL, do_lower_case=do_lower_case, cache_dir='xlnet_tokenizer')
        elif 'roberta' in SCOPE_MODEL:
            tokenizer = RobertaTokenizer.from_pretrained(SCOPE_MODEL, do_lower_case=do_lower_case, cache_dir='roberta_tokenizer')
        elif 'bert' in SCOPE_MODEL:
            tokenizer = BertTokenizer.from_pretrained(SCOPE_MODEL, do_lower_case=do_lower_case, cache_dir='bert_tokenizer')
        dl_sents = self.sentences
        dl_cues = self.cues
        
        sentences = dl_sents
        mytexts = []
        mycues = []
        mymasks = []
        if do_lower_case == True:
            sentences_clean = [sent.lower() for sent in sentences]
        else:
            sentences_clean = sentences
        
        for sent, cues in zip(sentences_clean, dl_cues):
            new_text = []
            new_cues = []
            new_masks = []
            for word, cue in zip(sent.split(), cues):
                sub_words = tokenizer._tokenize(word)
                for count, sub_word in enumerate(sub_words):
                    mask = 1
                    if count > 0:
                        mask = 0
                    new_masks.append(mask)
                    new_cues.append(cue)
                    new_text.append(sub_word)
            mymasks.append(new_masks)
            mytexts.append(new_text)
            mycues.append(new_cues)
        final_sentences = []
        final_masks = []
        if method == 'replace':
            for sent,cues in zip(mytexts, mycues):
                temp_sent = []
                for token,cue in zip(sent,cues):
                    if cue==3:
                        temp_sent.append(token)
                    else:
                        temp_sent.append(f'[unused{cue+1}]')
                final_sentences.append(temp_sent)
            final_masks = mymasks
        elif method == 'augment':
            for sent,cues,masks in zip(mytexts, mycues, mymasks):
                temp_sent = []
                temp_masks = []
                first_part = 0
                for token,cue,mask in zip(sent,cues,masks):
                    if cue!=3:
                        if first_part == 0:
                            first_part = 1
                            temp_sent.append(f'[unused{cue+1}]')
                            temp_masks.append(1)
                            #temp_label.append(label)
                            temp_sent.append(token)
                            temp_masks.append(0)
                            #temp_label.append(label)
                            continue
                        temp_sent.append(f'[unused{cue+1}]')
                        temp_masks.append(0)
                        #temp_label.append(label)
                    else:
                        first_part = 0
                    temp_masks.append(mask)
                    temp_sent.append(token)
                final_sentences.append(temp_sent)
                final_masks.append(temp_masks)
        else:
            raise ValueError("Supported methods for scope detection are:\nreplace\naugment")
    
        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in final_sentences],
                                  maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        
        final_masks = pad_sequences(final_masks,
                                maxlen=MAX_LEN, value=0, padding="post",
                                dtype="long", truncating="post").tolist()

        attention_masks = [[float(i>0) for i in ii] for ii in input_ids]

        inputs = torch.LongTensor(input_ids)
        masks = torch.LongTensor(attention_masks)
        final_masks = torch.LongTensor(final_masks)

        data = TensorDataset(inputs, masks, final_masks)
        dataloader = DataLoader(data, batch_size=bs)
        #print(final_sentences, mycues)

        return dataloader
    
#
# METRICS
#

def f1_cues(y_true, y_pred):
    '''Needs flattened cues'''
    tp = sum([1 for i,j in zip(y_true, y_pred) if (i==j and i!=3)])
    fp = sum([1 for i,j in zip(y_true, y_pred) if (j!=3 and i==3)])
    fn = sum([1 for i,j in zip(y_true, y_pred) if (i!=3 and j==3)])
    if tp==0:
        prec = 0.0001
        rec = 0.0001
    else:
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {2*prec*rec/(prec+rec)}")
    return prec, rec, 2*prec*rec/(prec+rec)
    
    
def f1_scope(y_true, y_pred, level = 'token'): #This is for gold cue annotation scope, thus the precision is always 1.
    if level == 'token':
        print(f1_score([i for i in j for j in y_true], [i for i in j for j in y_pred]))
    elif level == 'scope':
        tp = 0
        fn = 0
        fp = 0
        for y_t, y_p in zip(y_true, y_pred):
            if y_t == y_p:
                tp+=1
            else:
                fn+=1
        prec = 1
        rec = tp/(tp+fn)
        print(f"Precision: {prec}")
        print(f"Recall: {rec}")
        print(f"F1 Score: {2*prec*rec/(prec+rec)}")

def report_per_class_accuracy(y_true, y_pred):
    labels = list(np.unique(y_true))
    lab = list(np.unique(y_pred))
    labels = list(np.unique(labels+lab))
    n_labels = len(labels)
    data = pd.DataFrame(columns = labels, index = labels, data = np.zeros((n_labels, n_labels)))
    for i,j in zip(y_true, y_pred):
        data.at[i,j]+=1
    print(data)
    
def flat_accuracy(preds, labels, input_mask = None):
    pred_flat = [i for j in preds for i in j]
    labels_flat = [i for j in labels for i in j]
    return sum([1 if i==j else 0 for i,j in zip(pred_flat,labels_flat)]) / len(labels_flat)
    

def flat_accuracy_positive_cues(preds, labels, input_mask = None):
    pred_flat = [i for i,j in zip([i for j in preds for i in j],[i for j in labels for i in j]) if (j!=4 and j!=3)]
    labels_flat = [i for i in [i for j in labels for i in j] if (i!=4 and i!=3)]
    if len(labels_flat) != 0:
        return sum([1 if i==j else 0 for i,j in zip(pred_flat,labels_flat)]) / len(labels_flat)
    else:
        return None

def scope_accuracy(preds, labels):
    correct_count = 0
    count = 0
    for i,j in zip(preds, labels):
        if i==j:
            correct_count+=1
        count+=1
    return correct_count/count

#
# EARLYSTOPPING CODE
#

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = 0

    def __call__(self, score, model):

        #score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation F1 increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

# MODELS
# 
#

class CueModel:
    def __init__(self, full_finetuning = True, train = False, pretrained_model_path = 'Cue_Detection.pickle', device = 'cuda', learning_rate = 3e-5, class_weight = [100, 100, 100, 1, 0], num_labels = 5):
        self.model_name = CUE_MODEL
        self.task = TASK
        if 'xlnet' in CUE_MODEL:
            self.model = XLNetForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels, cache_dir = 'xlnet-base-cased-model')
        elif 'roberta' in CUE_MODEL:
            self.model = RobertaForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels, cache_dir = 'roberta-base-model')
        elif 'bert' in CUE_MODEL:
            self.model = BertForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels, cache_dir = 'bert_base_uncased_model')
        else:
            raise ValueError("Supported model types are: xlnet, roberta, bert")
        if train == False:
            # self.model = torch.load(pretrained_model_path)
            self.model.load_state_dict(torch.load(pretrained_model_path))
            print('loaded CueModel')
        self.device = torch.device(device)
        self.class_weight = class_weight
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        if device == 'cuda':
            self.model.cuda()
        else:
            self.model.cpu()
            
        if full_finetuning:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            if intermediate_neurons == None:
                param_optimizer = list(self.model.classifier.named_parameters()) 
            else:
                param_optimizer = list(self.model.classifier.named_parameters())+list(self.model.int_layer.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        self.optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate)

    # @telegram_sender(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)  
    def train(self, train_dataloader, valid_dataloaders, train_dl_name, val_dl_name, epochs = 5, max_grad_norm = 1.0, patience = 3):
        
        self.train_dl_name = train_dl_name
        return_dict = {"Task": f"{self.task} Cue Detection",
                       "Model": self.model_name,
                       "Train Dataset": train_dl_name,
                       "Val Dataset": val_dl_name,
                       "Best Precision": 0,
                       "Best Recall": 0,
                       "Best F1": 0}
        train_loss = []
        valid_loss = []
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        loss_fn = CrossEntropyLoss(weight=torch.Tensor(self.class_weight).to(self.device))
        for _ in tqdm(range(epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_mymasks = batch
                logits = self.model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels = b_labels.view(-1)[active_loss]
                loss = loss_fn(active_logits, active_labels)
                loss.backward()
                tr_loss += loss.item()
                if step % 100 == 0:
                    print(f"Batch {step}, loss {loss.item()}")
                train_loss.append(loss.item())
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad()
            print("Train loss: {}".format(tr_loss/nb_tr_steps))
            
            self.model.eval()
            eval_loss, eval_accuracy, eval_scope_accuracy, eval_positive_cue_accuracy = 0, 0, 0, 0
            nb_eval_steps, nb_eval_examples, steps_positive_cue_accuracy = 0, 0, 0
            predictions , true_labels, ip_mask = [], [], []
            for valid_dataloader in valid_dataloaders:
                for batch in valid_dataloader:
                    batch = tuple(t.to(self.device) for t in batch)
                    b_input_ids, b_input_mask, b_labels, b_mymasks = batch

                    with torch.no_grad():
                        logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
                        active_loss = b_input_mask.view(-1) == 1
                        active_logits = logits.view(-1, self.num_labels)[active_loss] #5 is num_labels
                        active_labels = b_labels.view(-1)[active_loss]
                        tmp_eval_loss = loss_fn(active_logits, active_labels)
                        
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()
                    mymasks = b_mymasks.to('cpu').numpy()
                    
                    if F1_METHOD == 'first_token':

                        logits = [list(p) for p in np.argmax(logits, axis=2)]
                        actual_logits = []
                        actual_label_ids = []
                        for l,lid,m in zip(logits, label_ids, mymasks):
                            actual_logits.append([i for i,j in zip(l,m) if j==1])
                            actual_label_ids.append([i for i,j in zip(lid, m) if j==1])

                        logits = actual_logits
                        label_ids = actual_label_ids

                        predictions.append(logits)
                        true_labels.append(label_ids)
                    
                    elif F1_METHOD == 'average':

                        logits = [list(p) for p in logits]
                        
                        actual_logits = []
                        actual_label_ids = []
                        
                        for l,lid,m in zip(logits, label_ids, mymasks):
                            
                            actual_label_ids.append([i for i,j in zip(lid, m) if j==1])
                            curr_preds = []
                            my_logits = []
                            in_split = 0
                            for i,j in zip(l,m):
                                if j==1:
                                    if in_split == 1:
                                        if len(my_logits)>0:
                                            curr_preds.append(my_logits[-1])
                                        mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                                        if len(my_logits)>0:
                                            my_logits[-1] = mode_pred
                                        else:
                                            my_logits.append(mode_pred)
                                        curr_preds = []
                                        in_split = 0
                                    my_logits.append(np.argmax(i))
                                if j==0:
                                    curr_preds.append(i)
                                    in_split = 1
                            if in_split == 1:
                                if len(my_logits)>0:
                                    curr_preds.append(my_logits[-1])
                                mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                                if len(my_logits)>0:
                                    my_logits[-1] = mode_pred
                                else:
                                    my_logits.append(mode_pred)
                            actual_logits.append(my_logits)
                            
                        logits = actual_logits
                        label_ids = actual_label_ids
                        
                        predictions.append(logits)
                        true_labels.append(label_ids)
                    
                    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                    tmp_eval_positive_cue_accuracy = flat_accuracy_positive_cues(logits, label_ids)
                    eval_loss += tmp_eval_loss.mean().item()
                    valid_loss.append(tmp_eval_loss.mean().item())
                    eval_accuracy += tmp_eval_accuracy
                    if tmp_eval_positive_cue_accuracy!=None:
                        eval_positive_cue_accuracy+=tmp_eval_positive_cue_accuracy
                        steps_positive_cue_accuracy+=1
                    nb_eval_examples += b_input_ids.size(0)
                    nb_eval_steps += 1
                eval_loss = eval_loss/nb_eval_steps
                
            print("Validation loss: {}".format(eval_loss))
            print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
            print("Validation Accuracy for Positive Cues: {}".format(eval_positive_cue_accuracy/steps_positive_cue_accuracy))
            labels_flat = [l_ii for l in true_labels for l_i in l for l_ii in l_i]
            pred_flat = [p_ii for p in predictions for p_i in p for p_ii in p_i]
            pred_flat = [p for p,l in zip(pred_flat, labels_flat) if l!=4]
            labels_flat = [l for l in labels_flat if l!=4]
            report_per_class_accuracy(labels_flat, pred_flat)
            print(classification_report(labels_flat, pred_flat))
            print("F1-Score Overall: {}".format(f1_score(labels_flat,pred_flat, average='weighted')))
            p,r,f1 = f1_cues(labels_flat, pred_flat)
            if f1>return_dict['Best F1']:
                return_dict['Best F1'] = f1
                return_dict['Best Precision'] = p
                return_dict['Best Recall'] = r
            early_stopping(f1, self.model)
        
            if early_stopping.early_stop:
                print("Early stopping")
                break

            labels_flat = [int(i!=3) for i in labels_flat]
            pred_flat = [int(i!=3) for i in pred_flat]
            print("F1-Score Cue_No Cue: {}".format(f1_score(labels_flat,pred_flat, average='weighted')))
            
        self.model.load_state_dict(torch.load('checkpoint.pt'))
        plt.xlabel("Iteration")
        plt.ylabel("Train Loss")
        plt.plot([i for i in range(len(train_loss))], train_loss)
        plt.figure()
        plt.xlabel("Iteration")
        plt.ylabel("Validation Loss")
        plt.plot([i for i in range(len(valid_loss))], valid_loss)
        plt.savefig('loss_curve_cue_detection', dpi=300)   # save the figure to file
        return return_dict

    # @telegram_sender(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)
    def evaluate(self, test_dataloader, test_dl_name):
        return_dict = {"Task": f"{self.task} Cue Detection",
                       "Model": self.model_name,
                       "Train Dataset": self.train_dl_name,
                       "Test Dataset": test_dl_name,
                       "Precision": 0,
                       "Recall": 0,
                       "F1": 0}
        self.model.eval()
        eval_loss, eval_accuracy, eval_scope_accuracy, eval_positive_cue_accuracy = 0, 0, 0, 0
        nb_eval_steps, nb_eval_examples, steps_positive_cue_accuracy = 0, 0, 0
        predictions , true_labels, ip_mask = [], [], []
        loss_fn = CrossEntropyLoss(weight=torch.Tensor(self.class_weight).to(self.device))
        for batch in test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_mymasks = batch
            
            with torch.no_grad():
                logits = self.model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels = b_labels.view(-1)[active_loss]
                tmp_eval_loss = loss_fn(active_logits, active_labels)
                logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            mymasks = b_mymasks.to('cpu').numpy()

            if F1_METHOD == 'first_token':

                logits = [list(p) for p in np.argmax(logits, axis=2)]
                actual_logits = []
                actual_label_ids = []
                for l,lid,m in zip(logits, label_ids, mymasks):
                    actual_logits.append([i for i,j in zip(l,m) if j==1])
                    actual_label_ids.append([i for i,j in zip(lid, m) if j==1])

                logits = actual_logits
                label_ids = actual_label_ids

                predictions.append(logits)
                true_labels.append(label_ids)

            elif F1_METHOD == 'average':
                logits = [list(p) for p in logits]
                    
                actual_logits = []
                actual_label_ids = []
                for l,lid,m in zip(logits, label_ids, mymasks):
                        
                    actual_label_ids.append([i for i,j in zip(lid, m) if j==1])
                    my_logits = []
                    curr_preds = []
                    in_split = 0
                    for i,j in zip(l,m):
                        if j==1:
                            if in_split == 1:
                                if len(my_logits)>0:
                                    curr_preds.append(my_logits[-1])
                                mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                                if len(my_logits)>0:
                                    my_logits[-1] = mode_pred
                                else:
                                    my_logits.append(mode_pred)
                                curr_preds = []
                                in_split = 0
                            my_logits.append(np.argmax(i))
                        if j==0:
                            curr_preds.append(i)
                            in_split = 1
                    if in_split == 1:
                        if len(my_logits)>0:
                            curr_preds.append(my_logits[-1])
                        mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                        if len(my_logits)>0:
                            my_logits[-1] = mode_pred
                        else:
                            my_logits.append(mode_pred)
                    actual_logits.append(my_logits)

                logits = actual_logits
                label_ids = actual_label_ids
                
                predictions.append(logits)
                true_labels.append(label_ids)    
                
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            tmp_eval_positive_cue_accuracy = flat_accuracy_positive_cues(logits, label_ids)
        
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            if tmp_eval_positive_cue_accuracy != None:
                eval_positive_cue_accuracy += tmp_eval_positive_cue_accuracy
                steps_positive_cue_accuracy+=1
            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        print("Validation Accuracy for Positive Cues: {}".format(eval_positive_cue_accuracy/steps_positive_cue_accuracy))
        labels_flat = [l_ii for l in true_labels for l_i in l for l_ii in l_i]
        pred_flat = [p_ii for p in predictions for p_i in p for p_ii in p_i]
        pred_flat = [p for p,l in zip(pred_flat, labels_flat) if l!=4]
        labels_flat = [l for l in labels_flat if l!=4]
        report_per_class_accuracy(labels_flat, pred_flat)
        print(classification_report(labels_flat, pred_flat))
        print("F1-Score: {}".format(f1_score(labels_flat,pred_flat,average='weighted')))
        p,r,f1 = f1_cues(labels_flat, pred_flat)
        return_dict['Precision'] = p
        return_dict['Recall'] = r
        return_dict['F1'] = f1
        labels_flat = [int(i!=3) for i in labels_flat]
        pred_flat = [int(i!=3) for i in pred_flat]
        print("F1-Score Cue_No Cue: {}".format(f1_score(labels_flat,pred_flat,average='weighted')))
        return return_dict

    def predict(self, dataloader):
        self.model.eval()
        predictions, ip_mask = [], []
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_mymasks = batch

            with torch.no_grad():
                logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
            logits = logits.detach().cpu().numpy()
            mymasks = b_mymasks.to('cpu').numpy()
            #predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            if F1_METHOD == 'first_token':

                logits = [list(p) for p in np.argmax(logits, axis=2)]
                actual_logits = []
                for l,m in zip(logits, mymasks):
                    actual_logits.append([i for i,j in zip(l,m) if j==1])
                
                predictions.append(actual_logits)

            elif F1_METHOD == 'average':
                logits = [list(p) for p in logits]
                    
                actual_logits = []
                actual_label_ids = []
                for l,m in zip(logits, mymasks):
                    my_logits = []
                    curr_preds = []
                    in_split = 0
                    for i,j in zip(l,m):
                        if j==1:
                            if in_split == 1:
                                if len(my_logits)>0:
                                    curr_preds.append(my_logits[-1])
                                mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                                if len(my_logits)>0:
                                    my_logits[-1] = mode_pred
                                else:
                                    my_logits.append(mode_pred)
                                curr_preds = []
                                in_split = 0
                            my_logits.append(np.argmax(i))
                        if j==0:
                            curr_preds.append(i)
                            in_split = 1
                    if in_split == 1:
                        if len(my_logits)>0:
                            curr_preds.append(my_logits[-1])
                        mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                        if len(my_logits)>0:
                            my_logits[-1] = mode_pred
                        else:
                            my_logits.append(mode_pred)
                    actual_logits.append(my_logits)
                predictions.append(actual_logits)
                
        return predictions

class ScopeModel:
    def __init__(self, full_finetuning = True, train = False, pretrained_model_path = 'Scope_Resolution_Augment.pickle', device = 'cuda', learning_rate = 3e-5):
        self.model_name = SCOPE_MODEL
        self.task = TASK
        self.num_labels = 2
        if 'xlnet' in SCOPE_MODEL:
            self.model = XLNetForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels, cache_dir = 'xlnet-base-cased-model')
        elif 'roberta' in SCOPE_MODEL:
            self.model = RobertaForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels, cache_dir = 'roberta-base-model')
        elif 'bert' in SCOPE_MODEL:
            self.model = BertForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels, cache_dir = 'bert_base_uncased_model')
        else:
            raise ValueError("Supported model types are: xlnet, roberta, bert")
        if train == False:
            # self.model = torch.load(pretrained_model_path)
            self.model.load_state_dict(torch.load(pretrained_model_path))
            print('loaded ScopeModel')
        self.device = torch.device(device)
        if device=='cuda':
            self.model.cuda()
        else:
            self.model.cpu()

        if full_finetuning:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters()) 
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        self.optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate)

    # @telegram_sender(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)    
    def train(self, train_dataloader, valid_dataloaders, train_dl_name, val_dl_name, epochs = 5, max_grad_norm = 1.0, patience = 3):
        self.train_dl_name = train_dl_name
        return_dict = {"Task": f"{self.task} Scope Resolution",
                       "Model": self.model_name,
                       "Train Dataset": train_dl_name,
                       "Val Dataset": val_dl_name,
                       "Best Precision": 0,
                       "Best Recall": 0,
                       "Best F1": 0}
        train_loss = []
        valid_loss = []
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        loss_fn = CrossEntropyLoss()
        for _ in tqdm(range(epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_mymasks = batch
                logits = self.model(b_input_ids, token_type_ids=None,
                             attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss] #2 is num_labels
                active_labels = b_labels.view(-1)[active_loss]
                loss = loss_fn(active_logits, active_labels)
                loss.backward()
                tr_loss += loss.item()
                train_loss.append(loss.item())
                if step%100 == 0:
                    print(f"Batch {step}, loss {loss.item()}")
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad()
            print("Train loss: {}".format(tr_loss/nb_tr_steps))
            
            self.model.eval()
            
            eval_loss, eval_accuracy, eval_scope_accuracy = 0, 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            predictions , true_labels, ip_mask = [], [], []
            loss_fn = CrossEntropyLoss()
            for valid_dataloader in valid_dataloaders:
                for batch in valid_dataloader:
                    batch = tuple(t.to(self.device) for t in batch)
                    b_input_ids, b_input_mask, b_labels, b_mymasks = batch

                    with torch.no_grad():
                        logits = self.model(b_input_ids, token_type_ids=None,
                                      attention_mask=b_input_mask)[0]
                        active_loss = b_input_mask.view(-1) == 1
                        active_logits = logits.view(-1, self.num_labels)[active_loss]
                        active_labels = b_labels.view(-1)[active_loss]
                        tmp_eval_loss = loss_fn(active_logits, active_labels)
                        
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()
                    b_input_ids = b_input_ids.to('cpu').numpy()

                    mymasks = b_mymasks.to('cpu').numpy()
                        
                    if F1_METHOD == 'first_token':

                        logits = [list(p) for p in np.argmax(logits, axis=2)]
                        actual_logits = []
                        actual_label_ids = []
                        for l,lid,m in zip(logits, label_ids, mymasks):
                            actual_logits.append([i for i,j in zip(l,m) if j==1])
                            actual_label_ids.append([i for i,j in zip(lid, m) if j==1])

                        logits = actual_logits
                        label_ids = actual_label_ids

                        predictions.append(logits)
                        true_labels.append(label_ids)
                    elif F1_METHOD == 'average':
                      
                        logits = [list(p) for p in logits]
                    
                        actual_logits = []
                        actual_label_ids = []
                        
                        for l,lid,m,b_ii in zip(logits, label_ids, mymasks, b_input_ids):
                                
                            actual_label_ids.append([i for i,j in zip(lid, m) if j==1])
                            my_logits = []
                            curr_preds = []
                            in_split = 0
                            for i,j,k in zip(l,m, b_ii):
                                '''if k == 0:
                                    break'''
                                if j==1:
                                    if in_split == 1:
                                        if len(my_logits)>0:
                                            curr_preds.append(my_logits[-1])
                                        mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                                        if len(my_logits)>0:
                                            my_logits[-1] = mode_pred
                                        else:
                                            my_logits.append(mode_pred)
                                        curr_preds = []
                                        in_split = 0
                                    my_logits.append(np.argmax(i))
                                if j==0:
                                    curr_preds.append(i)
                                    in_split = 1
                            if in_split == 1:
                                if len(my_logits)>0:
                                    curr_preds.append(my_logits[-1])
                                mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                                if len(my_logits)>0:
                                    my_logits[-1] = mode_pred
                                else:
                                    my_logits.append(mode_pred)
                            actual_logits.append(my_logits)
                            
                        predictions.append(actual_logits)
                        true_labels.append(actual_label_ids)    
                        
                    tmp_eval_accuracy = flat_accuracy(actual_logits, actual_label_ids)
                    tmp_eval_scope_accuracy = scope_accuracy(actual_logits, actual_label_ids)
                    eval_scope_accuracy += tmp_eval_scope_accuracy
                    valid_loss.append(tmp_eval_loss.mean().item())

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += len(b_input_ids)
                    nb_eval_steps += 1
                eval_loss = eval_loss/nb_eval_steps
            print("Validation loss: {}".format(eval_loss))
            print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
            print("Validation Accuracy Scope Level: {}".format(eval_scope_accuracy/nb_eval_steps))
            f1_scope([j for i in true_labels for j in i], [j for i in predictions for j in i], level='scope')
            labels_flat = [l_ii for l in true_labels for l_i in l for l_ii in l_i]
            pred_flat = [p_ii for p in predictions for p_i in p for p_ii in p_i]
            classification_dict = classification_report(labels_flat, pred_flat, output_dict= True)
            p = classification_dict["1"]["precision"]
            r = classification_dict["1"]["recall"]
            f1 = classification_dict["1"]["f1-score"]
            if f1>return_dict['Best F1']:
                return_dict['Best F1'] = f1
                return_dict['Best Precision'] = p
                return_dict['Best Recall'] = r
            print("F1-Score Token: {}".format(f1))
            print(classification_report(labels_flat, pred_flat))
            early_stopping(f1, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        self.model.load_state_dict(torch.load('checkpoint.pt'))
        plt.xlabel("Iteration")
        plt.ylabel("Train Loss")
        plt.plot([i for i in range(len(train_loss))], train_loss)
        plt.figure()
        plt.xlabel("Iteration")
        plt.ylabel("Validation Loss")
        plt.plot([i for i in range(len(valid_loss))], valid_loss)
        plt.savefig('loss_curve_scope_resolution', dpi=300)   # save the figure to file
        return return_dict

    # @telegram_sender(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)
    def evaluate(self, test_dataloader, test_dl_name = "SFU"):
        return_dict = {"Task": f"{self.task} Scope Resolution",
                       "Model": self.model_name,
                       "Train Dataset": self.train_dl_name,
                       "Test Dataset": test_dl_name,
                       "Precision": 0,
                       "Recall": 0,
                       "F1": 0}
        self.model.eval()
        eval_loss, eval_accuracy, eval_scope_accuracy = 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels, ip_mask = [], [], []
        loss_fn = CrossEntropyLoss()
        for batch in test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_mymasks = batch

            with torch.no_grad():
                logits = self.model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels = b_labels.view(-1)[active_loss]
                tmp_eval_loss = loss_fn(active_logits, active_labels)
                
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            b_input_ids = b_input_ids.to('cpu').numpy()
            
            mymasks = b_mymasks.to('cpu').numpy()

            if F1_METHOD == 'first_token':

                logits = [list(p) for p in np.argmax(logits, axis=2)]
                actual_logits = []
                actual_label_ids = []
                for l,lid,m in zip(logits, label_ids, mymasks):
                    actual_logits.append([i for i,j in zip(l,m) if j==1])
                    actual_label_ids.append([i for i,j in zip(lid, m) if j==1])

                logits = actual_logits
                label_ids = actual_label_ids

                predictions.append(logits)
                true_labels.append(label_ids)

            elif F1_METHOD == 'average':
                
                logits = [list(p) for p in logits]
                
                actual_logits = []
                actual_label_ids = []
                
                for l,lid,m,b_ii in zip(logits, label_ids, mymasks, b_input_ids):
                        
                    actual_label_ids.append([i for i,j in zip(lid, m) if j==1])
                    my_logits = []
                    curr_preds = []
                    in_split = 0
                    for i,j,k in zip(l,m,b_ii):
                        '''if k == 0:
                            break'''
                        if j==1:
                            if in_split == 1:
                                if len(my_logits)>0:
                                    curr_preds.append(my_logits[-1])
                                mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                                if len(my_logits)>0:
                                    my_logits[-1] = mode_pred
                                else:
                                    my_logits.append(mode_pred)
                                curr_preds = []
                                in_split = 0
                            my_logits.append(np.argmax(i))
                        if j==0:
                            curr_preds.append(i)
                            in_split = 1
                    if in_split == 1:
                        if len(my_logits)>0:
                            curr_preds.append(my_logits[-1])
                        mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                        if len(my_logits)>0:
                            my_logits[-1] = mode_pred
                        else:
                            my_logits.append(mode_pred)
                    actual_logits.append(my_logits)
                    
                predictions.append(actual_logits)
                true_labels.append(actual_label_ids)

            tmp_eval_accuracy = flat_accuracy(actual_logits, actual_label_ids)
            tmp_eval_scope_accuracy = scope_accuracy(actual_logits, actual_label_ids)
            eval_scope_accuracy += tmp_eval_scope_accuracy

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += len(b_input_ids)
            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        print("Validation Accuracy Scope Level: {}".format(eval_scope_accuracy/nb_eval_steps))
        f1_scope([j for i in true_labels for j in i], [j for i in predictions for j in i], level='scope')
        labels_flat = [l_ii for l in true_labels for l_i in l for l_ii in l_i]
        pred_flat = [p_ii for p in predictions for p_i in p for p_ii in p_i]
        classification_dict = classification_report(labels_flat, pred_flat, output_dict= True)
        p = classification_dict["1"]["precision"]
        r = classification_dict["1"]["recall"]
        f1 = classification_dict["1"]["f1-score"]
        return_dict['Precision'] = p
        return_dict['Recall'] = r
        return_dict['F1'] = f1
        print("Classification Report:")
        print(classification_report(labels_flat, pred_flat))
        return return_dict

    def predict(self, dataloader):
        self.model.eval()
        predictions, ip_mask = [], []
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_mymasks = batch

            with torch.no_grad():
                logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
            logits = logits.detach().cpu().numpy()
            mymasks = b_mymasks.to('cpu').numpy()

            if F1_METHOD == 'first_token':

                logits = [list(p) for p in np.argmax(logits, axis=2)]
                actual_logits = []
                for l,lid,m in zip(logits, label_ids, mymasks):
                    actual_logits.append([i for i,j in zip(l,m) if j==1])
                
                logits = actual_logits
                label_ids = actual_label_ids

                predictions.append(logits)
                true_labels.append(label_ids)

            elif F1_METHOD == 'average':
                
                logits = [list(p) for p in logits]
                
                actual_logits = []
                
                for l,m in zip(logits, mymasks):
                        
                    my_logits = []
                    curr_preds = []
                    in_split = 0
                    for i,j in zip(l,m):
                        
                        if j==1:
                            if in_split == 1:
                                if len(my_logits)>0:
                                    curr_preds.append(my_logits[-1])
                                mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                                if len(my_logits)>0:
                                    my_logits[-1] = mode_pred
                                else:
                                    my_logits.append(mode_pred)
                                curr_preds = []
                                in_split = 0
                            my_logits.append(np.argmax(i))
                        if j==0:
                            curr_preds.append(i)
                            in_split = 1
                    if in_split == 1:
                        if len(my_logits)>0:
                            curr_preds.append(my_logits[-1])
                        mode_pred = np.argmax(np.average(np.array(curr_preds), axis=0), axis=0)
                        if len(my_logits)>0:
                            my_logits[-1] = mode_pred
                        else:
                            my_logits.append(mode_pred)
                    actual_logits.append(my_logits)
                    
                predictions.append(actual_logits)
        return predictions

# mydata = CustomData(["Hi there this might be good"])
# dl = mydata.get_cue_dataloader()

# mydata = CustomData(["Hi there this might be good"], cues = [[3,3,3,1,3,3]])
# dl = mydata.get_scope_dataloader()
# model.predict(dl)

class NegaionDetectionAPI:
    def __init__(self, base_dir='.', functions=['cue_detection', 'scope_resolution']):
        self.functions = functions
        if torch.cuda.device_count() > 1:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.manual_negation_cue = ['not']
        # build models
        if 'cue_detection' in self.functions:
            self.model_cue = CueModel(full_finetuning=True, train=False, 
                                    pretrained_model_path=os.path.join(base_dir, 'models/cue_detection_checkpoint.pt'), 
                                    device=self.device,
                                    learning_rate = INITIAL_LEARNING_RATE)
        if 'scope_resolution' in self.functions:
            self.model_scope = ScopeModel(full_finetuning=True, train=False, 
                                    pretrained_model_path=os.path.join(base_dir, 'models/scope_resolution_checkpoint.pt'), 
                                    device=self.device,
                                    learning_rate = INITIAL_LEARNING_RATE)
        print('-> NegationDetection model loaded.')

    def pred(self, input_text_list):
        if 'cue_detection' in self.functions:
            # negation cue detection
            mydata = CustomData(input_text_list)
            dl = mydata.get_cue_dataloader()
            cues_output = self.model_cue.predict(dl)[0]
            print(cues_output)
            # forcely mark certain negation keywords defined in self.manual_negation_cue as normal cue
            for i, text in enumerate(input_text_list):
                text_list = text.split(' ')
                for neg_word in self.manual_negation_cue:
                    if neg_word in text_list:
                        neg_word_idx = text_list.index(neg_word)
                        cues_output[i][neg_word_idx] = 1
            print(cues_output)

        if 'scope_resolution' in self.functions:
            # negation scope resolution
            mydata = CustomData(input_text_list, cues=cues_output)
            dl = mydata.get_scope_dataloader()
            scope_output = self.model_scope.predict(dl)[0]
            print(scope_output)
        return cues_output, scope_output

if __name__ == '__main__':
    # api = NegaionDetectionAPI()
    # api.pred(["Hi there this might be good", "I'm not going to play the game",
    #         "The United States is not considering sending troops to Mozambique to combat the terrorist threat in the northern province of Cabo Delgado, but it is willing to boost \"civilian counter-terrorism capabilities\", said the US Coordinator for Counterterrorism, Nathan Sales, on Tuesday 8 December."])

    # bioscope_full_papers_data = Data('/content/gdrive/My Drive/path_to_file', dataset_name='bioscope')
    sfu_data = Data('SFU_Review_Corpus_Negation_Speculation', dataset_name='sfu')
    # bioscope_abstracts_data = Data('/content/gdrive/My Drive/path_to_file', dataset_name='bioscope')
    if TASK == 'negation':
        sherlock_train_data = Data('/content/gdrive/My Drive/path_to_file', dataset_name='starsem')
        sherlock_dev_data = Data('/content/gdrive/My Drive/path_to_file', dataset_name='starsem')
        sherlock_test_gold_cardboard_data = Data('/content/gdrive/My Drive/path_to_file', dataset_name='starsem')
        sherlock_test_gold_circle_data = Data('/content/gdrive/My Drive/path_to_file', dataset_name='starsem')

    for run_num in range(NUM_RUNS):
        first_dataset = None
        other_datasets = []
        if 'sfu' in TRAIN_DATASETS:
            first_dataset = sfu_data
        if 'bioscope_full_papers' in TRAIN_DATASETS:
            if first_dataset == None:
                first_dataset = bioscope_full_papers_data
            else:
                other_datasets.append(bioscope_full_papers_data)
        if 'bioscope_abstracts' in TRAIN_DATASETS:
            if first_dataset == None:
                first_dataset = bioscope_abstracts_data
            else:
                other_datasets.append(bioscope_abstracts_data)
        if 'sherlock' in TRAIN_DATASETS:
            if first_dataset == None:
                first_dataset = sherlock_train_data
            else:
                other_datasets.append(sherlock_train_data)

        if SUBTASK == 'cue_detection':
            train_dl, val_dls, test_dls = first_dataset.get_cue_dataloader(other_datasets = other_datasets)
            if 'sherlock' in TRAIN_DATASETS:
                val_dls = val_dls[:-1]
                append_dl, _, _ = sherlock_dev_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                val_dls.append(append_dl)
                test_dls = test_dls[:-1]
                sherlock_dl, _, _ = sherlock_test_gold_cardboard_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001, other_datasets = [sherlock_test_gold_circle_data])
                test_dls.append(sherlock_dl)

            test_dataloaders = {}
            idx = 0
            if 'sfu' in TRAIN_DATASETS:
                if 'sfu' in TEST_DATASETS:
                    test_dataloaders['sfu'] = test_dls[idx]
                idx+=1
            elif 'sfu' in TEST_DATASETS:
                sfu_dl, _, _ = sfu_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dataloaders['sfu'] = sfu_dl
            if 'bioscope_full_papers' in TRAIN_DATASETS:
                if 'bioscope_full_papers' in TEST_DATASETS:
                    test_dataloaders['bioscope_full_papers'] = test_dls[idx]
                idx+=1
            elif 'bioscope_full_papers' in TEST_DATASETS:
                bioscope_full_papers_dl, _, _ = bioscope_full_papers_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dataloaders['bioscope_full_papers'] = bioscope_full_papers_dl
            if 'bioscope_abstracts' in TRAIN_DATASETS:
                if 'bioscope_abstracts' in TEST_DATASETS:
                    test_dataloaders['bi oscope_abstracts'] = test_dls[idx]
                idx+=1
            elif 'bioscope_abstracts' in TEST_DATASETS:
                bioscope_abstracts_dl, _, _ = bioscope_abstracts_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dataloaders['bioscope_abstracts'] = bioscope_abstracts_dl
            if 'sherlock' in TRAIN_DATASETS:
                if 'sherlock' in TEST_DATASETS:
                    test_dataloaders['sherlock'] = test_dls[idx]
                idx+=1
            elif 'sherlock' in TEST_DATASETS:
                sherlock_dl, _, _ = sherlock_test_gold_cardboard_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001, other_datasets = [sherlock_test_gold_circle_data])
                test_dataloaders['sherlock'] = sherlock_dl

                
        elif SUBTASK == 'scope_resolution':
            train_dl, val_dls, test_dls = first_dataset.get_scope_dataloader(other_datasets = other_datasets)
            if 'sherlock' in TRAIN_DATASETS:
                val_dls = val_dls[:-1]
                append_dl, _, _ = sherlock_dev_data.get_scope_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                val_dls.append(append_dl)
                test_dls = test_dls[:-1]
                sherlock_dl, _, _ = sherlock_test_gold_cardboard_data.get_scope_dataloader(test_size = 0.00000001, val_size = 0.00000001, other_datasets = [sherlock_test_gold_circle_data])
                test_dls.append(sherlock_dl)

            test_dataloaders = {}
            idx = 0
            if 'sfu' in TRAIN_DATASETS:
                if 'sfu' in TEST_DATASETS:
                    test_dataloaders['sfu'] = test_dls[idx]
                idx+=1
            elif 'sfu' in TEST_DATASETS:
                sfu_dl, _, _ = sfu_data.get_scope_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dataloaders['sfu'] = sfu_dl
            if 'bioscope_full_papers' in TRAIN_DATASETS:
                if 'bioscope_full_papers' in TEST_DATASETS:
                    test_dataloaders['bioscope_full_papers'] = test_dls[idx]
                idx+=1
            elif 'bioscope_full_papers' in TEST_DATASETS:
                bioscope_full_papers_dl, _, _ = bioscope_full_papers_data.get_scope_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dataloaders['bioscope_full_papers'] = bioscope_full_papers_dl
            if 'bioscope_abstracts' in TRAIN_DATASETS:
                if 'bioscope_abstracts' in TEST_DATASETS:
                    test_dataloaders['bioscope_abstracts'] = test_dls[idx]
                idx+=1
            elif 'bioscope_abstracts' in TEST_DATASETS:
                bioscope_abstracts_dl, _, _ = bioscope_abstracts_data.get_scope_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dataloaders['bioscope_abstracts'] = bioscope_abstracts_dl
            if 'sherlock' in TRAIN_DATASETS:
                if 'sherlock' in TEST_DATASETS:
                    test_dataloaders['sherlock'] = test_dls[idx]
                idx+=1
            elif 'sherlock' in TEST_DATASETS:
                sherlock_dl, _, _ = sherlock_test_gold_cardboard_data.get_scope_dataloader(test_size = 0.00000001, val_size = 0.00000001, other_datasets = [sherlock_test_gold_circle_data])
                test_dataloaders['sherlock'] = sherlock_dl
        else:
            raise ValueError("Unsupported subtask. Supported values are: cue_detection, scope_resolution")


        if SUBTASK == 'cue_detection':
            model = CueModel(full_finetuning=True, train=True, learning_rate = INITIAL_LEARNING_RATE)
        elif SUBTASK == 'scope_resolution':
            model = ScopeModel(full_finetuning=True, train=True, learning_rate = INITIAL_LEARNING_RATE)
        else:
            raise ValueError("Unsupported subtask. Supported values are: cue_detection, scope_resolution")
        
        
        model.train(train_dl, val_dls, epochs=EPOCHS, patience=PATIENCE, train_dl_name = ','.join(TRAIN_DATASETS), val_dl_name = ','.join(TRAIN_DATASETS))

        for k in test_dataloaders.keys():
            print(f"Evaluate on {k}:")
            model.evaluate(test_dataloaders[k], test_dl_name = k)

        print(f"\n\n************ RUN {run_num+1} DONE! **************\n\n")