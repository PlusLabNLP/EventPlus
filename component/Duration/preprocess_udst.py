import pandas as pd
from decomp import UDSCorpus
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class UDS_T_Dataset(Dataset):
    def __init__(self, tsv_filename, processed_jsonl_filename=None, dataset_split="train", get_pairs=True, max_input_size=512):
        
        # if True, return sent 1 and sent 2 together (__getitem__() index: simply dataframe index);
        # else, dataset size = 2 * dataframe size (__getitem__() index: dataframe index * 2 (+ 1) )
        self.get_pairs = get_pairs 

        self.split = dataset_split

        self.max_input_size = max_input_size

        if processed_jsonl_filename is None:

            UDS_T_dataframe = pd.read_csv(tsv_filename, sep='\t')
            uds = UDSCorpus()

            # check assumptioin: all from ewt (uds['ewt-<split>-<id>'])
            lan1 = (UDS_T_dataframe['Sentence1.ID'].apply(lambda x: x.split(".conllu ")[0].split("-")[0]) == "en")
            lan2 = (UDS_T_dataframe['Sentence2.ID'].apply(lambda x: x.split(".conllu ")[0].split("-")[0]) == "en")
            where1 = (UDS_T_dataframe['Sentence1.ID'].apply(lambda x: x.split(".conllu ")[0].split("-")[1]) == "ud")
            where2 = (UDS_T_dataframe['Sentence2.ID'].apply(lambda x: x.split(".conllu ")[0].split("-")[1]) == "ud")
            assert(lan1.all())
            assert(lan2.all())
            assert(where1.all())
            assert(where2.all())

            # get sentence pair's split and id: sent1/2_split, sent1/2_id
            UDS_T_dataframe['Sent1_Split'] = UDS_T_dataframe['Sentence1.ID'].apply(lambda x: x.split(".")[0].split("-")[-1])
            UDS_T_dataframe['Sent1_Id'] = UDS_T_dataframe['Sentence1.ID'].apply(lambda x: x.split(".conllu ")[1])

            UDS_T_dataframe['Sent2_Split'] = UDS_T_dataframe['Sentence2.ID'].apply(lambda x: x.split(".")[0].split("-")[-1])
            UDS_T_dataframe['Sent2_Id'] = UDS_T_dataframe['Sentence2.ID'].apply(lambda x: x.split(".conllu ")[1])
            
            # check assumption: same split
            assert((UDS_T_dataframe['Sent1_Split'] == UDS_T_dataframe['Sent2_Split']).all())

            # get tokens for sentence pair
            UDS_T_dataframe['Sent1_str_tokens'] = [uds["ewt-{}-{}".format(curr_split, curr_id)].sentence.split() for curr_split, curr_id in zip(UDS_T_dataframe['Sent1_Split'], UDS_T_dataframe['Sent1_Id'])]
            UDS_T_dataframe['Sent2_str_tokens'] = [uds["ewt-{}-{}".format(curr_split, curr_id)].sentence.split() for curr_split, curr_id in zip(UDS_T_dataframe['Sent2_Split'], UDS_T_dataframe['Sent2_Id'])]
            

            # get root and span int index lists
            UDS_T_dataframe['Sent1_root_idxs'] = UDS_T_dataframe['Pred1.Token'].apply(lambda x: [int(x)])
            UDS_T_dataframe['Sent2_root_idxs'] = UDS_T_dataframe['Pred2.Token'].apply(lambda x: [int(x)])
            
            UDS_T_dataframe['Sent1_span_idxs'] = UDS_T_dataframe['Pred1.Span'].apply(lambda x: [int(y) for y in x.split("_")])
            UDS_T_dataframe['Sent2_span_idxs'] = UDS_T_dataframe['Pred2.Span'].apply(lambda x: [int(y) for y in x.split("_")])

            """ 
            GOAL of self.get_token_ids_masks():

                Token ids:

                    {Token ids for words before Span 1} {Span 1 token ids} {Token ids after Span 1 of sent 1} <SEP> 
                        {Token ids for words before Span 2} {Span 2 token ids} {Token ids after Span 2 of sent 2}
                        rest: pad tokens (space saving: pad in getitem)

                span only masks:
                    {0, 0, 0, ...                    0} {1, 1 ...       1} {0, 0, 0, ....                  0} <0>
                        {0, 0, 0, ...                    0} {1, 1 ...       1} {0, 0, 0, ....                  0}
                        rest: {0, 0, 0, ... } (space saving: pad in getitem)

            Then we need: starting and end points of each span in the sentences' token id list.

            Similarly for root
                
            """
            # tokenize
            # sentence1_token_ids (int)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

            UDS_T_dataframe = UDS_T_dataframe.apply(self.get_token_ids_masks, axis=1)

            """
            columns selected:
                
                Sent1:
                    Sent1_Split,
                    Sent1_Id,
                    X:  Sent1_bert_token_ids, 
                        Sent1_idx_offset (always 0),
                        Sent1_span_bert_start_idx, Sent1_span_bert_end_idx, Sent1_span_only_mask, 
                        Sent1_root_bert_start_idx, Sent1_root_bert_end_idx, Sent1_root_only_mask
                    Y:  Pred1.Duration,
                        Pred1.Duration.Confidence,
                        Pred1.Beg, Pred1.End,
                        Relation.Confidence (shared)
                
                Sent2:
                    Sent2_Split,
                    Sent2_Id,
                    X:  Sent2_bert_token_ids, 
                        Sent2_idx_offset (== len(Sent1) + 1),
                        Sent2_span_bert_start_idx, Sent2_span_bert_end_idx, Sent2_span_only_mask, 
                        Sent2_root_bert_start_idx, Sent2_root_bert_end_idx, Sent2_root_only_mask
                    Y:  Pred2.Duration,
                        Pred2.Duration.Confidence,
                        Pred2.Beg, Pred2.End,
                        Relation.Confidence (shared)
                
                Sent pair:
                    Token_ids,
                    Span1_only_masks,
                    Root1_only_masks,
                    Span2_only_masks,
                    Root2_only_masks

                For non-bert baselines:
                    'Sent1_root_idxs','Sent2_root_idxs','Sent1_span_idxs','Sent2_span_idxs'
            """
            selected_columns = [
                'Sent1_Split', 'Sent1_Id', 'Sent1_str_tokens', 'Sent2_Split', 'Sent2_Id', 'Sent2_str_tokens',
                
                'Sent1_bert_token_ids', 'Sent1_idx_offset',
                'Sent1_span_bert_start_idx', 'Sent1_span_bert_end_idx', 'Sent1_span_only_mask', 
                'Sent1_root_bert_start_idx', 'Sent1_root_bert_end_idx', 'Sent1_root_only_mask',
                
                'Sent2_bert_token_ids', 'Sent2_idx_offset',
                'Sent2_span_bert_start_idx', 'Sent2_span_bert_end_idx', 'Sent2_span_only_mask', 
                'Sent2_root_bert_start_idx', 'Sent2_root_bert_end_idx', 'Sent2_root_only_mask',
                
                'Pred1.Duration', 'Pred1.Duration.Confidence', 'Pred1.Beg', 'Pred1.End',
                'Pred2.Duration', 'Pred2.Duration.Confidence', 'Pred2.Beg', 'Pred2.End',
                'Relation.Confidence',
                
                'Token_ids', 'Span1_only_masks', 'Root1_only_masks', 'Span2_only_masks', 'Root2_only_masks',

                'Sent1_root_idxs','Sent2_root_idxs','Sent1_span_idxs','Sent2_span_idxs'
            ]

            # print(UDS_T_dataframe.columns)
            self.UDS_T_dataframe = UDS_T_dataframe[selected_columns]
            # save to json to reserve int list, read_csv does not read int list as int list (str instead)
            UDS_T_dataframe.to_json(tsv_filename.split(".tsv")[0] + "_preprocessed.jsonl", orient='records', lines=True)
            

        else:
            # simply load csv
            self.UDS_T_dataframe = pd.read_json(processed_jsonl_filename, orient='records', lines=True)

        #-----------------------------------------------------------------------
        
        # load the split ("train", "dev", "test")
        self.dataframe = self.UDS_T_dataframe[self.UDS_T_dataframe['Sent1_Split'] == self.split]

    def get_dur_label_loss_weight(self,):
        label_counts = [0] * 11
        train_df = self.UDS_T_dataframe[self.UDS_T_dataframe["Sent1_Split"] == 'train']
        for i in range(len(label_counts)):
            label_counts[i] = len(train_df[train_df['Pred1.Duration'] == i]) + \
                                    len(train_df[train_df['Pred2.Duration'] == i])
        print(label_counts)
        weights = [sum(label_counts) / label_count for label_count in label_counts]
        weights = [weight / min(weights) for weight in weights]
        print(weights)
        return torch.tensor(weights)

    def get_token_ids_masks(self, row):
        
        # sent1's token ids + span/root start/end indexes & masks
        sent1_span_start_idx = row['Sent1_span_idxs'][0]
        sent1_span_end_idx   = row['Sent1_span_idxs'][-1]
        
        # when: start_idx = 0 or end_idx = last ==> no before or no after, nothing to tokenize
        # FIX: assign [] to id_list
        before_span1_id_list = self.tokenizer.encode(row['Sent1_str_tokens'][:sent1_span_start_idx]) if sent1_span_start_idx != 0 else []
        span1_id_list        = self.tokenizer.encode(row['Sent1_str_tokens'][sent1_span_start_idx:sent1_span_end_idx+1])
        after_span1_id_list  = self.tokenizer.encode(row['Sent1_str_tokens'][(sent1_span_end_idx + 1):]) if sent1_span_end_idx + 1 < len(row['Sent1_str_tokens']) else []
        
        row['Sent1_bert_token_ids'] = before_span1_id_list + span1_id_list + after_span1_id_list
        
        row['Sent1_span_bert_start_idx'] = len(before_span1_id_list)
        row['Sent1_span_bert_end_idx']   = row['Sent1_span_bert_start_idx'] + len(span1_id_list)
        
        row['Sent1_span_only_mask']      = [0] * len(before_span1_id_list) + [1] * len(span1_id_list) + [0] * len(after_span1_id_list)
        
        sent1_root_start_idx = row['Sent1_root_idxs'][0]
        sent1_root_end_idx   = row['Sent1_root_idxs'][-1]
        
        before_root1_id_list = self.tokenizer.encode(row['Sent1_str_tokens'][:sent1_root_start_idx]) if sent1_root_start_idx != 0 else []
        root1_id_list        = self.tokenizer.encode(row['Sent1_str_tokens'][sent1_root_start_idx:sent1_root_end_idx+1])
        
        row['Sent1_root_bert_start_idx'] = len(before_root1_id_list)
        row['Sent1_root_bert_end_idx']   = row['Sent1_root_bert_start_idx'] + len(root1_id_list)
        
        row['Sent1_root_only_mask']      = [0] * len(before_root1_id_list) + [1] * len(root1_id_list) + [0] * (len(row['Sent1_bert_token_ids']) - len(before_root1_id_list) - len(root1_id_list))
        

        # sent2's token ids + span/root start/end indexes & masks

        sent2_span_start_idx = row['Sent2_span_idxs'][0]
        sent2_span_end_idx   = row['Sent2_span_idxs'][-1]
        
        before_span2_id_list = self.tokenizer.encode(row['Sent2_str_tokens'][:sent2_span_start_idx]) if sent2_span_start_idx != 0 else []
        span2_id_list        = self.tokenizer.encode(row['Sent2_str_tokens'][sent2_span_start_idx:sent2_span_end_idx+1])
        after_span2_id_list  = self.tokenizer.encode(row['Sent2_str_tokens'][(sent2_span_end_idx + 1):]) if sent2_span_end_idx + 1 < len(row['Sent2_str_tokens']) else []
        
        row['Sent2_bert_token_ids'] = before_span2_id_list + span2_id_list + after_span2_id_list
        
        row['Sent2_span_bert_start_idx'] = len(before_span2_id_list)
        row['Sent2_span_bert_end_idx']   = row['Sent2_span_bert_start_idx'] + len(span2_id_list)
        
        row['Sent2_span_only_mask']      = [0] * len(before_span2_id_list) + [1] * len(span2_id_list) + [0] * len(after_span2_id_list)
        
        sent2_root_start_idx = row['Sent2_root_idxs'][0]
        sent2_root_end_idx   = row['Sent2_root_idxs'][-1]
        
        before_root2_id_list = self.tokenizer.encode(row['Sent2_str_tokens'][:sent2_root_start_idx])  if sent2_root_start_idx != 0 else []
        root2_id_list        = self.tokenizer.encode(row['Sent2_str_tokens'][sent2_root_start_idx:sent2_root_end_idx+1])
        
        row['Sent2_root_bert_start_idx'] = len(before_root2_id_list)
        row['Sent2_root_bert_end_idx']   = row['Sent2_root_bert_start_idx'] + len(root2_id_list)
        
        row['Sent2_root_only_mask']      = [0] * len(before_root2_id_list) + [1] * len(root2_id_list) + [0] * (len(row['Sent2_bert_token_ids']) - len(before_root2_id_list) - len(root2_id_list))
        
        # pair 

        if row['Sent1_Id'] == row['Sent2_Id']:
            # get offset
            row['Sent1_idx_offset'] = 0
            row['Sent2_idx_offset'] = 0

            # model input token ids and span & root masks
            row['Token_ids'] = row['Sent1_bert_token_ids']
            row['Span1_only_masks'] = row['Sent1_span_only_mask']
            row['Span2_only_masks'] = row['Sent2_span_only_mask']
            row['Root1_only_masks'] = row['Sent1_root_only_mask']
            row['Root2_only_masks'] = row['Sent2_root_only_mask']
        else:
            # get offset
            row['Sent1_idx_offset'] = 0
            row['Sent2_idx_offset'] = len(row['Sent1_bert_token_ids']) + 1 # sentence 1 length + one sep token

            # model input token ids and span & root masks
            row['Token_ids'] = row['Sent1_bert_token_ids'] + [self.tokenizer.sep_token_id] + row['Sent2_bert_token_ids']
            row['Span1_only_masks'] = row['Sent1_span_only_mask'] + [0] * (1 + len(row['Sent2_span_only_mask']))
            row['Span2_only_masks'] = [0] * (1 + len(row['Sent1_span_only_mask'])) + row['Sent2_span_only_mask']
            row['Root1_only_masks'] = row['Sent1_root_only_mask'] + [0] * (1 + len(row['Sent2_root_only_mask']))
            row['Root2_only_masks'] = [0] * (1 + len(row['Sent1_root_only_mask'])) + row['Sent2_root_only_mask']

        return row
        
    def __len__(self, ):
        if self.get_pairs:
            return len(self.dataframe.index)
        else:
            return 2 * len(self.dataframe.index)
    
    def __getitem__(self, idx):

        if self.get_pairs:
            
            data_point = self.dataframe.iloc[idx]
            unpadded_token_ids        = torch.tensor(data_point['Token_ids']).squeeze()
            padded_token_ids          = F.pad(unpadded_token_ids, pad=(0, self.max_input_size - unpadded_token_ids.size()[0]), mode='constant', value=0)
            
            unpadded_span1_only_masks = torch.tensor(data_point['Span1_only_masks']).squeeze()
            padded_span1_only_masks   = F.pad(unpadded_span1_only_masks, pad=(0, self.max_input_size - unpadded_span1_only_masks.size()[0]), mode='constant', value=0)
            
            unpadded_root1_only_masks = torch.tensor(data_point['Root1_only_masks']).squeeze()
            padded_root1_only_masks   = F.pad(unpadded_root1_only_masks, pad=(0, self.max_input_size - unpadded_root1_only_masks.size()[0]), mode='constant', value=0)
            
            unpadded_span2_only_masks = torch.tensor(data_point['Span2_only_masks']).squeeze()
            padded_span2_only_masks   = F.pad(unpadded_span2_only_masks, pad=(0, self.max_input_size - unpadded_span2_only_masks.size()[0]), mode='constant', value=0)
            
            unpadded_root2_only_masks = torch.tensor(data_point['Root2_only_masks']).squeeze()
            padded_root2_only_masks   = F.pad(unpadded_root2_only_masks, pad=(0, self.max_input_size - unpadded_root2_only_masks.size()[0]), mode='constant', value=0)
            
            X = { # tensors
                'Token_ids':        padded_token_ids,
                'Span1_only_masks': padded_span1_only_masks,
                'Root1_only_masks': padded_root1_only_masks,
                'Span2_only_masks': padded_span2_only_masks,
                'Root2_only_masks': padded_root2_only_masks,
            },
            Y = { # int or float
                'Event_1_dur':torch.tensor(data_point['Pred1.Duration']),
                'Event_1_dur_confi':torch.tensor(data_point['Pred1.Duration.Confidence']),
                'Event_1_begin':torch.tensor(data_point['Pred1.Beg'] / 100.0),
                'Event_1_end':torch.tensor(data_point['Pred1.End'] / 100.0),
                'Event_2_dur':torch.tensor(data_point['Pred2.Duration']),
                'Event_2_dur_confi':torch.tensor(data_point['Pred2.Duration.Confidence']),
                'Event_2_begin':torch.tensor(data_point['Pred2.Beg'] / 100.0),
                'Event_2_end':torch.tensor(data_point['Pred2.End'] / 100.0),
                'Rel_confi':torch.tensor(data_point['Relation.Confidence']),
            }
            return X, Y
        else:
            index = idx // 2
            pick_one = (idx % 2) == 0
            data_point = self.self.dataframe.iloc[index]

            if pick_one:
                unpadded_token_ids        = torch.tensor(data_point['Token_ids']).squeeze()
                padded_token_ids          = F.pad(unpadded_token_ids, pad=(0, self.max_input_size - unpadded_token_ids.size()[0]), mode='constant', value=0)
                unpadded_span1_only_masks = torch.tensor(data_point['Span1_only_masks']).squeeze()
                padded_span1_only_masks   = F.pad(unpadded_span1_only_masks, pad=(0, self.max_input_size - unpadded_span1_only_masks.size()[0]), mode='constant', value=0)
                unpadded_root1_only_masks = torch.tensor(data_point['Root1_only_masks']).squeeze()
                padded_root1_only_masks   = F.pad(unpadded_root1_only_masks, pad=(0, self.max_input_size - unpadded_root1_only_masks.size()[0]), mode='constant', value=0)
                return {
                    'Sent1_bert_token_ids':padded_token_ids,
                    'Sent1_span_only_mask':padded_span1_only_masks,
                    'Sent1_root_only_mask':padded_root1_only_masks
                }, {
                    'Event_1_dur':torch.tensor(data_point['Pred1.Duration']),
                    'Event_1_dur_confi':torch.tensor(data_point['Pred1.Duration.Confidence']),
                    'Event_1_begin':torch.tensor(data_point['Pred1.Beg'] / 100.0),
                    'Event_1_end':torch.tensor(data_point['Pred1.End'] / 100.0),
                }
            else:
                unpadded_token_ids        = torch.tensor(data_point['Token_ids']).squeeze()
                padded_token_ids          = F.pad(unpadded_token_ids, pad=(0, self.max_input_size - unpadded_token_ids.size()[0]), mode='constant', value=0)
                unpadded_span2_only_masks = torch.tensor(data_point['Span2_only_masks']).squeeze()
                padded_span2_only_masks   = F.pad(unpadded_span2_only_masks, pad=(0, self.max_input_size - unpadded_span2_only_masks.size()[0]), mode='constant', value=0)
                unpadded_root2_only_masks = torch.tensor(data_point['Root2_only_masks']).squeeze()
                padded_root2_only_masks   = F.pad(unpadded_root2_only_masks, pad=(0, self.max_input_size - unpadded_root2_only_masks.size()[0]), mode='constant', value=0)
                return {
                    'Sent2_bert_token_ids':padded_token_ids,
                    'Sent2_span_only_mask':padded_span2_only_masks,
                    'Sent2_root_only_mask':padded_root2_only_masks
                }, {
                    'Event_2_dur':torch.tensor(data_point['Pred2.Duration']),
                    'Event_2_dur_confi':torch.tensor(data_point['Pred2.Duration.Confidence']),
                    'Event_2_begin':torch.tensor(data_point['Pred2.Beg'] / 100.0),
                    'Event_2_end':torch.tensor(data_point['Pred2.End'] / 100.0),
                }

    def explore(self, idx):
        # print(self.dataframe.iloc[idx])
        pass # TODO


if __name__ == "__main__":
    # # minitest
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # mini_dataset = UDS_T_Dataset(tsv_filename="UDS_T_data/first10.tsv", 
    #                     processed_jsonl_filename="UDS_T_data/first10_preprocessed.jsonl",
    #                     # processed_jsonl_filename=None,
    #                     dataset_split='train',
    #                     get_pairs=True,
    #                     max_input_size=512)
    # # dataset.explore(1)
    # # print(mini_dataset[1])
    # mini_dataloader = DataLoader(mini_dataset, batch_size=4, shuffle=True, drop_last=False)
    # for i, batch in enumerate(mini_dataloader):
    #     print(batch[0][0])
    #     print()
    #     print(batch[1])
    #     print()
    #     X = {key: value.to(device) for key, value in batch[0][0].items()} # not sure why X needs double [0]..
    #     Y = {key: value.to(device) for key, value in batch[1].items()}
    #     print(X)
    #     print()
    #     print(Y)
    #     print()
    
    # full run to generate the whole jsonl
    dataset = UDS_T_Dataset(tsv_filename="UDS_T_data/time_eng_ud_v1.2_2015_10_30.tsv", 
                        # processed_jsonl_filename="UDS_T_data/time_eng_ud_v1.2_2015_10_30.jsonl",
                        processed_jsonl_filename=None,
                        dataset_split='dev',
                        get_pairs=True,
                        max_input_size=512)
    # dataset.explore(1)
    print(dataset[1])
    