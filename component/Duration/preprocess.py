from predpatt import PredPatt
import json
from torch.utils.data import Dataset, DataLoader
''' Json input format:
    [
        {
            "tokens": ["word_0", "word_1", ...],
            "events": [
                {
                    "event_type": "Movement:Transport",
                    "triggers": [{
                        "event_type": "Movement:Transport", 
                        "text": "deploy", 
                        "start_token": 5, 
                        "end_token": 5
                        }],
                    ...
                },
                ...
            ],
            "ner": [[]]
        },
        ...
    ]
'''


def predicate_info(predicate):
    '''
    Author: sidvash <sidsvash26@gmail.com>

    Input: predicate object
    Output: pred_text, token, root_token
    
    Note: If predicate is copular: pred_text is only upto first 5 words
    '''      
    copula_bool = False
    
    #Extend predicate to start from the copula
    if predicate.root.tag not in ["VERB", "AUX"]:
        all_pred = predicate.tokens
        gov_rels = [tok.gov_rel for tok in all_pred]
        if 'cop' in gov_rels:
            copula_bool = True
            cop_pos = gov_rels.index('cop')
            pred = [x.text for x in all_pred[cop_pos:]]
            pred_token = [x.position for x in all_pred[cop_pos:]]
            def_pred_token = predicate.root.position  #needed for it_happen set
            cop_bool = True  
            #print(predicate, idx)
            
        elif predicate.root.tag == "ADJ":
            pred_token = [predicate.root.position]
            pred = [predicate.root.text]
            def_pred_token = predicate.root.position
        else: ## Different from protocol as we are considering all predicates
            pred_token = [predicate.root.position]
            pred = [predicate.root.text]
            def_pred_token = predicate.root.position
            
    #Else keep the root        
    else:
        pred_token = [predicate.root.position]
        pred = [predicate.root.text]
        def_pred_token = predicate.root.position 

    #Stringify pred and pred_tokens:
    #pred_token = "_".join(map(str, pred_token))

    if len(pred)>5:
        pred = pred[:5]
        pred = " ".join(pred) + "..."
    else:
        pred = " ".join(pred)
    
    return pred, pred_token, def_pred_token


def extract_pp_obj_instance(pp_obj_instance, pp_obj):
    _, span_idx_list, root_idx = predicate_info(pp_obj_instance)
    word_tokens = [token.text for token in pp_obj.tokens]
    span_text   = ' '.join([pp_obj.tokens[i].text for i in span_idx_list])
    root_text   = pp_obj.tokens[root_idx].text
    return word_tokens, span_text, span_idx_list, root_text, root_idx


class TempEveDataset(Dataset):
    def __init__(self, json_filename, from_UDST_dataset = False, from_pipeline = True):
        
        self.sentences_wordlist = [] # list of string list
        self.spans = []              # list of string
        self.spans_idx = []          # list of int list
        self.roots = []              # list of string
        self.roots_idx = []          # list of int
        
        if from_pipeline:
            if type(json_filename) == str:
                json_objs = json.load(open(json_filename))
            else:
                json_objs = json_filename

            print("json file size:", len(json_objs))

            for obj in json_objs:
                
                if len(obj['events']) > 0: # detected events
                    for event in obj['events']:
                        for trigger in event['triggers']:
                            self.sentences_wordlist.append(obj['tokens'])
                            self.spans.append(trigger['text'])
                            self.spans_idx.append(list(range(int(trigger['start_token']), int(trigger['end_token']) + 1))) # seems like one-word span only but just to make sure
                            self.roots.append(trigger['text'].split()[0]) # seems like one-word only but just to make sure
                            self.roots_idx.append(int(trigger['start_token']))
                # else: # no events detected
                #     pass
                #     # some odd error with "\""", see main below for examples
                #     if len(obj['tokens']) > 3: # if truly a sentence but without trigger/event detected from event extraction
                #         # try with predpatt
                #         sentence = " ".join(obj['tokens'])
                #         print(sentence)
                #         pp_obj = PredPatt.from_sentence(sentence) # https://github.com/hltcoe/PredPatt/blob/5ce4b88c4678dcf7c99a6b0377e0f641701b8390/predpatt/patt.py#L376
                #         if len(pp_obj.instances) > 0:
                #             for pp_obj_instance in pp_obj.instances:
                #                 word_tokens, span_text, span_idx_list, root_text, root_idx = extract_pp_obj_instance(pp_obj_instance, pp_obj)
                #                 self.sentences_wordlist.append(word_tokens)
                #                 self.spans.append(span_text)
                #                 self.spans_idx.append(span_idx_list)
                #                 self.roots.append(root_text)
                #                 self.roots_idx.append(root_idx)
                    #     else:
                    #         # do nothing, filter it outs
                    #         pass
                    # else:
                    #     # append entire sentence? no, do nothing (filter it out)
                    #     pass

        elif from_UDST_dataset:
            pass

    def __len__(self):
        return len(self.sentences_wordlist)

    def __getitem__(self, index):
        return {
            "words_list": self.sentences_wordlist[index],
            "span_text": self.spans[index],
            "root_text": self.roots[index],
            "span_idx_list": self.spans_idx[index],
            "root_idx": self.roots_idx[index]
        }


if __name__ == "__main__":
    # pp = PredPatt.from_sentence('Chris loves silly dogs and clever cats .')
    # print(predicate_info(pp.instances[0]))
    # print(pp.tokens[0].text)
    # test_word_list = ["We", "'re", "talking", "about", "possibilities", "of", "full", "scale", "war", "with", "former", "Congressman", "Tom", "Andrews", ",", "Democrat", "of", "Maine", "."]
    # test_word_list_orig = ["New", "Questions", "About", "Attacking", "Iraq", ";", "Is", "Torturing", "Terrorists", "Necessary", "?", ]
    # print('orig:\t', ' '.join(test_word_list_orig))
    # test_word_list_good = ["New", "Questions", "About", "Attacking", "\"", "Iraq", ";", "Is", "Torturing", "Terrorists", "Necessary", "?", ]
    # print('good:\t', ' '.join(test_word_list_good))
    # test_word_list_bad = ["New", "Questions", "About", "Attacking", "Iraq", ";", "Is", "Torturing", "Terrorists", "Necessary", "\"","?", ]
    # print('bad:\t', ' '.join(test_word_list_bad))
    # test_word_list = ["Why", "do", "we", "have", "to", "learn", "it", "from", "\"", "Newsweek", "\"", "?"]
    """
    odd "\n""
    error: "KeyError: 1"
    print("inside JPyoeBackend!! indices_to_words[index]:", indices_to_words[index])
    print(len(indices_to_words)) # 0
    print(index) # 1
    """
    # test_word_list = ["Why", "do", "we", "have", "to", "learn", "it", "from", "Newsweek", "?"]
    # test_word_list = ["And", "so", "I", "would", "like", "you", "to", "take", "a", "look", "at", "the", "CNN/\"USA", "TODAY\"", "\"", "Gallup", "poll", ",", "taken", "last", "week", ",", "should", "U.S.", "troops", "to", "go", "to", "Iraq", "to", "remove", "Saddam", "Hussein", "from", "power", "."]
    # sentence = " ".join(test_word_list)
    # print(sentence)
    # pp_obj = PredPatt.from_sentence(sentence)
    # for predicates in pp_obj.instances:
    #     span, span_idx_list, root_idx = predicate_info(predicates)
    #     print(span, span_idx_list, root_idx)
    #     print([token.text for token in pp_obj.tokens])
    #     print(' '.join([pp_obj.tokens[i].text for i in span_idx_list]))
    #     print(pp_obj.tokens[root_idx].text)

    dataset = TempEveDataset("mu_dev_out.json", False, True)
    print("dataset size:", len(dataset))
    print("data sample:", dataset[0])

    dataloader = DataLoader(dataset, batch_size=4)

    dataloader = iter(dataloader)
    batch = next(dataloader)

    print(batch)
