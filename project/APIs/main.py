import json
import sys
import requests
import time
import spacy
import json
sys.path.append("..")
sys.path.append("../component/BioMedEventEx") # need to import config.py
from component.BioMedEventEx.predict import BioMedEventExAPI
from component.TempRel.code.joint_model import TempRelAPI
sys.path.append("../component/BETTER/joint") # need to import model in absolute path
from component.BETTER.joint.event_pipeline_demo import BETTER_API
# sys.path.append("../component/Duration")
# from component.Duration.inference_api import DurationAPI
import timeit
import torch

def closest(lst, K): 
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 

flatten = lambda l: [item for sublist in l for item in sublist]

class EventAPIs:
    def __init__(self, negation_detection=False):
        # loading some resources and library
        print('Loading API classes')
        print('Number of cuda devices: ', torch.cuda.device_count())
        t0= time.clock()
        self.negation_detection = negation_detection
        self.rest_server_url = 'http://localhost'
        self.rest_server_port = 17000
        self.bioMedEventExAPI = BioMedEventExAPI('../component/BioMedEventEx')
        self.betterAPI = BETTER_API('../component/BETTER/joint')
        self.temprelAPI = TempRelAPI(base_dir = '../component/TempRel/code')
        # self.durationAPI = DurationAPI(base_dir = '../component/Duration',
        #                             gpu_id = 0 if torch.cuda.device_count() > 1 else -1)

        if self.negation_detection:
            sys.path.append("../component/NegationDetection")
            from component.NegationDetection.train import NegaionDetectionAPI
            self.negationdetectionAPI = NegaionDetectionAPI(base_dir='../component/NegationDetection', 
                                                            functions=['cue_detection', 'scope_resolution'])

        self.spacy_pretrained_model = spacy.load("en_core_web_sm")
        self.show_time = True

        self.flag_common_events = False
        t1 = time.clock() - t0
        print("Time elapsed: ", t1)

    def duration_rest_api(self, params, events, show=True):
        print ('---HTTP CLIENT MODE---')
        url = "%s:%s/duration" % (self.rest_server_url, self.rest_server_port)
        body = {
            'params': params,
            'events': events
        }
        response = requests.post(url, json=body)
        print("Status code: ", response.status_code)
        print(response)
        print(response.json())
        return response.json()['json_list']

    def analyze(self, params):
        print('APIs.main.analyze_text:', params['text'])
        print(params)
        if self.show_time:
            start = timeit.default_timer()

        # TOKENIZATION
        text = params['text']
        text = text.replace("\n", " ")
        text_tokenized = [i.text for i in self.spacy_pretrained_model(text)]
        # text_tokenized = [x for x in text_tokenized if x != ' ']
        print(text_tokenized)
        # text = ' '.join(text_tokenized)
        print(text)

        if self.show_time:
            s1 = timeit.default_timer()
            print('Time for tokenization: ', s1 - start)  

        # CALL APIs
        if params['domain'] == 'bio':
            event_ex_result = self.bioMedEventExAPI.pred(text)
            print('======Bio event_ex_result')
            print(event_ex_result)
        else:
            event_ex_result = self.betterAPI.pred(text_tokenized)
            print('======News event_ex_result')
            print(event_ex_result)

        if self.show_time:
            s2 = timeit.default_timer()
            print('Time for better: ', s2 - s1)  

        if params['domain'] != 'bio':
            try:
                temp_rel_result = self.temprelAPI.pred(text)
                print('======temp_rel')
                print (temp_rel_result)
            except:
                temp_rel_result = {
                    'events': [],
                    'relations': []
                }
        else:
            temp_rel_result = {
                'events': [],
                'relations': []
            }

        if self.show_time:
            s3 = timeit.default_timer()
            print('Time for temprel: ', s3 - s2)  

        # MERGE TEMP_REL_RESULT and EVENT_EX_RESULT
        # create a event index mapping table, from temprel events to Mu's event
        event_ex_result = event_ex_result[0]
        all_triggers = flatten([event_extracted['triggers'] for event_extracted in event_ex_result['events']])
        # print(all_triggers)
        # print(flatten(all_triggers))
        map_temp_rel_2_better_event_idx = {}
        for event in temp_rel_result['events']:
            found_map_flag = False
            event_idx = [i for i, x in enumerate(event_ex_result['tokens']) if x.replace('.','') == event[0]]
            if event[1] in event_idx:
                mapped_idx = event[1]
                found_map_flag = True
            elif len(event_idx) > 0:
                mapped_idx = closest(event_idx, event[1])
                if abs(mapped_idx - event[1]) <= 2:
                    found_map_flag = True
            if found_map_flag:
                map_temp_rel_2_better_event_idx[event[1]] = {
                    'mapped_idx': mapped_idx,
                    'text': event[0],
                    'contained': False
                }
                # check whether this event is within extraction result
                for tri in all_triggers:
                    if event[0] == tri['text'] and mapped_idx == tri['start_token']:
                        map_temp_rel_2_better_event_idx[event[1]]['contained'] = True
        print(map_temp_rel_2_better_event_idx)

        combined_result = event_ex_result

        if self.show_time:
            s4 = timeit.default_timer()
            print('Time for merge 1: ', s4 - s3)  

        # add event triggers that predicted by TempRel model but not in BETTER model to combined_result
        tmp_event_list = combined_result['events']
        for _, mapping in map_temp_rel_2_better_event_idx.items():
            if mapping['contained'] == False:
                tmp_event_list.append({
                    'event_type': '',
                    'triggers': [{
                        'event_type': '',
                        'text': mapping['text'],
                        'start_token': mapping['mapped_idx'],
                        'end_token': mapping['mapped_idx']
                    }],
                    'arguments': [],
                    'duration': ''
                })
        combined_result['events'] = tmp_event_list

        if self.show_time:
            s5 = timeit.default_timer()
            print('Time for merge 2: ', s5 - s4)  

        # manually add duration
        for i, event_single in enumerate(combined_result['events']):
            combined_result['events'][i]['duration'] = ''

        # only keep relation predictions that are within common event space
        relations_left = []
        for relation in temp_rel_result['relations']:
            if relation[0] in map_temp_rel_2_better_event_idx and relation[1] in map_temp_rel_2_better_event_idx:
                if (self.flag_common_events and map_temp_rel_2_better_event_idx[relation[0]]['contained'] and map_temp_rel_2_better_event_idx[relation[1]]['contained']) \
                    or (not self.flag_common_events):
                    # only keep the common events if the flag is set on
                    if relation[2] != 'VAGUE' and relation[2] != 'NONE':
                        # filter none and vague result
                        relations_left.append(relation)

        combined_result['relations'] = relations_left

        if self.show_time:
            s6 = timeit.default_timer()

        # CALL DURATION API AFTER EVENT MERGING
        if params['domain'] != 'bio':
            # try:
            duration_result = self.duration_rest_api(params, [combined_result])
            # duration_result = self.durationAPI.pred([combined_result])
            print('======duration_result')
            print(duration_result)
            # except:
            #     duration_result = []
        else:
            duration_result = []

        if self.show_time:
            s7 = timeit.default_timer()
            print('Time for duration: ', s7 - s6) 

        # MERGE DURATION_RESULT IN FINAL COMBINED_RESULT
        for i, event_single in enumerate(combined_result['events']):
            trigger_single = event_single['triggers'][0]
            for i_duration, duration_single in enumerate(duration_result):
                if trigger_single['start_token'] == duration_single['pred_idx']:
                    # and trigger_single['text'] == duration_single['pred_text']:
                    combined_result['events'][i]['duration'] = duration_single['duration']

        # USE NEGATION DETECTION MODULE TO ANNOTATE SPECULATION EVENTS
        if self.negation_detection:
            cues_outputs, scope_outputs = self.negationdetectionAPI.pred([' '.join(combined_result['tokens'])])
            # input and output for this API are both lists, so take the first item
            cues_output = cues_outputs[0]
            scope_output = scope_outputs[0]
            combined_result['negation_cue'] = cues_output
            combined_result['negation_scope'] = scope_output
            # add extra notation to events that in the speculation part
            for i, event_single in enumerate(combined_result['events']):
                trigger_single = event_single['triggers'][0]
                if scope_output[trigger_single['start_token']] == 1:
                    # this trigger is within the negation scope
                    combined_result['events'][i]['speculation'] = True

            if self.show_time:
                s8 = timeit.default_timer()
                print('Time for negation detection: ', s8 - s7) 
        
        if self.show_time:
            stop = timeit.default_timer()
            print('Time for the whole inference: ', stop - start) 

        print ('--------')
        print (combined_result)
        return combined_result

if __name__ == '__main__':
    eventAPIs = EventAPIs(negation_detection=True)
    test_cases = [{
        'text': 'A powerful ice storm continues to maintain its grip. Yesterday New York governor George Pataki toured five counties that have been declared under a state of emergency',
        'domain': 'news',
    },{
        'text': "The United States is not considering sending troops to Mozambique to combat the terrorist threat in the northern province of Cabo Delgado, but it is willing to boost \"civilian counter-terrorism capabilities\", said the US Coordinator for Counterterrorism, Nathan Sales, on Tuesday 8 December.",
        'domain': 'news',
    }]
    for test_case in test_cases:
        result = eventAPIs.analyze(test_case)
        print('======')
        print(result)