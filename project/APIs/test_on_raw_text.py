"""
Run this script to run the API on raw text input

# Under project
python APIs/test_on_raw_text.py
"""

import pickle
import sys
import os
import argparse
import json
from main import EventAPIs
from nltk import tokenize
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)

def save(args, result_list, not_done_list):
    with open(args.save_path, 'wb') as f:
        pickle.dump(result_list, f)

    result_json = {
        'error_list': not_done_list,
        'result_list': result_list
    }
    with open(args.save_path_json, 'w', encoding='utf-8') as f:
        # Use NumpyEncoder to convert numpy data to list
        # Previous error: Object of type int64 is not JSON serializable
        json.dump(result_json, f, indent=4, ensure_ascii=False,
                    cls=NumpyEncoder)
    print ('Saved')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-data', type=str, default='../../raw_text/julsepscan.txt')
    # p.add_argument('-data', type=str, default='../../ace_data/ace_rawtext_test.pkl')
    p.add_argument('-save_path', type=str, default='../../raw_text/julsepscan_pipelined.pkl')
    p.add_argument('-save_path_json', type=str, default='../../raw_text/julsepscan_pipelined.json')
    p.add_argument('-negation_detection', action='store_true', default=True,
                    help='Whether detection negation cue and scope resolution')
    args = p.parse_args()

    if args.negation_detection:
        eventAPIs = EventAPIs(negation_detection=True)
    else:
        eventAPIs = EventAPIs(negation_detection=False)
    print ('Loaded class')
    not_done_list = []

    if args.data.split('.')[-1] == 'pkl':
        with (open(args.data, "rb")) as f:
            data = pickle.load(f)
    elif args.data.split('.')[-1] == 'txt':
        with (open(args.data, "r")) as f:
            linelist = [line.rstrip() for line in f]
            data = []
            total_sen = 0
            # convert row text to list of sentences
            for line in linelist:
                sen_list = []
                if line != '':
                    # divide to sentences
                    sen_list = tokenize.sent_tokenize(line)
                data.append(sen_list)
                total_sen += len(sen_list)
            # print(data[0:100])
            # with open('../../raw_text/aprjunscan.pkl', 'wb') as f:
            #     pickle.dump(data, f)
            print ('Total sentences: ', total_sen)

    print ('Total lines: ', len(data))
    result_list = []
    for i_line, sen_list in enumerate(data):
        result_list_this_line = []
        for i_sen, text in enumerate(sen_list):
            print ('='*40, 'line num: ', i_line, "; sen num: ", i_sen)
            params_this = {
                'text': text,
                'domain': 'news'
            }
            try:
                combined_result = eventAPIs.analyze(params_this)
                combined_result['line_num'] = i_line
                combined_result['sen_num'] = i_sen
                combined_result['sentence'] = text
                result_list_this_line.append(combined_result)
            except Exception as e:
                print('?'*60)
                print('Error for this text: ', text)
                print(str(e))
                not_done_list.append([i_line, i_sen])
        result_list.append(result_list_this_line)
        if i_line % 20 == 0:
            save(args, result_list, not_done_list)

    # print (result_list)
    save(args, result_list, not_done_list)
    print ('Not successfuly text:')
    print (not_done_list)