"""
Run this script to run the API on the entire ACE dev and test data

# Under project
python APIs/test_on_ace_data.py
"""

import pickle
import sys
import os
import argparse
import json
from main import EventAPIs

def save(args, result_list, not_done_list):
    with open(args.save_path, 'wb') as f:
        pickle.dump(result_list, f)

    result_json = {
        'error_list': not_done_list,
        'result_list': result_list
    }
    with open(args.save_path_json, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=4)
    print ('Saved')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-data', type=str, default='../../ace_data/ace_rawtext_dev.pkl')
    # p.add_argument('-data', type=str, default='../../ace_data/ace_rawtext_test.pkl')
    p.add_argument('-save_path', type=str, default='../../ace_data/ace_rawtext_dev_pipelined.pkl')
    p.add_argument('-save_path_json', type=str, default='../../ace_data/ace_rawtext_dev_pipelined.json')
    args = p.parse_args()
    eventAPIs = EventAPIs()
    print ('Loaded class')
    not_done_list = []

    with (open(args.data, "rb")) as f:
        data = pickle.load(f)
    
    print ('Total sentences: ', len(data))
    result_list = []
    for i, text in enumerate(data):
        print ('='*40, i)
        params_this = {
            'text': text,
            'domain': 'news'
        }
        try:
            combined_result = eventAPIs.analyze(params_this)
            result_list.append(combined_result)
        except Exception as e:
            print('?'*60)
            print('Error for this text: ', text)
            print(str(e))
            not_done_list.append(i)
        if i % 20 == 0:
            save(args, result_list, not_done_list)

    # print (result_list)
    save(args, result_list, not_done_list)
    print ('Not successfuly text:')
    print (not_done_list)