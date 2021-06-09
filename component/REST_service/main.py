import argparse
import os
import sys
from collections import defaultdict
from flask import Flask, jsonify
import json
from flask import Response
from flask import request
import requests
import urllib.parse
import ast
import sys
sys.path.append("..")
sys.path.append("../Duration")
from Duration.inference_api import DurationAPI

if __name__ == "__main__":
    '''
    This program will establish or call an web service to call component.
    Mode 1: server. The machine is act as server to respond to web API REST calls (activate by run the program externally and set mode to 'server')
    Mode 2: client. The machine will call a server to get embedding. (activate by run the program externally and set mode to 'client')
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", "--mode", help="run as server, client [server, client]", type=str, default="server")
    parser.add_argument("-port", "--port", help="port to run this REST service", type=int, default=17000)
    args = parser.parse_args()

    # Option 2: Run as a server to provide API service
    if args.mode == "server":
        print ('-----component/REST_service: HTTP SERVER MODE-----')

        # Load component class
        durationAPI = DurationAPI(base_dir = '../Duration')

        app = Flask(__name__)
        @app.route('/duration', methods=['POST'])
        def response_pred():
            # get three parameters
            print('============REST_service')
            # text = request.args.get('text')
            # domain = request.args.get('domain')
            # events = request.args.get('events')
            # print(text)
            # print(domain)
            # print(events)
            json = request.get_json()
            print (json)
            events = json['events']
            print (events)
            json_list = durationAPI.pred(events)
            print (json_list)
            response_json = {'json_list': json_list}
            return jsonify(response_json)
        app.run(port=args.port)
    else:
        print ('-> MODE NOT CHOSEN')