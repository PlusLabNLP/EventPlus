from __future__ import unicode_literals
from django.shortcuts import render
from django.http.response import JsonResponse
# from django.shortcuts import render_to_response
from django.http import HttpResponse, HttpResponseRedirect
import spacy
from spacy import displacy
import json
import random
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from APIs.main import *
eventAPIs = EventAPIs()

def index(request):
    return render(request, 'index.html')

def json_generator(result):

    result_news = {
        "tokens": ["A", "powerful", "ice", "storm", "continues", "to", "maintain", "its", "grip", ".", "Yesterday",
                   "New", "York", "governor", "George", "Pataki", "toured", "five", "counties", "that", "have", "been",
                   "declared", "under", "a", "state", "of", "emergency"],
        "events": [{
            "event_type": "Movement:Transport",
            "triggers": [{
                "event_type": "Movement:Transport",
                "text": "toured",
                "start_token": 16,
                "end_token": 16
            }],
            "arguments": [{
                "role": "Artifact",
                "text": "George Pataki",
                "start_token": 14,
                "end_token": 15
            }, {
                "role": "Destination",
                "text": "counties",
                "start_token": 18,
                "end_token": 18
            }],
            "duration": "days"
        }],
        "ner": [
            [11, 12, "GPE"],
            [13, 13, "PER"],
            [14, 15, "PER"],
            [18, 18, "GPE"]
        ],
        "relations": []
    }

    result_bio = {'tokens': ['We', 'have', 'found', 'that', 'the', 'HTLV-1', 'transactivator', 'protein', ',', 'tax', ',',
                    'acts', 'as', 'a', 'costimulatory', 'signal', 'for', 'GM-CSF', 'and', 'IL-2', 'gene',
                    'transcription', ',', 'in', 'that', 'it', 'can', 'cooperate', 'with', 'TCR', 'signals', 'to',
                    'mediate', 'high', 'level', 'gene', 'expression', '.'], 'events': [
            {'event_type': 'Positive_regulation',
             'triggers': [{'event_type': 'Positive_regulation', 'text': 'acts', 'start_token': 11, 'end_token': 11}],
             'arguments': [{'role': 'Theme', 'text': 'transcription', 'start_token': 21, 'end_token': 21},
                           {'role': 'Cause', 'text': 'HTLV-1 transactivator protein', 'start_token': 5,
                            'end_token': 7}], "duration": "seconds"}, {'event_type': 'Positive_regulation', 'triggers': [
                {'event_type': 'Positive_regulation', 'text': 'acts', 'start_token': 11, 'end_token': 11}],
                                                'arguments': [
                                                    {'role': 'Theme', 'text': 'transcription', 'start_token': 21,
                                                     'end_token': 21},
                                                    {'role': 'Cause', 'text': 'HTLV-1 transactivator protein',
                                                     'start_token': 5, 'end_token': 7}], "duration": "seconds"},
            {'event_type': 'Positive_regulation',
             'triggers': [{'event_type': 'Positive_regulation', 'text': 'acts', 'start_token': 11, 'end_token': 11}],
             'arguments': [{'role': 'Theme', 'text': 'transcription', 'start_token': 21, 'end_token': 21},
                           {'role': 'Cause', 'text': 'tax', 'start_token': 9, 'end_token': 9}], "duration": "seconds"},
            {'event_type': 'Positive_regulation',
             'triggers': [{'event_type': 'Positive_regulation', 'text': 'acts', 'start_token': 11, 'end_token': 11}],
             'arguments': [{'role': 'Theme', 'text': 'transcription', 'start_token': 21, 'end_token': 21},
                           {'role': 'Cause', 'text': 'tax', 'start_token': 9, 'end_token': 9}], "duration": "seconds"},
            {'event_type': 'Transcription',
             'triggers': [{'event_type': 'Transcription', 'text': 'transcription', 'start_token': 21, 'end_token': 21}],
             'arguments': [{'role': 'Theme', 'text': 'GM-CSF', 'start_token': 17, 'end_token': 17}], "duration": "seconds"},
            {'event_type': 'Transcription',
             'triggers': [{'event_type': 'Transcription', 'text': 'transcription', 'start_token': 21, 'end_token': 21}],
             'arguments': [{'role': 'Theme', 'text': 'IL-2', 'start_token': 19, 'end_token': 19}], "duration": "seconds"}],
         'ner': [[5, 7, 'PROTEIN'], [9, 9, 'PROTEIN'], [17, 17, 'PROTEIN'], [19, 19, 'PROTEIN'], [29, 29, 'PROTEIN']]}

    result_bio_1 = {'tokens': ['We', 'show', 'that', 'ligand-induced', 'homodimerization', 'of', 'chimeric', 'surface',
                     'receptors', 'consisting', 'of', 'the', 'extracellular', 'and', 'transmembrane', 'domains', 'of',
                     'the', 'erythropoietin', 'receptor', 'and', 'of', 'the', 'intracellular', 'domain', 'of',
                     'IL-4Ralpha', 'induces', 'Janus', 'kinase', '1', '(', 'Jak1', ')', 'activation', ',', 'STAT6',
                     'activation', ',', 'and', 'Cepsilon', 'germline', 'transcripts', 'in', 'human', 'B', 'cell',
                     'line', 'BJAB', '.'], 'events': [{'event_type': 'Positive_regulation', 'triggers': [
            {'event_type': 'Positive_regulation', 'text': 'activation', 'start_token': 34, 'end_token': 34}],
                                                       'arguments': [{'role': 'Theme', 'text': 'Janus kinase 1',
                                                                      'start_token': 28, 'end_token': 30}], "duration": "seconds"},
                                                      {'event_type': 'Positive_regulation', 'triggers': [
                                                          {'event_type': 'Positive_regulation', 'text': 'activation',
                                                           'start_token': 34, 'end_token': 34}], 'arguments': [
                                                          {'role': 'Theme', 'text': 'Jak1', 'start_token': 32,
                                                           'end_token': 32}], "duration": "seconds"}, {'event_type': 'Positive_regulation',
                                                                                'triggers': [{
                                                                                                 'event_type': 'Positive_regulation',
                                                                                                 'text': 'activation',
                                                                                                 'start_token': 37,
                                                                                                 'end_token': 37}],
                                                                                'arguments': [
                                                                                    {'role': 'Theme', 'text': 'STAT6',
                                                                                     'start_token': 36,
                                                                                     'end_token': 36}], "duration": "seconds"},
                                                      {'event_type': 'Positive_regulation', 'triggers': [
                                                          {'event_type': 'Positive_regulation', 'text': 'induces',
                                                           'start_token': 27, 'end_token': 27}], 'arguments': [
                                                          {'role': 'Theme', 'text': 'activation', 'start_token': 34,
                                                           'end_token': 34},
                                                          {'role': 'Cause', 'text': 'IL-4Ralpha', 'start_token': 26,
                                                           'end_token': 26}], "duration": "seconds"}, {'event_type': 'Positive_regulation',
                                                                                'triggers': [{
                                                                                                 'event_type': 'Positive_regulation',
                                                                                                 'text': 'induces',
                                                                                                 'start_token': 27,
                                                                                                 'end_token': 27}],
                                                                                'arguments': [{'role': 'Theme',
                                                                                               'text': 'activation',
                                                                                               'start_token': 34,
                                                                                               'end_token': 34},
                                                                                              {'role': 'Cause',
                                                                                               'text': 'IL-4Ralpha',
                                                                                               'start_token': 26,
                                                                                               'end_token': 26}], "duration": "seconds"},
                                                      {'event_type': 'Positive_regulation', 'triggers': [
                                                          {'event_type': 'Positive_regulation', 'text': 'induces',
                                                           'start_token': 27, 'end_token': 27}], 'arguments': [
                                                          {'role': 'Theme', 'text': 'activation', 'start_token': 37,
                                                           'end_token': 37},
                                                          {'role': 'Cause', 'text': 'IL-4Ralpha', 'start_token': 26,
                                                           'end_token': 26}], "duration": "seconds"}],
          'ner': [[6, 8, 'PROTEIN'], [12, 15, 'PROTEIN'], [18, 19, 'PROTEIN'], [23, 24, 'PROTEIN'], [26, 26, 'PROTEIN'],
                  [28, 30, 'PROTEIN'], [32, 32, 'PROTEIN'], [36, 36, 'PROTEIN'], [40, 42, 'RNA'],
                  [44, 48, 'CELL_LINE']]}


    # result = result_news
    # result = result_bio
    # result = result_bio_1

    color_schema = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69", "#fccde5", "#bc80bd", "#ccebc5"]
    # get rid of indicators to handle the nested situation
    # indicators = np.zeros(len(result["tokens"]))
    result_array = []
    graph = {"nodes": [], "links": []}

    if "events" in result:
        color_dict = {}
        if len(color_schema) >= len(result["events"]):
            for i in range(len(result["events"])):
                color_dict[i] = color_schema[i]
        else:
            for i in range(len(color_schema)):
                color_dict[i] = color_schema[i]
            short = len(result["events"]) - len(color_schema)
            for i in range(short):
                r = lambda: random.randint(0, 255)
                color = '#%02X%02X%02X' % (r(), r(), r())
                color_dict[i + len(color_schema)] = color

        event_count = 0
        for event in result["events"]:
            if "triggers" in event:
                # according to team member, there's only one trigger per event
                trigger = event["triggers"][0]
                result_array.append({
                    "start": trigger["start_token"],
                    "end": trigger["end_token"] + 1,
                    "role": "trigger",
                    "label": trigger["event_type"],
                    "duration": event["duration"],
                    "event": event_count,
                    "color": color_dict[event_count]
                })

                graph["nodes"].append({
                    "name": trigger["text"],
                    "label": event["duration"],
                    "id": trigger['start_token'],
                    "color": color_dict[event_count],
                    "type": trigger["event_type"]
                })

            if "arguments" in event:
                arguments = event["arguments"]
                for argument in arguments:
                    result_array.append({
                        "start": argument["start_token"],
                        "end": argument["end_token"] + 1,
                        "role": "argument",
                        "label": argument["role"],
                        "event": event_count,
                        "color": color_dict[event_count]
                    })

            event_count += 1

    if "ner" in result:
        ner = result["ner"]
        for each_ner in ner:
            start = each_ner[0]
            end = each_ner[1] + 1
            label = each_ner[2]
            result_array.append({
                "start": start,
                "end": end,
                "ner": label
            })
            # if indicators[start] == 1:
            #     # this ner should be another attribute given to either triggers or arguments
            #     result_array[start]["ner"] = label
            #     if result_array[start]["end"] != end:
            #         raise Exception(
            #             f"The detected NER {(start, end)} is different from event labels {(start, result_array[indicators[start]]['end'])}")
            # else:
            #     result_array[start] = {
            #         "end": end,
            #         "ner": label
            #     }
            #     indicators[start] = 1

    if "relations" in result:
        relation = result["relations"]
        if len(relation) > 0:
            for item in relation:
                graph["links"].append({
                    "source": item[0],
                    "target": item[1],
                    "type": item[2]
                })

    return_dict = {
        "tokens": result["tokens"],
        "labels": result_array,
        "graph": graph
    }

    return return_dict

@csrf_exempt
def analyze_text(request):
    # Extract params
    parameters = request.POST
    print('views.analyze_text: ', parameters['text'])
    result = eventAPIs.analyze(parameters)
    print ('views.analyze_text: Start generating json by json_generator')
    sample_json = json_generator(result)
    print('views.analyze_text: ', sample_json)
    return JsonResponse(sample_json)