import json
import argparse
import numpy as np
from allennlp.predictors.predictor import Predictor

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

def get_coference(doc):
    pred = predictor.predict(document = doc)
    clusters = pred['clusters']
    document = pred['document'] 
    top_spans = pred['top_spans']

    # find the main span for each cluster
    clusters_top_span = []
    for i in range(0, len(clusters)):
        one_cl = clusters[i]
        span_rank = [top_spans.index(span) for span in one_cl]
        top_span = np.argmin(span_rank)
        clusters_top_span.append(one_cl[top_span])
    pred['clusters_top_span'] = clusters_top_span

    # convert top span for each cluster to text
    clusters_top_span_text = []
    for each_top_span in clusters_top_span:
        span_text = document[each_top_span[0]:(each_top_span[1]+1)]
        clusters_top_span_text.append(span_text)
    pred['clusters_top_span_text'] = clusters_top_span_text
    
    return pred

def save(args, result_json):
    # result_json = {
    #     'error_list': not_done_list,
    #     'result_list': result_list
    # }
    with open(args.save_path_json, 'w', encoding='utf-8') as f:
        # Use NumpyEncoder to convert numpy data to list
        # Previous error: Object of type int64 is not JSON serializable
        json.dump(result_json, f, indent=4, ensure_ascii=False,
                    cls=NumpyEncoder)
    print ('Saved')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-data', type=str, default='../../raw_text/test.0622_pipelined.json')
    p.add_argument('-save_path_json', type=str, default='../../raw_text/test.0622_pipelined_coref.json')
    args = p.parse_args()

    data = json.load(open(args.data))
    # load AllenNLP predictor
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")

    docs = data['result_list']

    for doc_num, doc in enumerate(docs):
        # ensemble the document from sentences
        doc_text_list = []
        doc_text_len = [0]
        for sen in doc:
            print(sen['tokens'])
            doc_text_list.append(sen['sentence'])
            doc_text_len.append(len(sen['tokens']))
        doc_text = ' '.join(doc_text_list)
        for i, num in enumerate(doc_text_len):
            if i >= 1:
                doc_text_len[i] += doc_text_len[i - 1]
        sens_idx_beg = doc_text_len[:-1]
        sens_idx_end = doc_text_len[1:]
        # sens_idx_beg saves the beginning token idx of each sentence in the doc
        # sens_idx_end saves the ending token idx of each sentence in the doc
        print(doc_text)
        print(sens_idx_beg)
        print(sens_idx_end)
        
        # get coreference result for the document
        coref_pred = get_coference(doc_text)
        print(coref_pred)
        
        # save coref result to json
        for i_cluster, cluster in enumerate(coref_pred['clusters']):
            print('------')
            print(cluster)
            for mention in cluster:
                print(mention)
                # identify which sentence that this mention belongs to
                sen_nums = [i for i, beg in enumerate(sens_idx_beg) if mention[0] >= beg and mention[0] < sens_idx_end[i]]
                for sen_num in sen_nums:
                    mention_idx_in_this_sen = [i - sens_idx_beg[sen_num] for i in mention]
                    events_of_this_sen = data['result_list'][doc_num][sen_num]['events']
                    for i_e, e in enumerate(events_of_this_sen):
                        for i_arg, arg_obj in enumerate(e['arguments']):
                            if arg_obj['start_token'] == mention_idx_in_this_sen[0] and arg_obj['end_token'] == mention_idx_in_this_sen[1]:
                                # this argument is exactly the one need to update its text to co-referenced span
                                data['result_list'][doc_num][sen_num]['events'][i_e]['arguments'][i_arg]["text"] = " ".join(coref_pred['clusters_top_span_text'][i_cluster])

    save(args, data)