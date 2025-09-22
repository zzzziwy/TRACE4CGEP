import json
from Handler.util_handler import *
import networkx as nx
import tqdm

def load_json_from_file(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json_data


def correct_data(dataSet):
    for i in range(len(dataSet)):
        dataSet[i]['node'] = doCorrect(dataSet[i]['node'])
        dataSet[i]['candiSet'] = doCorrect(dataSet[i]['candiSet'])
    return dataSet

def collect_mult_event(data, tokenizer):
    multi_event = []
    to_add = {}
    special_multi_event_token = []
    event_dict = {}
    reverse_event_dict = {}
    for sentence in data:
        multi_event, to_add, special_multi_event_token, event_dict, reverse_event_dict = doCollect(sentence['node'],
                                                                                                   tokenizer,
                                                                                                   multi_event, to_add,
                                                                                                   special_multi_event_token,
                                                                                                   event_dict,
                                                                                                   reverse_event_dict)
        multi_event, to_add, special_multi_event_token, event_dict, reverse_event_dict = doCollect(sentence['candiSet'],
                                                                                                   tokenizer,
                                                                                                   multi_event, to_add,
                                                                                                   special_multi_event_token,
                                                                                                   event_dict,
                                                                                                   reverse_event_dict)

    return multi_event, special_multi_event_token, event_dict, reverse_event_dict, to_add

def replace_mult_event(data, reverse_event_dict):
    for i in range(len(data)):
        data[i]['node'] = doReplace(data[i]['node'], reverse_event_dict)
        data[i]['candiSet'] = doReplace(data[i]['candiSet'], reverse_event_dict)
    return data

def savePredict(batch_indices, file, prediction, candiSet, labels):
    for i in range(len(prediction)):
        predtCandi = prediction[i][candiSet[i]].tolist()
        label = candiSet[i].index(labels[i])
        file.write(str(batch_indices[i].item()) + '\t')
        for score in predtCandi:
            file.write(str(score) + '\t')
        for id in candiSet[i]:
            file.write(str(id) + '\t')
        file.write(str(label) + '\n')
    return



