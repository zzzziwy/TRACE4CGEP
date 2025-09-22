# -*- coding: utf-8 -*-

import numpy as np
import random
from util import load_json_from_file
def load_data(args):
    if args.graph_enhance:
        print(f'load data with {args.dataset_path}/MAVENSubWoRe_en.npy')
        data_document = np.load(f'{args.dataset_path}/MAVENSubWoRe_en.npy', allow_pickle=True).item()
    else:
        print(f'load data with {args.dataset_path}/MAVENSubWoRe.npy')
        data_document = np.load(f'{args.dataset_path}/MAVENSubWoRe.npy', allow_pickle=True).item()
    train_data, valid_data, test_data = data_document['train'], data_document['valid'], data_document['test']
    # sample_train_data = fewShot(args, train_data)
    # return train_data, valid_data, test_data
    return train_data, valid_data, test_data

def load_vocab(args):
    if args.graph_enhance:
        reverse_event_dict = load_json_from_file(f"{args.dataset_path}/reverse_event_dict_en.json")
        to_add = load_json_from_file(f"{args.dataset_path}/to_add_en.json")
    else:
        reverse_event_dict = load_json_from_file(f"{args.dataset_path}/reverse_event_dict.json")
        to_add = load_json_from_file(f"{args.dataset_path}/to_add.json")
    return reverse_event_dict, to_add
def fewShot(args, data):
    random.seed(args.seed)
    sampleData = random.sample(data, int(len(data) * args.Sample_rate))
    return sampleData


