import torch
import numpy as np
import os

from Handler.tools_handler import *
from Handler.graph_importance_precompute import GraphImportancePrecomputer

# tokenize sentence and get event idx
def get_batch(data, args, indices, tokenizer, deactivate = False, linearization_mode='sequential'):
    batch_idx, batch_mask, batch_coevent, batch_causal = [], [], [], []
    event_tokenizer_pos, event_key_pos = [], []
    mask_indices, sentences, labels, candiSet = [], [], [], []
    for idx in indices:
        if deactivate:
            processed_sample = apply_graph_deactivation(data[idx], args, idx)
        else:
            processed_sample = data[idx]

        candi = [tokenizer.encode(processed_sample['candiSet'][i][5])[1] for i in range(len(processed_sample['candiSet']))]
        template, relation = getTemplate(args, processed_sample, linearization_mode)
        sentence = getSentence(args, tokenizer, processed_sample, processed_sample['edge'])
        assert relation[-1] == processed_sample['edge'][-1]
        assert candi[processed_sample['label']] == tokenizer.encode(processed_sample['node'][relation[-1][-1]][5])[1]

        arg_idx, arg_mask, arg_coevent, arg_causal = tokenizerHandler(args, template, tokenizer)
        arg_mask = getStructNeighbour(arg_idx, 27037, 'Nstruct')

        label = tokenizer.encode(processed_sample['node'][relation[-1][-1]][5])[1]
        labels.append(label)
        ePosition, ePositionKey = getposHandler(arg_idx, relation, tokenizer)
        event_tokenizer_pos.append(ePosition)
        event_key_pos.append(ePositionKey)

        sentences.append(sentence)
        candiSet.append(candi)
        if len(batch_idx) == 0:
            batch_idx, batch_mask, batch_coevent, batch_causal = arg_idx, arg_mask, arg_coevent, arg_causal
            mask_indices = torch.nonzero(arg_idx == tokenizer.mask_token_id, as_tuple=False)[0][1]
            mask_indices = torch.unsqueeze(mask_indices, 0)
        else:
            batch_idx, batch_mask = torch.cat((batch_idx, arg_idx), dim=0), torch.cat((batch_mask, arg_mask), dim=0)
            batch_coevent, batch_causal = torch.cat((batch_coevent, arg_coevent), dim=0), torch.cat((batch_causal, arg_causal), dim=0)
            mask_indices = torch.cat((mask_indices, torch.unsqueeze(torch.nonzero(arg_idx == tokenizer.mask_token_id, as_tuple=False)[0][1], 0)), dim=0)
    return batch_idx, batch_mask, batch_coevent, batch_causal, event_tokenizer_pos, event_key_pos, mask_indices, sentences, labels, candiSet

# calculate p, r, f1
def calculate(prediction, candiSet, labels, batch_size):
    mrr, hit1, hit3, hit10, hit20, hit50 = [], [], [], [], [], []
    for i in range(batch_size):
        predtCandi = prediction[i][candiSet[i]].tolist()
        label = candiSet[i].index(labels[i])
        labelScore = predtCandi[label]
        predtCandi.sort(reverse=True)
        rank = predtCandi.index(labelScore)
        mrr.append(1/(rank+1))
        hit1.append(int(rank < 1))
        hit3.append(int(rank < 3))
        hit10.append(int(rank < 10))
        hit20.append(int(rank < 20))
        hit50.append(int(rank < 50))

    return mrr, hit1, hit3, hit10, hit20, hit50



if __name__ == '__main__':
    # from load_data import load_data
    # from parameter import parse_args
    # from transformers import RobertaTokenizer
    #
    # arg = parse_args()
    # train_data, dev_data, test_data = load_data(arg)
    # get_batch(train_data, [20, 30, 50, 60], arg, RobertaTokenizer.from_pretrained(arg.model_name))
    # sample_for_CL(train_data, 3, arg, RobertaTokenizer.from_pretrained(arg.model_name))
    pass
