import torch
import numpy as np
import random
import networkx as nx
from scipy.spatial.distance import cosine

from .util_handler import find_distances, getDistance, find_paths, getAdja
from .graph_deactivation_config import default_config
def getTemplate(args, data, mode):
    edge = data['edge'][:-1] if len(data['edge'])<=(args.len_arg_template)//5 else data['edge'][0:(args.len_arg_template)//5]
    # random.shuffle(edge)
    causeRel = edge[0:len(edge)]
    template = ''
    relation = [] + causeRel
    assert data['edge'][-1] not in relation

    distance = getDistance(relation+[data['edge'][-1]])
    assert len(relation)+1==len(distance)

    weighted_characters = list(zip(distance[:-1], relation))
    if mode == "sequential":
        sorted_characters = sorted(weighted_characters, reverse=True)
    elif mode == "reverse":
        sorted_characters = sorted(weighted_characters, reverse=False)
    else:
        random.seed(args.seed)
        sorted_characters = weighted_characters.copy()
        random.shuffle(sorted_characters)

    sorted_relation_only = [char for weight, char in sorted_characters]



    for rel in sorted_relation_only:
        eId1 = rel[0]
        eId2 = rel[-1]
        rl = f"{data['node'][eId1][5]} {rel[1]} {data['node'][eId2][5]}"
        template = template + rl + ' '
    maskRel = data['edge'][-1]
    template = template + f"{data['node'][maskRel[0]][5]} {maskRel[1]} <mask>"
    return template, sorted_relation_only + [maskRel]

def getSentence(args, tokenizer, data, relation):
    sentence = {}
    for idx, node in enumerate(data['node']):
        if idx not in sentence.keys():
            sentence[idx] = data['node'][idx][6]
    sentence[relation[-1][-1]] = '<mask>'
    sentTokenizer = {}
    for e in sentence.keys():
        sent_dict = tokenizer.encode_plus(
                sentence[e],
                add_special_tokens=True,
                padding='max_length',
                max_length=args.len_arg_text,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
        event = data['node'][e][5]
        event_id = tokenizer.encode(event)[1]
        input_ids = sent_dict['input_ids'][0].tolist()
        if sentence[e] == '<mask>':
            position = 1
        else:
            position = input_ids.index(event_id)
        sentTokenizer[str(e)+'_'+str(tokenizer.encode(event)[1])] = {'input_ids':      sent_dict['input_ids'],
                                                                     'attention_mask': sent_dict['attention_mask'],
                                                                     'position':       position}
    return sentTokenizer



def assertHandler(index_list):
    assert index_list[0][0] == 2
    for i in range(len(index_list)-1):
        assert index_list[i+1][0] - index_list[i][0] == 3
    return
def getStructNeighbour(arg_1_idx, rel_id, flag):
    zero_pos = torch.nonzero(arg_1_idx == 0, as_tuple=False)[0][1].item()  # [CLS]
    two_pos = torch.nonzero(arg_1_idx == 2, as_tuple=False)[0][1].item()  # [EOS]
    temp_arg = arg_1_idx[0][zero_pos : two_pos+1]
    index_list = torch.nonzero(temp_arg == rel_id, as_tuple=False).tolist()
    assertHandler(index_list)
    arg_mask = np.zeros((two_pos+1, two_pos+1))
    special_token_id = [0, 2]
    unk_token_id = 3
    # <CLS> <SEP> <mask>
    for i in range(len(temp_arg)):
        # PLM-specific token
        if temp_arg[i] in special_token_id:
            arg_mask[i, :] = 1
        # relation token
        elif temp_arg[i] == rel_id:
            for j in range(len(temp_arg)):
                if temp_arg[j] in special_token_id:
                    arg_mask[i, j] = 1
            arg_mask[i, i] = 1
            arg_mask[i, i-1] = 1
            arg_mask[i, i+1] = 1
        # event token
        else:
            for j in range(len(temp_arg)):
                if i == j or temp_arg[j] in special_token_id:
                    arg_mask[i, j] = 1
            mid_rel_pos = index_list[(i-1) // 3][0]
            arg_mask[i, mid_rel_pos] = 1
            if flag == 'struct':
                if i > mid_rel_pos:
                    arg_mask[i, mid_rel_pos - 1] = 1
                else:
                    # arg_mask[i, mid_rel_pos + 1] = 1
                    continue
            else:
                if i > mid_rel_pos:
                    arg_mask[i, mid_rel_pos - 1] = 1
                else:
                    arg_mask[i, mid_rel_pos + 1] = 1
    final_mask = np.zeros((arg_1_idx.shape[1], arg_1_idx.shape[1]))
    final_mask[:two_pos+1, :two_pos+1] = arg_mask
    return torch.tensor(final_mask, dtype = arg_1_idx.dtype).unsqueeze(dim=0)
def tokenizerHandler(args, template, tokenizer):
    encode_dict = tokenizer.encode_plus(
        template,
        add_special_tokens=True,
        padding='max_length',
        max_length=args.len_arg_template,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    arg_1_idx = encode_dict['input_ids']
    arg_1_mask = encode_dict['attention_mask']

    arg_1_coevent = arg_1_idx.clone()
    mask = (arg_1_idx == 0) | (arg_1_idx == 1) | (arg_1_idx == 2) | (arg_1_idx == 27037)
    arg_1_coevent[mask] = -1

    zero_pos = torch.nonzero(arg_1_idx == 0, as_tuple=True)
    two_pos = torch.nonzero(arg_1_idx == 2, as_tuple=True)
    padding_idx = args.padding_idx
    arg_1_causal = torch.full_like(arg_1_idx, -padding_idx)
    for i in range(args.batch_size):
        zp = zero_pos[-1][i].item()
        tp = two_pos[-1][i].item()


        num_groups = (tp - zp - 1) // 3

        group_indices = torch.tile(torch.tensor([-padding_idx + 2, -padding_idx + 3, -padding_idx + 4]), (num_groups, ))

        arg_1_causal[i, zp + 1: tp] = group_indices
        arg_1_causal[i, zp] = -padding_idx + 1
        arg_1_causal[i, tp] = -padding_idx + 5

    return arg_1_idx, arg_1_mask, arg_1_coevent, arg_1_causal

def getposHandler(arg_idx, relation, tokenizer):
    tempPosition = torch.nonzero(arg_idx >= tokenizer.encode('<a_0>')[1]).tolist()
    ePosition = [row[-1] for row in tempPosition]
    ePositionKey = []
    sentId = []
    for rel in relation:
        sentId.append(rel[0])
        sentId.append(rel[-1])
    assert len(sentId) - 1 == len(ePosition)  # sentId 比 ePosition 多计数了一个 <mask>

    for iid in range(len(ePosition)):
        ePositionKey.append(str(sentId[iid]) + '_' + str(arg_idx[0][ePosition[iid]].item()))

    return ePosition, ePositionKey

def apply_graph_deactivation(sample, args, sample_idx):


    mask_edge = sample['edge'][-1]
    mask_edge_id = len(sample['edge'])-1
    mask_node_id = mask_edge[-1]
    anchor_node_id = mask_edge[0]

    protected_nodes_id = {anchor_node_id, mask_node_id}

    protected_edges_id = {mask_edge_id}


    temp_nodes = list(range(len(sample['node'])))
    temp_edges = list(sample['edge'])

    active_nodes = set(temp_nodes)
    if sample['deactivation_prob']['node_drop_prob'] is not None:
        rng_seed = getattr(args, 'seed', 209) + sample_idx
        rng = np.random.RandomState(rng_seed)
        
        for i in range(len(sample['node'])):
            if i not in protected_nodes_id:
                drop_prob = sample['deactivation_prob']['node_drop_prob'][i]
                if rng.random() < drop_prob:
                    active_nodes.discard(i)
                    temp_edges = [edge for edge in temp_edges if not (len(edge) >= 2 and (edge[0] == i or edge[-1] == i))]

    if sample['deactivation_prob']['edge_drop_prob'] is not None:
        rng_seed = getattr(args, 'seed', 209) + sample_idx
        rng = np.random.RandomState(rng_seed)

        edge_indices_in_temp = []
        for i, edge in enumerate(sample['edge']):
            if edge in temp_edges:
                edge_indices_in_temp.append(i)

        new_temp_edges = []
        for i, edge in enumerate(temp_edges):
            original_edge_index = edge_indices_in_temp[i]
            if original_edge_index not in protected_edges_id:
                drop_prob = sample['deactivation_prob']['edge_drop_prob'][original_edge_index]
                if rng.random() < drop_prob:
                    pass
                else:
                    new_temp_edges.append(edge)
            else:
                new_temp_edges.append(edge)
        
        temp_edges = new_temp_edges

    temp_sample = {
        'node': [sample['node'][i] for i in active_nodes] if len(active_nodes) < len(sample['node']) else sample['node'],
        'edge': temp_edges,
        'candiSet': sample['candiSet'],
        'label': sample['label']
    }

    if len(active_nodes) < len(sample['node']):
        old_to_new_indices = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(active_nodes))}

        updated_edges = []
        for edge in temp_sample['edge']:
            if len(edge) >= 2:
                if edge[0] in old_to_new_indices and edge[-1] in old_to_new_indices:
                    new_edge = list(edge)
                    new_edge[0] = old_to_new_indices[edge[0]]
                    new_edge[-1] = old_to_new_indices[edge[-1]]
                    updated_edges.append(new_edge)
            else:
                updated_edges.append(edge)
        temp_sample['edge'] = updated_edges
    
    return temp_sample
