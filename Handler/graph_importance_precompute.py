# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from .graph_deactivation_config import default_config
import torch
class GraphImportancePrecomputer:
    
    def __init__(self, encoder, device, config=None):
        self.config = config or default_config
        self.encoder = encoder  # 编码器，用于计算语义特征
        self.hidden_size = self.encoder.roberta.embeddings.word_embeddings.embedding_dim
        self.device = device  # 设备，用于控制计算位置
        self.node_importance_cache = {}  # 缓存节点重要性
        self.edge_importance_cache = {}  # 缓存边重要性
        self.deactivation_prob_cache = {}  # 缓存失活概率
        
    
    def _build_graph_context(self, sample):

        anchor_node_id = sample['edge'][-1][0]
        mask_node_id = sample['edge'][-1][-1]
        

        num_nodes = len(sample['node'])
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for edge in sample['edge']:
            if len(edge) >= 2:
                adj_matrix[edge[0], edge[-1]] = 1
        

        G = nx.DiGraph()
        for edge in sample['edge']:
            if len(edge) >= 2:
                G.add_edge(edge[0], edge[-1])

        node_to_mask_distances = {}
        if mask_node_id is not None and mask_node_id in G.nodes():
            try:
                G_undirected = G.to_undirected()
                distances = nx.shortest_path_length(G_undirected, target=mask_node_id)
                for node in range(num_nodes):
                    node_to_mask_distances[node] = distances.get(node, float('inf'))
            except nx.NetworkXNoPath:
                for node in range(num_nodes):
                    node_to_mask_distances[node] = float('inf')
        else:
            for node in range(num_nodes):
                node_to_mask_distances[node] = float('inf')
        

        edge_to_mask_distances = {}

        mask_connected_edges = set()
        for i, edge in enumerate(sample['edge']):
            if len(edge) >= 2:
                dist1 = node_to_mask_distances.get(edge[0], float('inf'))
                dist2 = node_to_mask_distances.get(edge[-1], float('inf'))
                edge_to_mask_distances[i] = min(dist1, dist2)
                

                if mask_node_id is not None and (edge[0] == mask_node_id or edge[-1] == mask_node_id):
                    mask_connected_edges.add(i)
        
        return {
            'mask_node_id': mask_node_id,
            'anchor_node_id': anchor_node_id,
            'num_nodes': num_nodes,
            'num_edges': len(sample['edge']),
            'adj_matrix': adj_matrix,
            'graph': G,
            'node_to_mask_distances': node_to_mask_distances,
            'edge_to_mask_distances': edge_to_mask_distances,
            'node_indices': list(range(num_nodes)),
            'edge_indices': list(range(len(sample['edge']))),
            'mask_connected_edges': mask_connected_edges
        }
    
    def _compute_node_importance(self, sample, graph_ctx, tokenizer, args):

        num_nodes = graph_ctx['num_nodes']
        G = graph_ctx['graph']
        node_to_mask_distances = graph_ctx['node_to_mask_distances']
        mask_node_id = graph_ctx['mask_node_id']  # 获取掩码节点ID
        

        temp_G = G.copy()
        if mask_node_id is not None and mask_node_id in temp_G.nodes():
            temp_G.remove_node(mask_node_id)
        

        non_mask_nodes = [node for node in range(num_nodes) if node != mask_node_id]
        node_index_map = {node: idx for idx, node in enumerate(non_mask_nodes)}
        num_non_mask_nodes = len(non_mask_nodes)
        

        topology_importance = np.zeros(num_non_mask_nodes, dtype=np.float32)
        semantic_importance = np.zeros(num_non_mask_nodes, dtype=np.float32)
        prediction_importance = np.zeros(num_non_mask_nodes, dtype=np.float32)
        

        if temp_G.number_of_nodes() > 0:
            if self.config.use_betweenness_centrality:
                try:
                    betweenness_centrality = nx.betweenness_centrality(temp_G, normalized=True)
                    for node in non_mask_nodes:
                        if node in temp_G.nodes():
                            idx = node_index_map[node]
                            topology_importance[idx] = betweenness_centrality.get(node, 0.0)

                except:

                    if self.config.use_degree_centrality:
                        try:
                            degree_centrality = nx.degree_centrality(temp_G)
                            for node in non_mask_nodes:
                                if node in temp_G.nodes():
                                    idx = node_index_map[node]
                                    topology_importance[idx] = degree_centrality.get(node, 0.0)
                        except:
                            for node in temp_G.nodes():
                                if node in non_mask_nodes:
                                    idx = node_index_map[node]
                                    topology_importance[idx] = temp_G.degree(node) / max(1, temp_G.number_of_nodes() - 1)
                    else:
                        for node in temp_G.nodes():
                            if node in non_mask_nodes:
                                idx = node_index_map[node]
                                topology_importance[idx] = temp_G.degree(node) / max(1, temp_G.number_of_nodes() - 1)
        

        event_representations = self._get_event_representations(sample)

        non_mask_event_representations = torch.zeros(num_non_mask_nodes, self.hidden_size, dtype=torch.float32, device=self.device)
        for node in non_mask_nodes:
            idx = node_index_map[node]
            non_mask_event_representations[idx] = event_representations[node]
        
        semantic_importance = self._compute_semantic_importance_with_subset(graph_ctx, non_mask_event_representations, non_mask_nodes, node_index_map)

        non_mask_distances = {node: dist for node, dist in node_to_mask_distances.items() if node != mask_node_id}
        max_distance = max(non_mask_distances.values()) if non_mask_distances else 1.0
        if max_distance > 0 and max_distance != float('inf'):
            for node in non_mask_nodes:
                idx = node_index_map[node]
                distance = node_to_mask_distances.get(node, max_distance)
                if distance != float('inf'):
                    prediction_importance[idx] = 1.0 / (1.0 + distance)
                else:
                    prediction_importance[idx] = 0.1
        

        max_val = np.max(topology_importance)
        min_val = np.min(topology_importance)
        if max_val > min_val:
            topology_importance = (topology_importance - min_val) / (max_val - min_val)
        

        max_val = np.max(semantic_importance)
        min_val = np.min(semantic_importance)
        if max_val > min_val:
            semantic_importance = (semantic_importance - min_val) / (max_val - min_val)
        

        max_val = np.max(prediction_importance)
        min_val = np.min(prediction_importance)
        if max_val > min_val:
            prediction_importance = (prediction_importance - min_val) / (max_val - min_val)
        

        non_mask_node_importance = (self.config.topology_weight * topology_importance + 
                                   self.config.semantic_weight * semantic_importance + 
                                   self.config.prediction_weight * prediction_importance)
        

        node_importance = np.zeros(num_nodes, dtype=np.float32)
        for node in non_mask_nodes:
            idx = node_index_map[node]
            node_importance[node] = non_mask_node_importance[idx]

        if mask_node_id is not None:
            node_importance[mask_node_id] = 1.0
            
        return node_importance
    
    def _compute_edge_importance(self, sample, graph_ctx, tokenizer, args):

        num_edges = graph_ctx['num_edges']
        G = graph_ctx['graph']
        edge_to_mask_distances = graph_ctx['edge_to_mask_distances']
        mask_node_id = graph_ctx['mask_node_id']
        

        mask_connected_edges = graph_ctx['mask_connected_edges']
        

        non_mask_edges = [i for i in range(num_edges) if i not in mask_connected_edges]
        edge_index_map = {edge: idx for idx, edge in enumerate(non_mask_edges)}
        num_non_mask_edges = len(non_mask_edges)
        

        topology_importance = np.zeros(num_non_mask_edges, dtype=np.float32)
        prediction_importance = np.zeros(num_non_mask_edges, dtype=np.float32)
        

        temp_G = G.copy()
        if mask_node_id is not None and mask_node_id in temp_G.nodes():
            temp_G.remove_node(mask_node_id)
        

        if temp_G.number_of_nodes() > 0 and temp_G.number_of_edges() > 0:
            if self.config.use_path_coverage:
                try:
                    all_pairs_shortest_paths = dict(nx.all_pairs_shortest_path_length(temp_G))
                    
                    for i, edge in enumerate(sample['edge'][:num_edges]):
                        if i in mask_connected_edges:
                            continue
                            
                        if len(edge) >= 2:
                            source, target = edge[0], edge[-1]
                            if temp_G.has_edge(source, target):
                                edge_path_count = 0
                                total_possible_paths = 0
                                
                                for start_node in temp_G.nodes():
                                    for end_node in temp_G.nodes():
                                        if start_node != end_node:
                                            try:
                                                if start_node in all_pairs_shortest_paths and end_node in all_pairs_shortest_paths[start_node]:
                                                    path_length = all_pairs_shortest_paths[start_node][end_node]
                                                    if path_length > 1:
                                                        total_possible_paths += 1
                                                        try:
                                                            shortest_path = nx.shortest_path(temp_G, start_node, end_node)
                                                            for j in range(len(shortest_path) - 1):
                                                                if (shortest_path[j] == source and shortest_path[j+1] == target) or \
                                                                   (shortest_path[j] == target and shortest_path[j+1] == source):
                                                                    edge_path_count += 1
                                                                    break
                                                        except:
                                                            continue
                                            except:
                                                continue
                                
                                if total_possible_paths > 0:
                                    idx = edge_index_map[i]
                                    topology_importance[idx] = edge_path_count / total_possible_paths
                except:
                    for i, edge in enumerate(sample['edge'][:num_edges]):

                        if i in mask_connected_edges:
                            continue
                            
                        if len(edge) >= 2:
                            source, target = edge[0], edge[-1]
                            if temp_G.has_edge(source, target):
                                source_degree = temp_G.degree(source)
                                target_degree = temp_G.degree(target)
                                idx = edge_index_map[i]
                                topology_importance[idx] = (source_degree + target_degree) / (2 * max(1, temp_G.number_of_nodes() - 1))

        non_mask_distances = {i: dist for i, dist in edge_to_mask_distances.items() if i not in mask_connected_edges}
        max_distance = max(non_mask_distances.values()) if non_mask_distances else 1.0
        if max_distance > 0 and max_distance != float('inf'):
            for i in range(num_edges):
                if i in mask_connected_edges:
                    continue
                    
                distance = edge_to_mask_distances.get(i, max_distance)
                if distance != float('inf'):
                    idx = edge_index_map[i]
                    prediction_importance[idx] = 1.0 / (1.0 + distance)
                else:
                    idx = edge_index_map[i]
                    prediction_importance[idx] = 0.1

        max_val = np.max(topology_importance)
        min_val = np.min(topology_importance)
        if max_val > min_val:
            topology_importance = (topology_importance - min_val) / (max_val - min_val)

        max_val = np.max(prediction_importance)
        min_val = np.min(prediction_importance)
        if max_val > min_val:
            prediction_importance = (prediction_importance - min_val) / (max_val - min_val)

        non_mask_edge_importance = (self.config.topology_weight * topology_importance + 
                                   self.config.prediction_weight * prediction_importance)
        

        edge_importance = np.zeros(num_edges, dtype=np.float32)
        for i in non_mask_edges:
            idx = edge_index_map[i]
            edge_importance[i] = non_mask_edge_importance[idx]

        for i in mask_connected_edges:
            edge_importance[i] = 1.0
            
        return edge_importance
    
    def _get_event_representations(self, sample):
        event_representations = torch.zeros(len(sample['node']), self.hidden_size, dtype=torch.float32, device=self.device)

        sentence_data = sample['sentence']
        
        for node_id in range(len(sample['node'])):
            node_key = None
            for key in sentence_data.keys():
                if key.startswith(f"{node_id}_"):
                    node_key = key
                    break

            node_sentence = sentence_data[node_key]
            input_ids = node_sentence['input_ids']
            attention_mask = node_sentence['attention_mask']
            event_position = node_sentence['position']

            with torch.no_grad():
                target_device = self.device
                input_ids = input_ids.to(target_device)
                attention_mask = attention_mask.to(target_device)
                sent_emb = self.encoder.roberta(input_ids=input_ids, attention_mask=attention_mask)[0]
                event_vector = sent_emb[0][event_position]

                event_representations[node_id] = event_vector
        return event_representations
    
    def _compute_semantic_importance(self, graph_ctx, event_representations):
        num_nodes = graph_ctx['num_nodes']
        semantic_importance = np.zeros(num_nodes, dtype=np.float32)
        

        if event_representations is not None and event_representations.shape[0] > 0:
            avg_representation = torch.mean(event_representations, dim=0)

            for node_id in range(num_nodes):
                node_repr = event_representations[node_id]
                similarity = torch.cosine_similarity(node_repr.unsqueeze(0), avg_representation.unsqueeze(0), dim=1)
                semantic_importance[node_id] = max(0.0, min(1.0, similarity.item()))
        
        return semantic_importance
    
    def _compute_semantic_importance_with_subset(self, graph_ctx, event_representations, node_list, node_index_map):

        num_nodes = len(node_list)
        semantic_importance = np.zeros(num_nodes, dtype=np.float32)

        if event_representations is not None and event_representations.shape[0] > 0:

            avg_representation = torch.mean(event_representations, dim=0)

            for i in range(num_nodes):
                node_repr = event_representations[i]
                similarity = torch.cosine_similarity(node_repr.unsqueeze(0), avg_representation.unsqueeze(0), dim=1)
                semantic_importance[i] = max(0.0, min(1.0, similarity.item()))
        
        return semantic_importance
    
    def _compute_deactivation_probability(self, importance, base_rate, mask_indices=None):

        size = len(importance)
        drop_prob = np.zeros(size, dtype=np.float32)

        if mask_indices is None:
            mask_indices = []

        non_mask_indices = [i for i in range(size) if i not in mask_indices]
        
        if non_mask_indices:
            non_mask_importance = importance[non_mask_indices]
            max_importance = np.max(non_mask_importance)

            for i in non_mask_indices:
                if importance[i] > 0:
                    prob = base_rate * (1-(importance[i]/max_importance)**self.config.importance_alpha)
                    drop_prob[i] = min(1.0, max(0.0, prob))
                else:
                    drop_prob[i] = base_rate

        return drop_prob
