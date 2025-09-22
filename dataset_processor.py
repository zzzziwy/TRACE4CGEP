# -*- coding: utf-8 -*-

import torch
import json
import numpy as np
from tokenizers import AddedToken
from transformers import RobertaTokenizer, RobertaForMaskedLM
from util import correct_data, collect_mult_event, replace_mult_event
from load_data import load_vocab

from Handler.tools_handler import getTemplate, getSentence, tokenizerHandler, getposHandler, getStructNeighbour
from Handler.graph_importance_precompute import GraphImportancePrecomputer

class DatasetProcessor(object):
    
    def __init__(self, args, device, tokenizer, encoder):

        self.args = args
        self.device = device
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.original_vocab_size = args.vocab_size
        self.hidden_size = self.encoder.roberta.embeddings.word_embeddings.embedding_dim

        self.multi_event = []
        self.special_multi_event_token = []
        self.event_dict = {}
        self.reverse_event_dict = {}
        self.to_add = {}

        self.train_data = None
        self.dev_data = None
        self.test_data = None

        self.importance_precomputer = None
        
    def process_datasets(self, train_data, dev_data, test_data):

        print("Start processing the dataset...")

        print("Step 1: Correct the data to make the token indices of multi-token events consecutive....")
        self.train_data = correct_data(train_data)
        self.dev_data = correct_data(dev_data)
        self.test_data = correct_data(test_data)

        print("Step 2: Collect the event words and special identifier conversion table...")
        self._collect_multi_token_events()

        print("Step 3: Expand the vocabulary of the tokenizer and the encoder...")
        self._extend_tokenizer_vocab()
        self._extend_encoder_vocab()

        print("Step 4: Replace event tokens with special tokens...")
        self._replace_multi_token_events()

        print("Step 5: Preprocess sample-related information (build temporary sentence encodings for importance computation)...")
        print("train_data processing")
        self._preprocess_sample_info(self.train_data)
        print("dev_data processing")
        self._preprocess_sample_info(self.dev_data)
        print("test_data processing")
        self._preprocess_sample_info(self.test_data)

        print("Step 6: Calculate the importance of graph elements...")
        self._compute_graph_importance()

        print("Step 7: Save the preprocessed dataset...")
        self._save_processed_data()
        
        print("The dataset processing is complete!")
        return self.train_data, self.dev_data, self.test_data
    
    def _collect_multi_token_events(self):
        all_data = self.train_data + self.dev_data + self.test_data
        (self.multi_event, self.special_multi_event_token, 
         self.event_dict, self.reverse_event_dict, self.to_add) = collect_mult_event(all_data, self.tokenizer)
        # 将reverse_event_dict和to_add都存储为json格式。
        import json
        if self.args.graph_enhance:
            with open(f'{self.args.dataset_path}/reverse_event_dict_en.json', 'w', encoding='utf-8') as f:
                json.dump(self.reverse_event_dict, f, ensure_ascii=False, indent=4)
            with open(f'{self.args.dataset_path}/to_add_en.json', 'w', encoding='utf-8') as f:
                json.dump(self.to_add, f, ensure_ascii=False, indent=4)
        else:
            with open(f'{self.args.dataset_path}/reverse_event_dict.json', 'w', encoding='utf-8') as f:
                json.dump(self.reverse_event_dict, f, ensure_ascii=False, indent=4)
            with open(f'{self.args.dataset_path}/to_add.json', 'w', encoding='utf-8') as f:
                json.dump(self.to_add, f, ensure_ascii=False, indent=4)

        print("The data has been successfully saved as a JSON file.")
        print(f"{len(self.multi_event)} event tokens is collected")
        print(f"{len(self.special_multi_event_token)} event tokens is generated")


    
    def _extend_tokenizer_vocab(self):

        for token in self.special_multi_event_token:
            self.tokenizer.add_tokens(
                AddedToken(token, rstrip=True, lstrip=True, single_word=False, normalized=True)
            )

        self.args.vocab_size = len(self.tokenizer)
        print(f"the vocabulary size of tokenizer has been extended: {self.original_vocab_size} -> {self.args.vocab_size}")
    
    def _extend_encoder_vocab(self):
        self.encoder.resize_token_embeddings(self.args.vocab_size)
        da = self.encoder.roberta.embeddings.word_embeddings.weight
        for i in self.to_add.keys():
            l = self.to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).to(self.device)
                for j in l:
                    temp += da[j]
                temp /= len(l)

                da[self.tokenizer.convert_tokens_to_ids(i)] = temp
        print(f"the vocabulary size of encoder has been extended: {self.original_vocab_size} -> {self.args.vocab_size}")

    def _replace_multi_token_events(self):

        self.train_data = replace_mult_event(self.train_data, self.reverse_event_dict)
        self.dev_data = replace_mult_event(self.dev_data, self.reverse_event_dict)
        self.test_data = replace_mult_event(self.test_data, self.reverse_event_dict)
        
        print("event token has been replaced to special token")
    
    def _save_processed_data(self):

        direct_used_data = {
            'train': self.train_data,
            'dev': self.dev_data,
            'test': self.test_data,
            'metadata': {
                'importance_computed': True,
                'importance_version': '1.0',
                'computed_at': str(np.datetime64('now')),
                'total_samples': len(self.train_data) + len(self.dev_data) + len(self.test_data)
            }
        }

        if self.args.graph_enhance:
            np.save(f'{self.args.dataset_path}/direct_used_data_en.npy', direct_used_data)
        else:
            np.save(f'{self.args.dataset_path}/direct_used_data.npy', direct_used_data)
        print("processed data has been saved as npy file")

        metadata_file = f'{self.args.dataset_path}/dataset_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(direct_used_data['metadata'], f, ensure_ascii=False, indent=4)
        print(f"The dataset metadata has been saved to: {metadata_file}")

    def _preprocess_sample_info(self, data):

        for idx in range(len(data)):
            if idx % 300 == 0:
                print(f"Processing sample {idx}/{len(data)}...")
            sentence = getSentence(self.args, self.tokenizer, data[idx], data[idx]['edge'])
            data[idx]['sentence'] = sentence

    def _compute_graph_importance(self):

        print("Start calculating the importance of graph elements...")

        self.importance_precomputer = GraphImportancePrecomputer(encoder=self.encoder, device=self.device)

        print(f"Number of train set samples: {len(self.train_data)}")
        print(f"Number of dev set samples: {len(self.dev_data)}")
        print(f"Number of test set samples: {len(self.test_data)}")

        if self.train_data:
            print("The importance of the train set is being calculated....")
            self._compute_and_store_importance_for_dataset(self.train_data, "train set")
        
        if self.dev_data:
            print("The importance of the dev set is being calculated....")
            self._compute_and_store_importance_for_dataset(self.dev_data, "dev set")
        
        if self.test_data:
            print("The importance of the test set is being calculated....")
            self._compute_and_store_importance_for_dataset(self.test_data, "test set")
        
        print("Calculation of the importance of graphic elements has been completed")
    
    def _compute_and_store_importance_for_dataset(self, dataset, dataset_name):

        print(f"Importance information is being calculated and stored for {dataset_name}...")
        
        for idx, sample in enumerate(dataset):
            if idx % 100 == 0:
                print(f"Progress of {dataset_name}: {idx}/{len(dataset)}")

            graph_ctx = self.importance_precomputer._build_graph_context(sample)

            node_importance = self.importance_precomputer._compute_node_importance(sample, graph_ctx, self.tokenizer, self.args)

            edge_importance = self.importance_precomputer._compute_edge_importance(sample, graph_ctx, self.tokenizer, self.args)
            

            mask_node_indices = [graph_ctx['mask_node_id']] if graph_ctx['mask_node_id'] is not None else []
            node_drop_prob = self.importance_precomputer._compute_deactivation_probability(
                node_importance, self.importance_precomputer.config.node_deactivate_rate, mask_node_indices
            )
            node_drop_prob[graph_ctx['anchor_node_id']] = 0.0
            node_drop_prob[graph_ctx['mask_node_id']] = 0.0
            mask_edge_indices = list(graph_ctx['mask_connected_edges']) if 'mask_connected_edges' in graph_ctx else []
            edge_drop_prob = self.importance_precomputer._compute_deactivation_probability(
                edge_importance, self.importance_precomputer.config.edge_deactivate_rate, mask_edge_indices
            )
            edge_drop_prob[-1] = 0.0

            sample['node_importance'] = node_importance
            sample['edge_importance'] = edge_importance
            sample['deactivation_prob'] = {
                'node_drop_prob': node_drop_prob,
                'edge_drop_prob': edge_drop_prob
            }
            sample['importance_metadata'] = {
                'has_importance': True,
                'importance_version': '1.0',
                'computed_at': str(np.datetime64('now'))
            }

            if 'sentence' in sample:
                try:
                    del sample['sentence']
                except Exception:
                    sample['sentence'] = None
        
        print(f"The importance information of {dataset_name} has been calculated and stored successfully!")


    def load_processed_data(self):
        try:
            if self.args.graph_enhance:
                direct_used_data = np.load(f'{self.args.dataset_path}/direct_used_data_en.npy', allow_pickle=True).item()
            else:
                direct_used_data = np.load(f'{self.args.dataset_path}/direct_used_data.npy', allow_pickle=True).item()

            self.train_data = direct_used_data['train']
            self.dev_data = direct_used_data['dev']
            self.test_data = direct_used_data['test']

            self.reverse_event_dict, self.to_add = load_vocab(self.args)
            self.special_multi_event_token = [self.reverse_event_dict[k] for k in self.reverse_event_dict.keys()]

            print(f"Adding {len(self.special_multi_event_token)} event tokens to tokenizer")
            for token in self.special_multi_event_token:
                self.tokenizer.add_tokens(AddedToken(token, rstrip=True, lstrip=True, single_word=False, normalized=True))

            self.args.vocab_size = len(self.tokenizer)
            
            return self.train_data, self.dev_data, self.test_data, self.to_add, self.tokenizer
        except FileNotFoundError:
            print("The processed data file was not found. Please run the \"process_datasets\" method first.")
            return None, None, None, None, None


    
    def check_importance_info(self, sample=None):
        if sample is not None:
            has_importance = 'node_importance' in sample and 'edge_importance' in sample and 'deactivation_prob' in sample
            return {
                'has_importance': has_importance,
                'sample_id': id(sample),
                'node_importance_shape': sample.get('node_importance', {}).shape if has_importance else None,
                'edge_importance_shape': sample.get('edge_importance', {}).shape if has_importance else None,
                'deactivation_prob_keys': list(sample.get('deactivation_prob', {}).keys()) if has_importance else None
            }

        results = {}
        for dataset_name, dataset in [('train', self.train_data), ('dev', self.dev_data), ('test', self.test_data)]:
            if dataset is None:
                results[dataset_name] = {'status': 'not_loaded', 'has_importance': False}
                continue

            check_samples = min(5, len(dataset))
            sample_checks = []
            for i in range(check_samples):
                sample = dataset[i]
                sample_check = self.check_importance_info(sample)
                sample_checks.append(sample_check)

            has_importance_count = sum(1 for check in sample_checks if check['has_importance'])
            results[dataset_name] = {
                'status': 'loaded',
                'total_samples': len(dataset),
                'checked_samples': check_samples,
                'has_importance_count': has_importance_count,
                'has_importance_ratio': has_importance_count / check_samples if check_samples > 0 else 0,
                'sample_checks': sample_checks
            }
        
        return results
