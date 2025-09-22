# coding: UTF-8
import torch
import torch.nn as nn
import sys
from transformers import RobertaForMaskedLM, BertForMaskedLM
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
embedding_size = 768


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.roberta_model = RobertaForMaskedLM.from_pretrained(args.model_name).to(device)
        self.roberta_model.resize_token_embeddings(args.vocab_size)
        for param in self.roberta_model.parameters():
            param.requires_grad = True

        self.robert_text = deepcopy(self.roberta_model)
        for param in self.robert_text.parameters():
            param.requires_grad = True

        self.hidden_size = 768

        self.vocab_size = args.vocab_size


    def forward(self, mode, batch_arg, arg_mask, arg_coevent, arg_causal, event_tokenizer_pos, event_key_pos, mask_indices, sentences, batch_size):

        for i in range(batch_size):
            for k in sentences[i]:
                sent_emb = self.robert_text.roberta(input_ids=sentences[i][k]['input_ids'], attention_mask=sentences[i][k]['attention_mask'])[0].to(device)
                sentences[i][k]['emb'] = sent_emb[0][sentences[i][k]['position']]

        word_emb = self.roberta_model.roberta.embeddings.word_embeddings(batch_arg).to(device)

        for i in range(batch_size):
            for j in range(len(event_tokenizer_pos[i])):
                sent_emb = (sentences[i][event_key_pos[i][j]]['emb']).clone().unsqueeze(0)
                word_emb[i][event_tokenizer_pos[i][j]] = sent_emb
                assert str(int(batch_arg[i][event_tokenizer_pos[i][j]])) in event_key_pos[i][j]

        temp_emb = self.roberta_model.roberta(attention_mask=arg_mask, inputs_embeds=word_emb,
                                              co_node_idx=arg_coevent, minibatch_pos=arg_causal)[0].to(device)
        anchor_mask = temp_emb[0][mask_indices]

        prediction = self.roberta_model.lm_head(anchor_mask)
        return prediction


    def handler(self, to_add, tokenizer):
        da = self.roberta_model.roberta.embeddings.word_embeddings.weight
        for i in to_add.keys():
            l = to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).to(device)
                for j in l:
                    temp += da[j]
                temp /= len(l)

                da[tokenizer.convert_tokens_to_ids(i)] = temp
