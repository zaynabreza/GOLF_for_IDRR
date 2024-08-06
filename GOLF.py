# -*- coding: utf-8 -*-
"""
Created on Jan 21 2023
@author: JIANG Yuxin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import numpy as np
from CoAttention import MultiHeadAttention
import pickle
from torch_geometric.nn import GCNConv
import scipy.sparse as sp

    
class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x
    

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        
        # BERT encoder
        self.bert = AutoModel.from_pretrained(args.model_name_or_path)
        # for param in self.bert.parameters():
        #     param.requires_grad = not(args.freeze_bert)

        self.layer_norm = nn.LayerNorm(args.config.hidden_size, eps=args.config.layer_norm_eps)
        
        
        # classifier
        print(f"Number of sections (classes): {args.n_sec}")

  
        self.fc_sec = nn.Linear(args.config.hidden_size, args.n_sec)
       
    
    def dice(self, A, B):
        return (2 * len(set(A).intersection(set(B)))) / (len(set(A)) + len(set(B)))
    
    def forward(self, x, mask,y1_sec, arg1_mask, arg2_mask, train=False):
        if train:
            return self.train_forward(x, mask, y1_sec, arg1_mask, arg2_mask)
        else:
            return self.evaluate_forward(x, mask, arg1_mask, arg2_mask)
    
    def evaluate_forward(self, x, mask, arg1_mask, arg2_mask):
        ### BERT encoder
        context = x  # (batch, len)
        bert_out = self.bert(context, attention_mask=mask)
       
        
        pooled = bert_out.last_hidden_state[:, 0, :] # (batch, hidden)
        logits_sec = self.fc_sec(pooled) # (batch, sec)
        
        return logits_sec
    
    def train_forward(self, x, mask, y1_sec, arg1_mask, arg2_mask):
        ### BERT encoder
        # for name,param in self.bert.named_parameters():
        #     print(param.requires_grad,name)
        context = x
        bert_out = self.bert(context, attention_mask=mask)
        

        ### classification loss
        pooled = bert_out.last_hidden_state[:, 0, :] # (batch, hidden)
        logits_sec = self.fc_sec(pooled) # (batch, sec)


        loss_fct = nn.CrossEntropyLoss()
        classification_loss = loss_fct(logits_sec, y1_sec)
    
        
        return logits_sec, classification_loss
    
        
    
    
    
    
    
    
    
    
    
    

    
    
