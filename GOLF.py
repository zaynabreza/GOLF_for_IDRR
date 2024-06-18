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
        for param in self.bert.parameters():
            param.requires_grad = not(args.freeze_bert)
        
        # dual multi head attention
        self.co_attention_layer_1 = MultiHeadAttention(
                                n_head=args.config.num_attention_heads, 
                                d_model=args.config.hidden_size, 
                                d_k=(args.config.hidden_size // args.config.num_attention_heads), 
                                d_v=(args.config.hidden_size // args.config.num_attention_heads), 
                                dropout=args.config.attention_probs_dropout_prob
                              )
        self.co_attention_layer_2 = MultiHeadAttention(
                                n_head=args.config.num_attention_heads, 
                                d_model=args.config.hidden_size, 
                                d_k=(args.config.hidden_size // args.config.num_attention_heads), 
                                d_v=(args.config.hidden_size // args.config.num_attention_heads), 
                                dropout=args.config.attention_probs_dropout_prob
                              )
        self.layer_norm = nn.LayerNorm(args.config.hidden_size, eps=args.config.layer_norm_eps)
        
        
        # classifier
        self.fc_top = nn.Linear(args.config.hidden_size, args.n_top)
        self.fc_sec = nn.Linear(args.config.hidden_size + args.n_top, args.n_sec)
        self.fc_conn = nn.Linear(args.config.hidden_size + args.n_sec, args.n_conn)
    
    def dice(self, A, B):
        return (2 * len(set(A).intersection(set(B)))) / (len(set(A)) + len(set(B)))
    
    def forward(self, x, mask, y1_top, y1_sec, y1_conn, arg1_mask, arg2_mask, train=False):
        if train:
            return self.train_forward(x, mask, y1_top, y1_sec, y1_conn, arg1_mask, arg2_mask)
        else:
            return self.evaluate_forward(x, mask, arg1_mask, arg2_mask)
    
    def evaluate_forward(self, x, mask, arg1_mask, arg2_mask):
        ### BERT encoder
        context = x  # (batch, len)
        bert_out = self.bert(context, attention_mask=mask)
        
        
        ### dual multi-head attention
        arg1_mask = arg1_mask[:, None, None, :]
        arg2_mask = arg2_mask[:, None, None, :]
        
        hidden_last = bert_out.last_hidden_state
        for i in range(self.args.num_co_attention_layer):
            arg2_hidden_last, _ = self.co_attention_layer_1(q=hidden_last, 
                                                        k=hidden_last, 
                                                        v=hidden_last, 
                                                        mask=arg1_mask)
            arg1_hidden_last, _ = self.co_attention_layer_2(q=hidden_last, 
                                                            k=hidden_last, 
                                                            v=hidden_last, 
                                                            mask=arg2_mask)
            updated_hidden_last = (arg1_hidden_last * arg1_mask.squeeze().unsqueeze(dim=-1)) \
                                + (arg2_hidden_last * arg2_mask.squeeze().unsqueeze(dim=-1))
            hidden_last = self.layer_norm(updated_hidden_last) # (batch, seq_len, hidden)
        
        
        ### classifier
        pooled = hidden_last[:, 0, :] # (batch, hidden)
        logits_top = self.fc_top(pooled) # (batch, top)
        logits_sec = self.fc_sec(torch.cat([pooled, logits_top], dim=-1)) # (batch, sec)
        logits_conn = self.fc_conn(torch.cat([pooled, logits_sec], dim=-1)) # (batch, conn)
        
        return logits_top, logits_sec, logits_conn
    
    def train_forward(self, x, mask, y1_top, y1_sec, y1_conn, arg1_mask, arg2_mask):
        ### BERT encoder
        bs = x.shape[0]
        context = torch.cat([x, x], dim=0)  # (batch*2, len)
        mask = torch.cat([mask, mask], dim=0) # (batch*2, len)
        bert_out = self.bert(context, attention_mask=mask)
        
        
        ### dual multi-head attention
        arg1_mask = torch.cat([arg1_mask, arg1_mask], dim=0)[:, None, None, :]
        arg2_mask = torch.cat([arg2_mask, arg2_mask], dim=0)[:, None, None, :]
        
        hidden_last = bert_out.last_hidden_state
        for i in range(self.args.num_co_attention_layer):
            arg2_hidden_last, _ = self.co_attention_layer_1(q=hidden_last, 
                                                        k=hidden_last, 
                                                        v=hidden_last, 
                                                        mask=arg1_mask)
            arg1_hidden_last, _ = self.co_attention_layer_2(q=hidden_last, 
                                                            k=hidden_last, 
                                                            v=hidden_last, 
                                                            mask=arg2_mask)
            updated_hidden_last = (arg1_hidden_last * arg1_mask.squeeze().unsqueeze(dim=-1)) \
                                + (arg2_hidden_last * arg2_mask.squeeze().unsqueeze(dim=-1))
            hidden_last = self.layer_norm(updated_hidden_last) # (batch*2, seq_len, hidden)
      
       
        
    
        ### classification loss
        pooled = hidden_last[:, 0, :] # (batch, hidden)
        logits_top = self.fc_top(pooled) # (batch, top)
        logits_sec = self.fc_sec(torch.cat([pooled, logits_top], dim=-1)) # (batch, sec)
        logits_conn = self.fc_conn(torch.cat([pooled, logits_sec], dim=-1)) # (batch, conn)

        # Adjust target sizes by repeating
        y1_top = y1_top.repeat(2)
        y1_sec = y1_sec.repeat(2)
        y1_conn = y1_conn.repeat(2)
    
        loss_fct = nn.CrossEntropyLoss()
        classification_loss = loss_fct(logits_top, y1_top) \
                            + loss_fct(logits_sec, y1_sec) \
                            + loss_fct(logits_conn, y1_conn)
        
        
        loss = classification_loss
        
        return logits_top, logits_sec, logits_conn, loss
    
        
    
    
    
    
    
    
    
    
    
    

    
    
