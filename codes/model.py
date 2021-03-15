# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy
import numpy as np 
import math

from ContextAttention import MultContextSelfAttention as ContextAttention
  
#######################################
class MyNewTrajPreModel(nn.Module):
    """model"""
    def __init__(self, pars):
        super(MyNewTrajPreModel, self).__init__()
        self.loc_size = pars.loc_size
        self.loc_emb_size = pars.loc_emb_size
        self.hidden_size = pars.hidden_size
        self.heads = pars.head_num
        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.weight = self.emb_loc.weight
        # print('weight:', self.weight.size())
        input_size = self.loc_emb_size      
        #-------------model---------------
        # self.fc = nn.Linear(self.hidden_size, self.loc_size)
        self.c_attn = ContextAttention(self.heads, input_size, pars.dropout_p)
        self.model_type= 'new-attention' #model_type[0]#
        self.init_weights()
        self.dropout = nn.Dropout(p=pars.dropout_p)

    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, loc, tim, user, index):
        loc_emb = self.emb_loc(loc) #[seq,batch,emb_dim]
        att_x = self.dropout(loc_emb)
        #Self-attention
        att_out = att_x.transpose(0,1)
        att_out = self.c_attn(att_out,att_out,att_out, user, index) 
        att_out = att_out.transpose(0,1)
        
        out = att_out.squeeze(1)#out: [seq_len, hidden_size]
        out = F.selu(out)
        out = self.dropout(out)
        y = F.linear(out, self.weight)#[loc_len, C]
        # y = self.fc(out)
        score = F.log_softmax(y,dim=1)  # calculate loss by NLLoss
        return score