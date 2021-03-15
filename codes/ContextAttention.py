import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np 
from sklearn import preprocessing
from sklearn.metrics.pairwise import rbf_kernel
from config import args  
def calculate_attention(user, heads, attention_len, index):
    user = user.cpu().numpy()[0]
    dex = index.cpu().numpy()[0]
    if index < 0:
        start = 0
    else:
        start = dex
    end = start + attention_len
    if attention_len != end-start:
        print('len is not equal!')
    tim_scores = args.data_neural[user]['tim_att'][start:end,start:end]
    geo_scores = args.data_neural[user]['geo_att'][start:end,start:end]
    geo_scores = preprocessing.binarize(geo_scores, threshold=50.0)
    np.fill_diagonal(geo_scores, 1)
    geo_scores = 1 - geo_scores
    tim_scores = np.abs(tim_scores ) / 3600.0 / 24
    tim_scores = preprocessing.minmax_scale(tim_scores)
    tim_scores = 1 - tim_scores
    tim_attention = np.tile(tim_scores, (heads, 1))
    geo_attention = np.tile(geo_scores, (heads, 1))
    tim_attention = torch.from_numpy(tim_attention).view(1, heads, attention_len, -1)
    geo_attention = torch.from_numpy(geo_attention).view(1, heads, attention_len, -1)
    #[bt, heads, seqlen, seqlen]
    tim_attention = Variable(tim_attention.type(torch.FloatTensor)).cuda()
    geo_attention = Variable(geo_attention.type(torch.FloatTensor)).cuda()
    return geo_attention, tim_attention

# geo, tim = calculate_attention(torch.LongTensor([0]), 1, 7, torch.LongTensor([10]))
# print(geo)
# print(tim)

def contextattention(q, k, v, user, heads, d_k, index, mask=None, dropout=None):
    #q: [bs * N * sl * d_model]
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    #scores: [bs * N * sl * sl]
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention = F.softmax(scores, dim=-1)
    att_len = attention.size()[-1]
    
    geo_att, tim_att = calculate_attention(user, heads, att_len, index)
    
    # print(geo_att.size())
    # print(attention.size())
    # all_scores = attention
    all_scores = attention #+ 0.8*geo_att#+ 0.8*tim_att#geo_att#- 0.3*geo_att - 0.3*tim_att#  
    all_scores = F.softmax(all_scores, dim=-1)
    if dropout is not None:
        all_scores = dropout(all_scores)
        
    output = torch.matmul(all_scores, v)
    return output

class MultContextSelfAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, user,st_index, mask=None):
        
        bs = q.size(0) #batch
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = contextattention(q, k, v, user, self.h, self.d_k, st_index, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output