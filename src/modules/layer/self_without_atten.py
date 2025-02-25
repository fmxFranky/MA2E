## Version : 1.0 (without actions)

import torch.nn as nn
import torch.nn.functional as F
import torch 
import time

class Self_Without_Attention(nn.Module):
    def __init__(self, input_dim,heads,output_dim):
        super(Self_Without_Attention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim*heads)
        self.key = nn.Linear(input_dim, input_dim*heads)
        self.value = nn.Linear(input_dim, input_dim*heads)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x) :
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        #scores = torch.bmm(queries[:,1:], keys[:,0:1].transpose(1, 2)) / (self.input_dim ** 0.5)
        #attention = self.softmax(scores)
        #weighted = torch.bmm(attention.transpose(1,2), values[:,1:])
        
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)

        return attention, weighted

'''
class Self_Without_Attention(nn.Module):
    def __init__(self, input_size, heads, embed_size):
        super().__init__()
        self.input_size = input_size
        self.heads = heads
        self.emb_size = embed_size

        self.tokeys = nn.Linear(self.input_size, self.emb_size * heads, bias = False)
        self.toqueries = nn.Linear(self.input_size, self.emb_size * heads, bias = False)
        self.tovalues = nn.Linear(self.input_size, self.emb_size * heads, bias = False)
        
    def forward(self, x,agent_num,index):
        b, t, hin = x.size()
        assert hin == self.input_size, f'Input size {{hin}} should match {{self.input_size}}'
        
        h = self.heads 
        e = self.emb_size
        
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # dot-product attention
        # folding heads to batch dimensions
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4))
        keys = keys / (e ** (1/4))
        
        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b*h, t, t)

        # row wise self attention probabilities
        ### without self values
        dot[index,:,agent_num] = -999999         
        ####
        
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        return out
'''