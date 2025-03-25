import torch.nn as nn
import torch
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self,head_size,n_embed,block_size):
        super().__init__()
        self.key = nn.Linear(n_embed,head_size,bias = False)
        self.query = nn.Linear(n_embed,head_size,bias = False)
        self.value = nn.Linear(n_embed,head_size,bias = False)

        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        B,T,C = x.shape
        q = self.query(x)
        k = self.key(x)

        wei = q @ k.transpose(-2,-1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0,float('-inf'))
        wei = F.softmax(wei,dim  = 1)
        v = self.value(x)
        # print(v.shape,wei.shape,(wei@v).shape)
        out = wei @ v
        return out