import torch.nn as nn
from LLM.multiheadedattention import MultiHeadedAttention
from LLM.feedforward import FeedForward

class Block(nn.Module):
    def __init__(self,n_embed,n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadedAttention(n_heads,head_size,n_embed)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x +self.ffwd(self.ln2(x))
        return x
        