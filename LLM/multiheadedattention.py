import torch.nn as nn
import torch
from LLM.head import Head
from LLM.hyperparameter import block_size,n_embed

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embed, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)  # ADD THIS

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))  # ADD DROPOUT
        return out        