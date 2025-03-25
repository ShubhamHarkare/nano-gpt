import torch
import torch.nn as nn
import torch.nn.functional as F

from LLM.hyperparameter import n_embed,block_size,num_heads,num_layers,device
from utils import vocab_size
from LLM.block import Block


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size,n_embed)
        self.position_embeddings = nn.Embedding(block_size,n_embed)
        self.lm_head = nn.Linear(n_embed,vocab_size)
        self.blocks = nn.Sequential(*[Block(n_embed,num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embed)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    
    def forward(self,idx,targets = None):
        B,T = idx.shape
        token_embeddings = self.token_embedding(idx) # Shape = (B,T,C)
        positional_embeddings = self.position_embeddings(torch.arange(T,device = device)) # Shape = (T,C)
        x = token_embeddings + positional_embeddings # Shape = (B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x) # Shape -> (B,T,C)
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss


    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-block_size:]
            logits,loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim = 1)
            idx_next = torch.multinomial(probs,num_samples = 1)
            idx = torch.cat((idx,idx_next),dim = 1)
        return idx