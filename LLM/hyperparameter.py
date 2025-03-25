import torch

max_iters = 1000
eval_iters = 300
eval_interval = max_iters//6
batch_size = 128
block_size = 64
n_embed = 128
lr = 1e-3
device = torch.device('mps' if torch.backends.mps.is_available else 'cpu')
num_heads = 8
head_size = n_embed // num_heads
num_layers = 32
