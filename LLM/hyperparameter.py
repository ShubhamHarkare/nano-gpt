import torch

max_iters = 250
eval_iters = 300
eval_interval = max_iters//6
batch_size = 128
block_size = 64
n_embed = 128
lr = 1e-3
import torch

# Auto-detect best device (CUDA > MPS > CPU)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Apple MPS")
else:
    device = torch.device('cpu')
    print("Using CPU")
num_heads = 8
head_size = n_embed // num_heads
num_layers = 32
