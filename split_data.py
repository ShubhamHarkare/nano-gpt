import torch
from utils import encode

with open('tiny-shakeshpere.txt','r') as f:
    text = f.read()

data = torch.tensor(encode(text),dtype = torch.long)
print(data.shape,data.dtype)


n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
