from utils import encode,decode,text
import torch
from LLM.hyperparameter import block_size,batch_size,device,max_iters,eval_interval,lr,eval_iters
from LLM.languagemodel import BigramLanguageModel
from split_data import train_data,val_data
#! ---------- HYPERPARAMETERS ----------
# ----- HYPERPARAMETERS
print(f'The device that we are going to use is: {device}')

#! Reading the tiny-shakespehere data


#! Printing how many data points we have in this dataset


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i+1:i + block_size + 1] for i in ix])
    x,y = x.to(device),y.to(device)
    return x,y

torch.manual_seed(1337)

@torch.no_grad()
def eval_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = BigramLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(m.parameters(),lr=lr)


for iterator in range(max_iters):
    if iterator % eval_interval == 0:
        losses = eval_loss()
        print(f'Step : {iterator} | Train Loss : {losses['train']} | Eval Loss : {losses['val']}')
    xb,yb = get_batch('train')
    logits,loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()