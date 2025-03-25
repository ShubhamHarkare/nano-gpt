with open('tiny-shakeshpere.txt','r') as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch  in s]
decode = lambda l: ''.join([itos[i] for i in l])

vocab_size = len(chars)

# print(encode('hi there'))
# print(decode(encode('hi there')))