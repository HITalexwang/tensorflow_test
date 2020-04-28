import torch
import numpy as np
import torch.nn as nn

batch_size = 2
seq_len = 3
d = 4
out_size = 2

x = torch.LongTensor(batch_size, seq_len, seq_len).random_() % 3
rel = torch.LongTensor(batch_size, seq_len, seq_len, out_size).random_() % 10
words = torch.LongTensor(batch_size, seq_len).random_() % 3
mask = np.array([[1,1,0],[1,0,0]])
mask = torch.from_numpy(mask)
#mask = mask.ne(0)
#mask = words.ne(0)

#print (words)
print (mask)

minus_mask = mask.eq(0).unsqueeze(2)
x_ = x.masked_fill(minus_mask, -1)

minus_mask = mask.eq(0).unsqueeze(1)
x_2 = x.masked_fill(minus_mask, -1)

print (x)
#print (x_)
#print (x_2)
#print (x[mask])

#print (rel)
#print (rel[mask])
print (x * mask.unsqueeze(-1))