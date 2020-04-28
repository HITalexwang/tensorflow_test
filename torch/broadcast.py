import torch
import torch.nn as nn

batch_size = 2
seq_len = 3
d = 4
out_size = 2

x = torch.LongTensor(batch_size, seq_len, d).random_() % 3
y = torch.LongTensor(batch_size, seq_len, d).random_() % 3
weight = torch.LongTensor(out_size, d, d).random_() % 3

# [batch_size, 1, seq_len, d]
x = x.unsqueeze(1)
# [batch_size, 1, seq_len, d]
y = y.unsqueeze(1)
# [batch_size, n_out, seq_len, seq_len]
s = x @ weight @ y.transpose(-1, -2)
s_ = torch.matmul(torch.matmul(x, weight), y.transpose(-1, -2))

print (x)
print (weight)
print (x@weight)
print (y.transpose(-1, -2))
print (s)
print (s.squeeze(1))