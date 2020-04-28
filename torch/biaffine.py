import torch
import torch.nn as nn

b = 5
l = 4
n_in = 3
n_out = 2

def forward(x, y, w):
  # [batch_size, 1, seq_len, d]
  x = x.unsqueeze(1)
  # [batch_size, 1, seq_len, d]
  y = y.unsqueeze(1)

  s1 = x @ weight
  s1_ = torch.matmul(x, weight)
  # [batch_size, n_out, seq_len, seq_len]
  s2 = x @ weight @ y.transpose(-1, -2)
  s2_ = torch.matmul(torch.matmul(x, weight), y.transpose(-1, -2))
  # remove dim 1 if n_out == 1
  s = s2.squeeze(1)

  print ('s1:\n', s1)
  print ('s1_:\n', s1_)
  print ('s2:\n', s2)
  print ('s2_:\n', s2_)

  return s

weight = torch.Tensor(n_out, n_in, n_in).random_(3)
x = torch.Tensor(b, l, n_in).random_(3)
y = torch.Tensor(b, l, n_in).random_(3)

print ('w:\n',weight)
print ('x:\n', x)
print ('y:\n', y)

forward(x,y,weight)