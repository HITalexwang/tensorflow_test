import torch
import torch.nn as nn



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

b = 3 # batch size
n = 2 # label block number
l = 4
d = 5

block = torch.Tensor(n, d, d).random_(2)
h = torch.Tensor(b, l, d).random_(3)
#y = torch.Tensor(b, l, n_in).random_(3)

block = block.unsqueeze(0)
h = h.unsqueeze(1)

h_b = torch.matmul(h, block)

print ('block:\n',block)
print ('h:\n', h)
print ('h_b:\n', h_b)

n_r = 6 # label number

coe = torch.Tensor(n_r, n).random_(2)
coe_ = coe.unsqueeze(-1).unsqueeze(-1).expand(n_r, n, l, d)
print ("coe:\n", coe)
#print ("coe_:\n", coe_)

mat = h_b.unsqueeze(1) * coe_.unsqueeze(0)
#print ("mat:\n", mat)
# (b, n_r, l, d)
h_r = mat.sum(2)
print ("h_r:\n", h_r)

rels = torch.Tensor(b, l, l).random_(n_r).long()

print ("rels:\n", rels)

rels_ = torch.zeros(b, n_r, l, l)

#print (rels_.shape)
# direct scatter
# (b, n_r, l, l)
rels_.scatter_(1, rels.unsqueeze(1), 1)

rels_2 = torch.zeros(b, l, l, n_r)

#print (rels_.shape)
# direct scatter
# (b, l, l, n_r)
rels_2.scatter_(-1, rels.unsqueeze(-1), 1)
rels_2 = rels_2.permute(0,3,1,2)

#print (rels_.shape, rels_2.shape)

assert (rels_ == rels_2).all()

#print ("rels_:\n", rels_)

h_a = torch.matmul(rels_, h_r)
print ("h_a:\n", h_a)

h_out = h_a.sum(1)

print ("h_out:\n", h_out)