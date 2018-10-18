from __future__ import print_function
import torch

x = torch.empty(3,5)
x = torch.zeros(5,3,dtype=torch.long)
x = torch.tensor([1,2.5])
#print (x)

x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes

x = x.new_zeros(5,3)
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size

print (x.size()) # torch.Size is in fact a tuple, so it supports all tuple operations.