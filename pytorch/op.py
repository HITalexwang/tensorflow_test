import init
import torch

x = torch.rand(2,3)
y = torch.rand_like(x)
print (x+y)
