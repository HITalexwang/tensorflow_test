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

#The Torch Tensor and NumPy array will share their underlying memory locations, 
#and changing one will change the other.

# Converting a Torch Tensor to a NumPy Array
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

# Converting NumPy Array to Torch Tensor
# All the Tensors on the CPU except a CharTensor support converting to NumPy and back.

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
print (torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!