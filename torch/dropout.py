import torch
import numpy as np


batch = 2
length = 3
dim = 4

dropout_out = torch.nn.Dropout2d(p=0.5)

# (batch, 2*len, dim)
arc = torch.Tensor(batch, 2*length, dim).random_() % 10

print (arc)

print (dropout_out(arc))

print (dropout_out(arc.transpose(1, 2)).transpose(1, 2))