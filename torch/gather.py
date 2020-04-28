import torch
import numpy as np

batch = 2
length = 3
type_space = 4

heads = np.array([[2,0,1],[1,2,0]])
heads = torch.from_numpy(heads)
type_h = torch.LongTensor(batch, length, type_space).random_() % 10
types = torch.LongTensor(batch, length, length).random_() % 10
#print (type_h)
print (types)

print (heads)

type_h = type_h.gather(dim=1, index=heads.unsqueeze(2).expand(type_h.size()))

#print (type_h)
types = types.gather(dim=-1, index=heads.unsqueeze(2))

print (types)