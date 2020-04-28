import torch
import numpy as np

batch = 2
length = 3
type_space = 4

heads = np.array([[2,0,1],[1,2,0]])
heads = torch.from_numpy(heads)

types = torch.IntTensor(batch, length).random_() % 5 + 3
print (heads)
print (types)

heads_ = heads.unsqueeze_(-1)

# (batch, seq_len, seq_len)
head_3D = torch.zeros(batch, length, length, dtype=torch.int32)
head_3D.scatter_(-1, heads_, 1)

print (head_3D)
# (batch, seq_len, seq_len)
type_3D = torch.zeros((batch, length, length), dtype=torch.int32)
# (batch, seq_len, 1) (batch, seq_len,1 )

#print (heads)
#print (types.unsqueeze_(-1).float())

type_3D.scatter_(-1, heads_, types.unsqueeze_(-1))

print (type_3D)