import torch
import numpy as np

n_layers = 2
batch_size = 3
seq_len = 4
# Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
y = torch.LongTensor(n_layers,batch_size, seq_len).random_() % seq_len
y = y.unsqueeze_(-1)

print (y)

# One hot encoding buffer that you create out of the loop and just keep reusing
y_onehot = torch.zeros(n_layers,batch_size, seq_len, seq_len)

# In your for loop
y_onehot.scatter_(-1, y, 1)

print(y_onehot)

# (batch_size, seq_len)
mask = np.zeros([batch_size, seq_len])
mask[0, :3] = 1
mask[1, :] = 1
mask[2, :2] = 1
mask = torch.from_numpy(mask)
mask = mask.unsqueeze_(0).unsqueeze_(-1)
print (mask)

masked_onehot = y_onehot * mask

print (masked_onehot)