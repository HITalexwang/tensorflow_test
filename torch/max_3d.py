import torch
import numpy as np

def _max_3D(tensor):
	"""
	Input:
	    tensor: (batch, seq_len, seq_len)
	Return:
	    max_val: (batch), the max value
	    max_tensor_3D: (batch, seq_len, seq_len)
	    max_heads_2D: (batch, seq_len), each entry is the index of head
	"""
	batch_size, seq_len, _ = tensor.size()
	# (batch, seq_len*seq_len)
	flatten_tensor = tensor.view([batch_size, -1])
	print (flatten_tensor)
	# (batch)
	max_val, max_indices = flatten_tensor.max(dim=1)
	print (max_val, max_indices)
	max_dep_indices = max_indices // seq_len
	max_head_indices = max_indices % seq_len
	print (max_dep_indices, max_head_indices)

	dep_mask = torch.zeros(batch_size, seq_len, dtype=torch.int32, device=tensor.device)
	dep_mask.scatter_(1, max_dep_indices.unsqueeze(1), 1)

	max_heads_2D = torch.zeros(batch_size, seq_len, dtype=torch.int32, device=tensor.device)
	max_heads_2D.scatter_(1, max_dep_indices.unsqueeze(1), max_head_indices.unsqueeze(1).int())

	max_tensor_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32, device=tensor.device)
	max_tensor_3D.scatter_(-1, max_heads_2D.unsqueeze(-1).long(), 1)
	max_tensor_3D = max_tensor_3D * dep_mask.unsqueeze(-1)

	return max_val, max_tensor_3D, max_heads_2D

batch = 2
length = 3
hidden_size = 4

logp = torch.Tensor(batch, length, length).random_() % 10
print ('score:\n',logp)

max_val, max_tensor_3D, max_heads_2D = _max_3D(logp)

print ('val:\n', max_val)
print ('3D:\n', max_tensor_3D)
print ('2D:\n', max_heads_2D)