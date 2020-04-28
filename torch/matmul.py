import torch
import numpy as np

def matmul_rel(x, y, z=None):
	"""
	x: (batch_size, num_heads, seq_len, head_size)
	y: (batch_size, num_heads, seq_len, head_size)
	z: (batch_size, num_heads, seq_len, seq_len, head_size)
	"""
	sizes = list(x.size())
	seq_len, hidden_size = sizes[-2:]
	#print (seq_len, hidden_size)
	new_sizes = sizes[:-2] + [seq_len] + sizes[-2:]
	if z is not None:
		assert list(z.size()) == new_sizes
	#print (new_sizes)
	x_ = x.unsqueeze(-2).expand(new_sizes)
	y_ = y.unsqueeze(-3).expand(new_sizes)
	#print ("x_:\n", x_)
	#print ("y_:\n", y_)
	if z is not None:
		y_ = y_ + z
		#print ("y_+z:\n", y_)
	out = (x_ * y_).sum(-1).squeeze(-1)

	return out

batch_size = 2
num_heads = 2
seq_len = 3
head_size = 4
# Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
x = torch.LongTensor(batch_size, num_heads, seq_len, head_size).random_() % 3
y = torch.LongTensor(batch_size, num_heads, seq_len, head_size).random_() % 3
z = torch.LongTensor(batch_size, seq_len, seq_len, head_size).random_() % 3

print ("x:\n", x)
print ("y:\n", y)
print ("z:\n", z)
z_ = z.unsqueeze(1).expand(batch_size, num_heads, seq_len, seq_len, head_size)
print ("z_:\n", z_)
out = torch.matmul(x, y.transpose(-1,-2))
print ("out:\n", out)
print ("out_:\n", matmul_rel(x,y,z_))
exit()
