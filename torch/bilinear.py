import torch
import torch.nn as nn

def manual_bilinear(x1, x2, A, b):
	#print (A)
	#print (x2)
	#print (torch.mm(A, x2))
	return torch.mm(x1, torch.mm(A, x2)) + b

x_ones = torch.ones(2,3)
x_zeros = torch.zeros(2,3)


# ---------------------------
# With Bias:
"""
B = nn.Bilinear(2, 2, 1)
A = B.weight
print(B.bias)
# > tensor([-0.6748], requires_grad=True)
b = B.bias

print(B(x_ones, x_zeros))
# > tensor([-0.6748], grad_fn=<ThAddBackward>)
print(manual_bilinear(x_ones.view(1, 2), x_zeros.view(2, 1), A.squeeze(), b))
# > tensor([[-0.6748]], grad_fn=<ThAddBackward>)

print(B(x_ones, x_ones))
# > tensor([-1.7684], grad_fn=<ThAddBackward>)
print(manual_bilinear(x_ones.view(1, 2), x_ones.view(2, 1), A.squeeze(), b))
# > tensor([[-1.7684]], grad_fn=<ThAddBackward>)
"""
# ---------------------------
# Without Bias:

B = nn.Bilinear(3, 3, 1, bias=False)
A = B.weight
print (A)
#print(B.bias)
# None
b = torch.zeros(1)
"""
print(B(x_ones, x_zeros))
# > tensor([0.], grad_fn=<ThAddBackward>)
print(manual_bilinear(x_ones.view(1, 2), x_zeros.view(2, 1), A.squeeze(), b))
# > tensor([0.], grad_fn=<ThAddBackward>)
"""
print(B(x_ones, x_ones * 2))
# > tensor([-0.7897], grad_fn=<ThAddBackward>)
print(manual_bilinear(x_ones.view(2, 3), x_ones.view(3, 2) * 2, A.squeeze(), b))
# > tensor([[-0.7897]], grad_fn=<ThAddBackward>)