import torch
import torch.optim as optim
from torch import nn

class func(nn.Module):
	def __init__(self):
		super(func, self).__init__()
		#self.w1 = torch.randn(3, 3)
		#self.w1.requires_grad = True
		#self.w2 = torch.randn(3, 3)
		#self.w2.requires_grad = True
		self.w1 = nn.Linear(3,3, bias=False)
		self.w2 = nn.Linear(4,4, bias=False)

f = func()
#o = optim.Adam(f.parameters())
o = optim.Adam([{'params':f.w1.parameters(), 'lr':1e-5},
								{'params':f.w2.parameters()}], lr=0.01)
print(o.param_groups)

#o.add_param_group({'params':f.w1.parameters(), 'lr':1e-5})
#print (o.param_groups)