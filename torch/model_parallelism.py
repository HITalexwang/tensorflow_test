import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

gpu=[0,1]

class Model(nn.Module):
  def __init__(self, dim):
    super(Model, self).__init__()
    self.x2h = nn.Linear(dim, 5)
    self.x2h.cuda(gpu[0])
    self.h2y = nn.Linear(5, 3)
    self.h2y.cuda(gpu[1])
    self.criterion = nn.CrossEntropyLoss(size_average=False)
    
  def forward(self, x, y):
    x = torch.from_numpy(x).float()
    x = x.cuda()
    y = torch.from_numpy(y)
    y = y.cuda(gpu[1])
    # gpu[0]
    h = self.x2h(x)
    # h gpu0 => gpu1
    h = h.cuda(gpu[1])
    y_ = self.h2y(h)
    return self.criterion(y_, y)

model = Model(4)
optimizer = optim.SGD(model.parameters(), lr=0.1)
x = np.random.randn(2,4)
print (x)
y = np.array([0,2])
model.zero_grad()
loss = model(x, y)
print ("x2h:\n{}\ngrad:\n{}\nh2y:\n{}\ngrad:\n{}".format(model.x2h.weight, model.x2h.weight.grad, model.h2y.weight, model.h2y.weight.grad))
print ("loss:",loss)
loss.backward()
optimizer.step()
print ("x2h:\n{}\ngrad:\n{}\nh2y:\n{}\ngrad:\n{}".format(model.x2h.weight, model.x2h.weight.grad, model.h2y.weight, model.h2y.weight.grad))

