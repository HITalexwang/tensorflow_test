import init
import torch

x = torch.rand(2,3)
y = torch.rand_like(x)
print("x:{} \ny:{}".format(x,y))

print (x+y)

print (torch.add(x,y))

result = torch.empty_like(x)
torch.add(x, y, out=result)
print (result)


c = y.add(x)
print ("c = y.add(x)=>\ny:{}\nc:{}".format(y,c))

#Any operation that mutates a tensor in-place is post-fixed with an _.
#For example: x.copy_(y), x.t_(), will change x.
y.add_(x)
print ("y.add_(x)=>\ny:{}".format(y))

# transpose
print (x.t_())

# You can use standard NumPy-like indexing with all bells and whistles!
print (x[1:,:])

# Resizing: If you want to resize/reshape tensor, you can use torch.view
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 2, 2)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# If you have a one element tensor, use .item() to get the value as a Python number

x = torch.randn(1)
print(x)
print(x.item())