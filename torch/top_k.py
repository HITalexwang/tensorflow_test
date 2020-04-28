import torch
import numpy as np

def _get_recomp_logp(head_logp):
	"""
	Input:
	  head_logp: (batch, seq_len, seq_len), the log probability of every arc
	"""
	batch_size, seq_len, _ = head_logp.size()
	# (batch, seq_len*seq_len)
	flatten_logp = head_logp.view([batch_size, -1])
	# (batch)
	values, flatten_indices = torch.topk(flatten_logp, k=2, dim=-1)
	# (batch)
	#indices_dep = flatten_indices[:,0] // seq_len
	#indices_head = flatten_indices[:,0] % seq_len
	indices_dep = flatten_indices // seq_len
	indices_head = flatten_indices % seq_len

	print ("flatten_indices:\n",flatten_indices)
	print ("indices_dep:\n",indices_dep.unsqueeze(1).unsqueeze(2))
	print ("indices_head:\n",indices_head)

	max_dep_onehot = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.int32)
	max_dep_onehot.scatter_(1, indices_dep.unsqueeze(1).unsqueeze(2).expand_as(max_dep_onehot), 1)

	max_head_onehot = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.int32)
	max_head_onehot.scatter_(2, indices_head.unsqueeze(1).unsqueeze(2).expand_as(max_dep_onehot), 1)
	
	#print ("max_dep_onehot:\n",max_dep_onehot)
	#print ("max_head_onehot:\n",max_head_onehot)
	print ("multi:\n", max_dep_onehot*max_head_onehot)

def get_topk(k, max_tensor, logits):
  """
  Input:
      tensor: (batch, seq_len, seq_len)
  Return:
      max_val: (batch), the max value
      max_tensor_3D: (batch, seq_len, seq_len)
      max_heads_2d: (batch, seq_len), each entry is the dep_indices of head
  """
  batch_size, seq_len, _ = logits.size()
  mask = max_tensor.sum(-1).unsqueeze(-1).expand_as(logits)
  print ("mask:\n", mask)
  neg_inf = torch.ones_like(logits) * -1e9
  masked_logits = torch.where(mask == 1, logits, neg_inf)
  print ("masked_logits:\n", masked_logits)
  # (batch, seq_len*seq_len)
  flatten_tensor = masked_logits.view([batch_size, -1])
  # (batch)
  val, indices = flatten_tensor.topk(k=k, dim=-1)
  # 
  dep_indices = indices // seq_len
  head_indices = indices % seq_len

  print ("indices:\n", indices)
  print ("dep_indices:\n", dep_indices)
  print ("head_indices:\n", head_indices)

  dep_mask = torch.zeros(batch_size, seq_len, dtype=torch.int32, device=logits.device)
  dep_mask.scatter_(1, dep_indices, 1)

  print ("dep_mask:\n", dep_mask)

  heads_2d = torch.zeros(batch_size, seq_len, dtype=torch.int32, device=logits.device)
  heads_2d.scatter_(1, dep_indices, head_indices.int())

  print ("heads_2d:\n", heads_2d)

  max_tensor_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32, device=logits.device)
  max_tensor_3D.scatter_(-1, max_heads_2d.unsqueeze(-1).long(), 1)
  max_tensor_3D = max_tensor_3D * dep_mask.unsqueeze(-1)

  return max_val, max_tensor_3D, max_heads_2d


def get_topk_v2(k, max_tensor, logits):
  """
  Input:
      tensor: (batch, seq_len, seq_len)
  Return:
  """
  batch_size, seq_len, _ = logits.size()
  # (batch, seq_len)
  max_tensor_2d = max_tensor.sum(-1)
  _, dep_indices = max_tensor_2d.max(-1)
  dep_indices_ = dep_indices.unsqueeze(1).expand_as(max_tensor_2d).unsqueeze(1)
  logits_ = logits.gather(1, dep_indices_).squeeze(1)
  vals, head_indices = logits_.topk(k=k, dim=-1)
  
  cand_tensor_2d = torch.zeros_like(max_tensor_2d)
  cand_tensor_2d.scatter_(1, head_indices, 1)
  cand_tensor_3d = torch.zeros_like(logits)
  cand_tensor_3d.scatter_(1, dep_indices_, cand_tensor_2d.unsqueeze(1))

  #print ("max_tensor_2d:\n", max_tensor_2d)
  #print ("dep_indices_:\n", dep_indices_)
  #print ("logits_:\n", logits_)
  #print ("head_indices:\n", head_indices)
  #print ("cand_tensor_2d:\n", cand_tensor_2d)
  print ("cand_tensor_3d:\n", cand_tensor_3d)


batch = 2
length = 3

logp = torch.Tensor(batch, length, length).random_() % 10
print ('logp:\n',logp)

#a = torch.Tensor([.9,.2,.5])
#flag = torch.ge(a, 0.5).sum() == 2
#if flag:
#	print ("flag")

#exit()

#_get_recomp_logp(logp)
max_tensor = torch.zeros_like(logp)
max_tensor[0,2,2] = 1
max_tensor[1,2,1] = 1
print ("max_tensor:\n", max_tensor)
get_topk_v2(2, max_tensor, logp)