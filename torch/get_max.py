import torch
import numpy as np

def _get_top_arc_hidden_states(dep_hidden_states, head_hidden_states, head_logp):
	"""
	Input:
	  hidden_sates: (batch, seq_len, hidden_size)
	  head_logp: (batch, seq_len, seq_len), the log probability of every arc
	"""
	batch_size, seq_len, hidden_size = dep_hidden_states.size()
	# (batch, seq_len*seq_len)
	flatten_logp = head_logp.view([batch_size, -1])
	# (batch)
	values, flatten_max_indices = torch.max(flatten_logp, -1)
	# (batch)
	max_indices_dep = flatten_max_indices // seq_len
	max_indices_head = flatten_max_indices % seq_len

	id_dep = max_indices_dep.unsqueeze(1).unsqueeze(2).expand(batch_size,1,hidden_size)
	print (id_dep)
	# (batch, hidden_size)
	selected_dep_hidden_states = dep_hidden_states.gather(dim=1, index=id_dep).squeeze(1)
	print (selected_dep_hidden_states)

	id_head = max_indices_head.unsqueeze(1).unsqueeze(2).expand(batch_size,1,hidden_size)
	print (id_head)
	# (batch, hidden_size)
	selected_head_hidden_states = head_hidden_states.gather(dim=1, index=id_head).squeeze(1)
	print (selected_head_hidden_states)

batch = 2
length = 3
hidden_size = 4

heads = np.array([[2,0,1],[1,2,0]])
heads = torch.from_numpy(heads)
hidden = torch.LongTensor(batch, length, hidden_size).random_() % 10
print ('hidden_states:\n',hidden)

logp = torch.Tensor(batch, length, length).random_() % 10
print ('logp:\n',logp)

_get_top_arc_hidden_states(hidden, hidden, logp)