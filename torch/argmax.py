import torch
import numpy as np

def argmax(logits):
  """
  Input:
      logits: (batch, seq_len, seq_len), as probs
  Output:
      graph_matrix: (batch, seq_len, seq_len), in onehot
  """
  # (batch, seq_len)
  index = logits.argmax(-1).unsqueeze(2)
  # (batch, seq_len, seq_len)
  graph_matrix = torch.zeros_like(logits).int()
  graph_matrix.scatter_(-1, index, 1)
  return graph_matrix

n_layers = 2
batch_size = 3
seq_len = 4
# Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
y = torch.LongTensor(n_layers, seq_len, seq_len).random_() % 10

print (y)

print (argmax(y))
