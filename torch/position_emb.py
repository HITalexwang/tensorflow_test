import torch
from torch import nn
import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self, hidden_size, max_position_embeddings=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(max_position_embeddings, hidden_size))
        print (self.pos_table)

    def _get_sinusoid_encoding_table(self, max_position_embeddings, hidden_size):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / hidden_size) for hid_j in range(hidden_size)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(max_position_embeddings)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        """
        Input:
            x: (batch, seq_len,)
        """
        return x + self.pos_table[:, :x.size(1)].detach()

pos = PositionalEncoding(10, 5)
x = torch.zeros(3, 4, 10)
print (pos.forward(x))