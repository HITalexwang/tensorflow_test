import torch
import torch.nn as nn
from torch.autograd import Variable

def drop_sequence_sharedmask(inputs, dropout, batch_first=True):
    if batch_first:
        inputs = inputs.transpose(0, 1)
        seq_length, batch_size, hidden_size = inputs.size()
        drop_masks = inputs.data.new(batch_size, hidden_size).fill_(1 - dropout)
        drop_masks = Variable(torch.bernoulli(drop_masks), requires_grad=False)
        drop_masks = drop_masks / (1 - dropout)
        print (drop_masks)
        drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
        print (drop_masks)
        inputs = inputs * drop_masks

    return inputs.transpose(1, 0)

def drop_input_independent(word_embeddings, tag_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)#6*98 ;0.67
    #print(word_masks)
    word_masks = Variable(torch.bernoulli(word_masks), requires_grad=False)#6*78 ;0,1
    #print(word_masks)
    tag_masks = tag_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    tag_masks = Variable(torch.bernoulli(tag_masks), requires_grad=False)
    scale = 3.0 / (2.0 * word_masks + tag_masks + 1e-12)#batch_size*seq_length 1,1.5,3
    #print(scale)
    word_masks *= scale#batch_size*seq_length 0,1,1.5
    tag_masks *= scale
    print(word_masks)
    word_masks = word_masks.unsqueeze(dim=2)#6*78*1
    print(word_masks)
    tag_masks = tag_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    tag_embeddings = tag_embeddings * tag_masks
    return word_embeddings, tag_embeddings

input = torch.Tensor(2,3,4).random_() % 10
print (input)
#output = drop_sequence_sharedmask(input, 0.5)

output, _ = drop_input_independent(input, input, 0.5)
print (output)

drop = nn.Dropout(0.5)

print (drop(input))