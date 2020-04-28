import torch
import numpy as np
from transformers import *

model_path = "/mnt/hgfs/share/xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
PAD = "<PAD>"

sent1 = ["The","Dow","fell","22.6","%","on","blacked","Monday"]
sent2 = ["The", "Dow", "Jones", "industrials", "closed", "at", "2569.26", "<PAD>"]
sents = [sent1, sent2]
bpe_ids = tokenizer.encode(" ".join(sent2))
print (bpe_ids)

all_wordpiece_list = []
all_first_index_list = []

for sent in sents:
	wordpiece_list = []
	first_index_list = []
	for token in sent:
		if token == PAD:
			token = tokenizer.pad_token
		wordpiece = tokenizer.tokenize(token)
		# add 1 for cls_token <s>
		first_index_list.append(len(wordpiece_list)+1)
		wordpiece_list += wordpiece
		#print (wordpiece)
	#print (wordpiece_list)
	#print (first_index_list)
	bpe_ids = tokenizer.convert_tokens_to_ids(wordpiece_list)
	#print (bpe_ids)
	bpe_ids = tokenizer.build_inputs_with_special_tokens(bpe_ids)
	#print (bpe_ids)
	all_wordpiece_list.append(bpe_ids)
	all_first_index_list.append(first_index_list)

all_wordpiece_max_len = max([len(w) for w in all_wordpiece_list])
all_wordpiece = np.stack(
      [np.pad(a, (0, all_wordpiece_max_len - len(a)), 'constant', constant_values=tokenizer.pad_token_id) for a in all_wordpiece_list])
all_first_index_max_len = max([len(i) for i in all_first_index_list])
all_first_index = np.stack(
      [np.pad(a, (0, all_first_index_max_len - len(a)), 'constant', constant_values=0) for a in all_first_index_list])

print (all_wordpiece)
print (all_first_index)


model = XLMRobertaModel.from_pretrained(model_path)

# (batch, max_bpe_len)
input_ids = torch.from_numpy(all_wordpiece)
# (batch, seq_len)
first_index = torch.from_numpy(all_first_index)
# (batch, max_bpe_len, hidden_size)
xlm_output = model(input_ids)[0]
print (xlm_output.size())

size = list(first_index.size()) + [xlm_output.size()[-1]]
output = xlm_output.gather(1, first_index.unsqueeze(-1).expand(size))
print (output.size())