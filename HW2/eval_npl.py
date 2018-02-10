import torchtext
import torch
import math 
import torch.nn as nn 
from torchtext.vocab import Vectors
from NPL_gpu import NPLM 

# Our input $x$
TEXT = torchtext.data.Field()
# Data distributed with the assignment
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path=".", 
    train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)
TEXT.build_vocab(train)

n_embedding = 30
n_hidden = 60
seq_len = 5
npl = NPLM(len(TEXT.vocab), n_embedding, n_hidden, seq_len)
npl.load_state_dict(torch.load('npl_big_model.pt'))
with open("sample.txt", "w") as fout: 
    print("id,word", file=fout)
    for i, l in enumerate(open("input.txt"), 1):
        print(l)
        input_tokens = l.split(" ")[-6:-1]
        input_index = torch.LongTensor([TEXT.vocab.stoi[t] for t in input_tokens]).unsqueeze(1)
        output = npl(torch.autograd.Variable(input_index))
        max_values, max_indices = torch.topk(output, 20)
        print(" ".join([TEXT.vocab.itos[int(i)] for i in max_indices.data[0, :]]))
        
        #print("%d,%s"%(i, " ".join(predictions)), file=fout)
