import torch
import torchtext
import math
import torch.nn as nn
from torchtext.vocab import Vectors, GloVe
from LSTM_gpu import LSTM, evaluate

n_embedding = 30
n_hidden = 300
seq_len = 32
batch_size = 12

# Our input $x$
TEXT = torchtext.data.Field()
# Data distributed with the assignment
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path=".", 
    train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)
TEXT.build_vocab(train)
print('len(TEXT.vocab)', len(TEXT.vocab))
TEXT.vocab.load_vectors(vectors= GloVe(name="6B", dim="300"))
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
            (train, val, test), batch_size=12, device=None, bptt_len=32, repeat=False, shuffle=False)

lstm_model = LSTM(len(TEXT.vocab), n_embedding, n_hidden, seq_len, batch_size)
lstm_model.load_state_dict(torch.load('final_models/LSTM_full_model.pt'))
lstm_model.cuda()
lstm_model.eval()

train_loss = evaluate(lstm_model, train_iter) 
print("Epoch Train Loss: ", train_loss, "Perplexity: ", math.exp(train_loss))
   
val_loss = evaluate(lstm_model, val_iter)
print("Epoch Val Loss: ", val_loss, "Perplexity: ", math.exp(val_loss))
       

with open("sample.txt", "w") as fout: 
    print("id,word", file=fout)
    for i, l in enumerate(open("input.txt"), 1):
        print(l)
        input_tokens = l.split(" ")[:-1]
        input_index = torch.LongTensor([TEXT.vocab.stoi[t] for t in input_tokens]).unsqueeze(1)
        lstm_model.hidden = lstm_model.init_hidden()
        output = lstm_model(torch.autograd.Variable(input_index).cuda())
        clean_output = output[-1, 0, :].view(-1, lstm_model.vocab_dim)
        max_values, max_indices = torch.topk(clean_output, 20)
        predictions = [TEXT.vocab.itos[int(i)] for i in max_indices.data[0, :]]
        print("%d,%s"%(i, " ".join(predictions)), file=fout)
