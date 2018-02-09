NPL_gpu.py

# Text text processing library
import torchtext
import torch
import math 
import torch.nn as nn 
from torchtext.vocab import Vectors
DEBUG = True 

class NPLM(nn.Module):

    def __init__(self, V_vocab_dim, M_embed_dim, H_hidden_dim, N_seq_len):
        super(NPLM, self).__init__()
        self.vocab_size = V_vocab_dim
        self.embed = nn.Embedding(V_vocab_dim, M_embed_dim)
        self.hidden_linear = nn.Linear(M_embed_dim*N_seq_len, H_hidden_dim, bias=True)
        self.tanh_act = nn.Tanh()
        self.U_linear = nn.Linear(H_hidden_dim, V_vocab_dim, bias=True)
        self.W_linear = nn.Linear(M_embed_dim*N_seq_len, V_vocab_dim, bias=True)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x_embed = self.embed(x)
        print(x.shape)
        #print(x_embed.shape)
        x_flat = torch.autograd.Variable(torch.cat(x_embed.data, dim=1))
        #print(x_flat.shape)
        hidden_feat = self.hidden_linear(x_flat)
        #print(hidden_feat.shape)
        hidden_act = self.tanh_act(hidden_feat)
        #print(hidden_act.shape)
        hidden_out = self.U_linear(hidden_act)
        #print(hidden_out.shape)
        direct = self.W_linear(x_flat)
        #print(direct.shape)
        out = direct + hidden_out
        #print(out.shape)
        return self.softmax(out)

def evaluate(model, data_iterator):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    batch_count = 0 
    for batch in iter(data_iterator):
        for i in range(batch.text.size(0) - 5):
            output = model(batch.text[i:i+5])
            target = batch.target[i+5]
            total_loss += criterion(output, target).data
            batch_count += 1 
    return total_loss[0] / batch_count

def train_batch(model, criterion, optim, batch, label):
    # initialize hidden vectors
    model.zero_grad()
    # calculate forward pass
    y = model(batch)
    # calculate loss    
    loss = criterion(y, label)
    # backpropagate and step
    loss.backward()
    optim.step()
    return loss.data[0]

# training loop
def train(model, criterion, optim, data_iterator):

    for e in range(n_epochs):
        batches = 0
        epoch_loss = 0
        avg_loss = 0
        for batch in iter(data_iterator):
            for i in range(batch.text.size(0) - 5): 
                batch_loss = train_batch(model, criterion, optim, batch.text[i:i+5], batch.target[i+5])
                batches += 1
                epoch_loss += batch_loss
                avg_loss = ((avg_loss * (batches - 1)) + batch_loss) / batches
        print("Epoch ", e, " Loss: ", epoch_loss, "Perplexity: ", math.exp(avg_loss))
        loss = evaluate(model, val_iter)
        print("Epoch Val Loss: ", loss, "Perplexity: ", math.exp(loss)) 
        loss = evaluate(model, test_iter)
        print("Epoch Test Loss: ", loss, "Perplexity: ", math.exp(loss))


if torch.cuda.is_available():
	print("running cuda device")
	torch.cuda.manual_seed(args.seed)


# size of the embeddings and vectors
n_embedding = 30
n_hidden = 60
seq_len = 5
n_epochs = 10
learning_rate = .1

# Our input $x$
TEXT = torchtext.data.Field()
# Data distributed with the assignment
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path=".", 
    train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)
TEXT.build_vocab(train)
print('len(TEXT.vocab)', len(TEXT.vocab))

if DEBUG == True:
    TEXT.build_vocab(train, max_size=1000)
    len(TEXT.vocab)
    print('len(TEXT.vocab)', len(TEXT.vocab))
    
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val, test), batch_size=12, device=None, bptt_len=32, repeat=False, shuffle=False)


# initialize LSTM
npl = NPLM(len(TEXT.vocab), n_embedding, n_hidden, seq_len)
nlp.cuda() 
criterion = nn.NLLLoss()
optim = torch.optim.SGD(npl.parameters(), lr = learning_rate)


train(npl, criterion, optim, test_iter)