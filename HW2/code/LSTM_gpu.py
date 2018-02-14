# Text text processing library
import torchtext
import torch
import math
import torch.nn as nn
from torchtext.vocab import Vectors

DEBUG = False
RETRAIN = True
class LSTM(nn.Module):
    def __init__(self, V_vocab_dim, M_embed_dim, H_hidden_dim, N_seq_len, B_batch_size):
        super(LSTM, self).__init__()
        self.batch_size = B_batch_size
        self.hidden_dim = H_hidden_dim
        self.vocab_dim = V_vocab_dim
        self.embed = nn.Embedding(V_vocab_dim, M_embed_dim)
        self.lstm = nn.LSTM(M_embed_dim, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, V_vocab_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if torch.cuda.is_available():
            return (torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)).cuda(),
                    torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)).cuda())
        else:
            return (torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                    torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        #print(sentence.shape)
        # input size N_seq_len x B_batch_size
        embeds = self.embed(sentence)
        #print(embeds.shape)
        #print(self.hidden.shape)
        # embeds size N_seq_len x B_batch_size x M_embed_dim
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        #print(lstm_out.shape)
        # lstm_out N_seq_len x B_batch_size x H_hidden_dim
        out = self.fc(self.dropout(lstm_out))
        # print(out.shape)
        # out N_seq_len x B_batch_size x V_vocab_dim
        return out

    def repackage_hidden(self, h):
        if type(h) == torch.autograd.Variable:
            if torch.cuda.is_available():
                return torch.autograd.Variable(h.data).cuda()
            else:
                return torch.autograd.Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

def evaluate(model, data_iterator):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    batch_count = 0
    for batch in iter(data_iterator):
        #model.hidden = model.init_hidden()
        model.hidden = model.repackage_hidden(model.hidden)
        output = model(batch.text)
        batch_loss = criterion(output.view(-1, model.vocab_dim), batch.target.view(-1)).data
        total_loss += batch_loss
        batch_count += 1
    return total_loss[0] / batch_count

def train_batch(model, criterion, optim, batch, target):
    # initialize hidden vectors
    model.zero_grad()
    #model.hidden = model.init_hidden()
    model.hidden = model.repackage_hidden(model.hidden)
    # calculate forward pass
    y = model(batch)
    # calculate loss
    loss = criterion(y.view(-1, model.vocab_dim), target.view(-1))
    # backpropagate and step
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
    optim.step()
    return loss.data[0]

# training loop
def run_training(model, criterion, optim, data_iterator, val_iter):

    for e in range(n_epochs):
        model.train()
        batches = 0
        epoch_loss = 0
        for batch in iter(data_iterator):
            batch_loss = train_batch(model, criterion, optim, batch.text, batch.target)
            batches += 1
            epoch_loss += batch_loss
        epoch_loss /= batches
        print("Epoch ", e, " Loss: ", epoch_loss, "Perplexity: ", math.exp(epoch_loss))
        train_loss = evaluate(model, data_iterator)
        print("Epoch Train Loss: ", train_loss, "Perplexity: ", math.exp(train_loss))
        val_loss = evaluate(model, val_iter)
        print("Epoch Val Loss: ", val_loss, "Perplexity: ", math.exp(val_loss))
        torch.save(model.state_dict(), 'LSTM_full_model.pt')

if __name__ == "__main__":
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

    if torch.cuda.is_available():
        train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
            (train, val, test), batch_size=12, device=None, bptt_len=32, repeat=False, shuffle=False)
    else: 
        train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
        (train, val, test), batch_size=12, device=-1, bptt_len=32, repeat=False, shuffle=False)

    # size of the embeddings and vectors
    n_embedding = 300
    n_hidden = 650
    seq_len = 32
    batch_size = 12

    # initialize LSTM
    lstm_model = LSTM(len(TEXT.vocab), n_embedding, n_hidden, seq_len, batch_size)
    if RETRAIN == True: 
        lstm_model.load_state_dict(torch.load('LSTM_small_model.pt'))
    if torch.cuda.is_available():
        lstm_model.cuda()

    n_epochs = 30
    learning_rate = .7
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(lstm_model.parameters(), lr=learning_rate)

    run_training(lstm_model, criterion, optim, train_iter, val_iter)
