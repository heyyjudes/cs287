import torch
import torchtext
from torchtext.vocab import Vectors, GloVe
import torch.nn as nn
import torch.nn.functional as F
from random import shuffle

def test_lstm(input_model, data):
    criterion = nn.NLLLoss()
    correct = 0.
    num_examples = 0.
    nll = 0.
    for batch in data:
        if batch.text.size(1) == 10:
            text = batch.text
            label = batch.label
            input_model.hidden = input_model.init_hidden()
            y_pred = input_model(text)
            nll_batch = criterion(y_pred, label - 1)
            nll += nll_batch.data[0] * text.size(0)  # by default NLL is averaged over each batch
            y_pred_max, y_pred_argmax = torch.max(y_pred, 1)  # prediction is the argmax
            correct += (y_pred_argmax.data == label.data - 1).sum()
            num_examples += text.size(1)
    return nll / num_examples, correct / num_examples


class LSTM(nn.Module):
    def __init__(self, hidden_dim, vocab, batch_size, seq_len, embedding_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.embed = nn.Embedding(len(vocab), embedding_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, 2)
        self.hidden = self.init_hidden()
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self):
        return (torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.embed(sentence.t())
        x = torch.t(embeds)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.fc(torch.mean(lstm_out, dim=0))
        out = F.log_softmax(y, dim=1)
        return out

def run_lstm():
    # Our input $x$
    seq_length = 56
    TEXT = torchtext.data.Field(fix_length=seq_length)

    # Our labels $y$
    LABEL = torchtext.data.Field(sequential=False)

    train, val, test = torchtext.datasets.SST.splits(
        TEXT, LABEL,
        filter_pred=lambda ex: ex.label != 'neutral')

    TEXT.build_vocab(train)
    LABEL.build_vocab(train)
    # this is just the set of stuff
    print('len(TEXT.vocab)', len(TEXT.vocab))
    print('len(LABEL.vocab)', len(LABEL.vocab))

    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train, val, test), batch_size=10, device=-1, repeat=False)

    # Glove embeddings
    TEXT.vocab.load_vectors(vectors=GloVe(name="6B", dim="300"))

    lstm_model = LSTM(hidden_dim=100, vocab=TEXT.vocab, batch_size=10, seq_len=56, embedding_dim=300)
    criterion = nn.NLLLoss()
    parameters = filter(lambda p: p.requires_grad, lstm_model.parameters())
    optim = torch.optim.SGD(parameters, lr=0.05, weight_decay=0.0001)
    num_epochs = 20

    for e in range(num_epochs):
        for batch in train_iter:
            optim.zero_grad()
            lstm_model.hidden = lstm_model.init_hidden()
            text = batch.text
            label = batch.label
            y_pred = lstm_model(text)
            nll_batch = criterion(y_pred, label - 1)
            nll_batch.backward()
            optim.step()
            # shuffle text data
        #         optim.zero_grad()
        #         lstm_model.hidden = lstm_model.init_hidden()
        #         text = torch.cat((batch.text[torch.randperm(40), :], batch.text[40:, :]))
        #         text = batch.text
        #         label = batch.label
        #         y_pred = lstm_model(text)
        #         nll_batch = criterion(y_pred, label-1)
        #         nll_batch.backward()
        #         optim.step()
        nll_train, accuracy_train = test_lstm(lstm_model, train_iter)
        nll_val, accuracy_val = test_lstm(lstm_model, val_iter)
        nll_test, accuracy_test = test_lstm(lstm_model, test_iter)
        print('Test performance after epoch %d: NLL: %.4f, Accuracy: %.4f' % (e + 1, nll_test, accuracy_test))
        print('Training performance after epoch %d: NLL: %.4f, Accuracy: %.4f' % (e + 1, nll_train, accuracy_train))
        print('Validation performance after epoch %d: NLL: %.4f, Accuracy: %.4f' % (e + 1, nll_val, accuracy_val))

if __name__ == "__main__":
    run_lstm()