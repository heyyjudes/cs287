import torch
import torchtext
from torchtext.vocab import GloVe
import torch.nn as nn


def test_cnn(model, data):
    criterion = nn.NLLLoss()
    correct = 0.
    num_examples = 0.
    nll = 0.
    for batch in data:
        text = batch.text
        label = batch.label
        y_pred = model(text)
        nll_batch = criterion(y_pred, label - 1)
        nll += nll_batch.data[0] * text.size(0) #by default NLL is averaged over each batch
        y_pred_max, y_pred_argmax = torch.max(y_pred, 1) #prediction is the argmax
        correct += (y_pred_argmax.data == label.data - 1).sum()
        num_examples += text.size(1)
    return nll/num_examples, correct/num_examples

class CNN(nn.Module):

    def __init__(self, vocab, seq_len, embedding_dim, output_dim=2):
        super(CNN, self).__init__()
        self.embed = nn.Embedding(len(vocab), embedding_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        map_size = 100
        filter_w = embedding_dim
        filter_h = 4
        self.conv1 = nn.Conv2d(1, map_size, (filter_h, filter_w))
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        pool_h = seq_len-filter_h + 1
        pool_w = 1
        self.pooling = nn.MaxPool2d((pool_h, pool_w))
        self.fc = nn.Linear(map_size, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        # here x is batch size x length of post X embedding dim
        x_embed = self.embed(x.t())
        #print(x_embed.shape)
        fc = self.conv1(x_embed.unsqueeze(1))
        pool = self.pooling(self.dropout(fc))
        relu = self.relu(pool)
        out = self.fc(pool.view(x_embed.size(0), -1))
        out = self.sigmoid(out)
        return self.logsoftmax(out)

def run_CNN():
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

    cnn_model = CNN(TEXT.vocab, seq_len=seq_length, embedding_dim=300)
    criterion = nn.NLLLoss()
    parameters = filter(lambda p: p.requires_grad, cnn_model.parameters())
    optim = torch.optim.SGD(parameters, lr = 0.1)
    num_epochs = 20
    for e in range(num_epochs):
        for batch in train_iter:
            optim.zero_grad()
            text = batch.text
            label = batch.label
            y_pred = cnn_model(text)
            nll_batch = criterion(y_pred, label-1)
            nll_batch.backward()
            optim.step()
        nll_train, accuracy_train = test_cnn(cnn_model, train_iter)
        nll_val, accuracy_val = test_cnn(cnn_model, val_iter)
        print('Training performance after epoch %d: NLL: %.4f, Accuracy: %.4f'% (e+1, nll_train, accuracy_train))
        print('Validation performance after epoch %d: NLL: %.4f, Accuracy: %.4f'% (e+1, nll_val, accuracy_val))


if __name__ == "__main__":
    run_CNN()