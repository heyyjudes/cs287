import torch
import torchtext
import torch.nn as nn
from torchtext.vocab import GloVe


def test_cbow(model, data):
    correct = 0.
    num_examples = 0.
    nll = 0.
    criterion = nn.NLLLoss()
    for batch in data:
        text = batch.text
        label = batch.label
        y_pred = model(text)
        nll_batch = criterion(y_pred, label - 1)
        nll += nll_batch.data[0] * text.size(0)  # by default NLL is averaged over each batch
        y_pred_max, y_pred_argmax = torch.max(y_pred, 1)  # prediction is the argmax
        correct += (y_pred_argmax.data == label.data - 1).sum()
        num_examples += text.size(1)
    return nll / num_examples, correct / num_examples


class CBOW(nn.Module):
    def __init__(self, vocab, embedding_dim, output_dim=2):
        super(CBOW, self).__init__()
        # linear classifier
        self.embed = nn.Embedding(len(vocab), embedding_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.weight.requires_grad = False
        self.linear = nn.Linear(embedding_dim, output_dim, bias=True)
        # activation function
        self.sigmoid = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        x_embed = self.embed(x.t())
        x_flatten = torch.sum(x_embed, dim=1)
        out = self.linear(x_flatten)
        out = self.sigmoid(out)
        return self.logsoftmax(out)

def run_CBOW():
    # Our input $x$
    TEXT = torchtext.data.Field()
    # Our labels $y$
    LABEL = torchtext.data.Field(sequential=False)

    train, val, test = torchtext.datasets.SST.splits(
        TEXT, LABEL,
        filter_pred=lambda ex: ex.label != 'neutral')

    TEXT.build_vocab(train)
    LABEL.build_vocab(train)
    #this is just the set of stuff
    print('len(TEXT.vocab)', len(TEXT.vocab))
    print('len(LABEL.vocab)', len(LABEL.vocab))

    TEXT.vocab.load_vectors(vectors=GloVe(name="6B", dim="300"))

    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train, val, test), batch_size=10, device=-1, repeat=False)

    cbow_model = CBOW(TEXT.vocab, embedding_dim=300)
    print(cbow_model)
    criterion = nn.NLLLoss()
    parameters = filter(lambda p: p.requires_grad, cbow_model.parameters())
    optim = torch.optim.SGD(parameters, lr=0.5)
    num_epochs = 20
    for e in range(num_epochs):
        for batch in train_iter:
            optim.zero_grad()
            text = batch.text
            label = batch.label
            y_pred = cbow_model(text)
            nll_batch = criterion(y_pred, label - 1)
            nll_batch.backward()
            optim.step()
        nll_train, accuracy_train = test_cbow(cbow_model, train_iter)
        nll_val, accuracy_val = test_cbow(cbow_model, val_iter)
        print('Training performance after epoch %d: NLL: %.4f, Accuracy: %.4f' % (e + 1, nll_train, accuracy_train))
        print('Validation performance after epoch %d: NLL: %.4f, Accuracy: %.4f' % (e + 1, nll_val, accuracy_val))

if __name__ == "__main__":
    run_CBOW()