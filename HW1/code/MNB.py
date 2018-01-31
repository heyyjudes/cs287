import torch
import torchtext

def batch_vectorize_init(word_ind, labels, vocab_dim):
    num_pos = (labels.data == 2).sum()
    num_neg = (labels.data == 1).sum()
    pos_out = torch.zeros(vocab_dim)
    neg_out = torch.zeros(vocab_dim)
    for j in range(word_ind.size(1)):
        curr_vec = torch.zeros(vocab_dim)
        for i in range(word_ind.size(0)):
            curr_vec[int(word_ind[i, j])] = 1
        # neg class
        if labels.data[j] == 1:
            neg_out += curr_vec
        elif labels.data[j] == 2:
            pos_out += curr_vec
        else:
            print("class not found")

    return pos_out, neg_out, num_pos, num_neg


def batch_vectorize(word_ind, vocab_size):
    out = torch.zeros(word_ind.size(1), vocab_size)
    for j in range(word_ind.size(1)):
        for i in range(word_ind.size(0)):
            out[j, int(word_ind[i, j])] = 1
    return out


def test_naive(data, vocab_size, R, b):
    correct = 0.
    num_examples = 0.
    nll = 0.
    for batch in data:
        if len(batch.label) == 10:
            text, label = batch.text, batch.label
            x = batch_vectorize(batch.text, vocab_size)
            y_pred = torch.mm(x, R.t()) + b
            y_pred_max, y_pred_argmax = torch.max(y_pred, 1)  # prediction is the argmax
            correct += (y_pred_argmax == label.data - 1).sum()
            num_examples += text.size(1)
    return correct / num_examples

def run_MNB():
    # Our input $x$
    TEXT = torchtext.data.Field()
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

    n_pos = 0
    n_neg = 0
    alpha = 0.5
    p = torch.zeros(len(TEXT.vocab))
    q = torch.zeros(len(TEXT.vocab))
    for batch in train_iter:
        text, label = batch.text, batch.label
        pos_vec, neg_vec, curr_pos, curr_neg = batch_vectorize_init(batch.text, batch.label, len(TEXT.vocab))
        n_pos += curr_pos
        n_neg += curr_neg
        p += pos_vec
        q += neg_vec
    n_pos_vec = torch.log(torch.Tensor([n_pos / (n_pos + n_neg)])).repeat(10)
    n_neg_vec = torch.log(torch.Tensor([n_neg / (n_pos + n_neg)])).repeat(10)
    b = torch.cat((n_neg_vec, n_pos_vec)).view(2, -1)
    p += alpha
    q += alpha
    R = torch.log(torch.cat((q / torch.abs(q).sum(), p / torch.abs(p).sum())).view(2, -1))

    print("train: ", test_naive(train_iter, len(TEXT.vocab), R, b))
    print("valid: ", test_naive(val_iter, len(TEXT.vocab), R, b))
    print("test: ", test_naive(test_iter, len(TEXT.vocab), R, b))

if __name__ == "__main__":
    run_MNB()