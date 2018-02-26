import torch
import random
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
import numpy as np
import spacy

SOS_token = 2
EOS_token = 3
teacher_forcing_ratio = 0.5


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, h_size, batch_size, n_layers=1, dropout=0):
        super(EncoderLSTM, self).__init__()
        self.num_layers = n_layers
        self.hidden_size = h_size
        self.batch_size = batch_size
        self.embed = nn.Embedding(input_size, h_size)
        self.lstm = nn.LSTM(h_size, h_size, dropout=dropout, num_layers=n_layers)

    def forward(self, input_src, hidden):
        embedded = self.embed(input_src)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        result = (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)),
                  Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)))
        if use_cuda:
            return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).cuda(),
                    Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).cuda())
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, dropout=0.1, n_layers=1, max_length=MAX_LEN):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.max_length = max_length
        self.num_layers = n_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, dropout=dropout, num_layers=n_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_data, hidden, encoder_outputs):
        # seq_len x batch_size x hidden_dim
        embedded = self.embedding(input_data)
        embedded = self.dropout(embedded)

        # embedded[0] is batch_size x hidden_dim
        # hidden[0] is batch_size x hidden_dim
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0][0]), 1)), dim=1)

        # atten_weights is batch_size x max_len
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.permute(1, 0, 2))
        # attn_applied is 1 x batch_size x hidden_dim

        output = torch.cat((embedded[0], attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self):
        result = (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)),
                  Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)))
        if use_cuda:
            return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).cuda(),
                    Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).cuda())
        else:
            return result


def train_batch(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
                max_length=MAX_LEN):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    batch_size = target_variable.size()[1]
    loss = 0
    if batch_size != 32:
        return loss

    encoder_output_short, encoder_hidden = encoder(input_variable, encoder_hidden)

    encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    encoder_outputs[:input_length, :, :] = encoder_output_short

    decoder_input = Variable(torch.ones(1, batch_size).long()) * SOS_token
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for t in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = target_variable[t].view(1, -1)
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            loss += criterion(decoder_output.squeeze(0), target_variable[t]) / target_length
    else:
        for t in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = torch.max(decoder_output, 1)
            decoder_input = topi.view(1, -1)
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            loss += criterion(decoder_output.squeeze(0), target_variable[t]) / target_length

    loss.backward()

    torch.nn.utils.clip_grad_norm(encoder.parameters(), 5.0)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), 5.0)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0]


def validate(encoder, decoder, val_iter, criterion, max_length=MAX_LEN):
    total_loss = 0
    num_batches = 0
    for batch in iter(val_iter):

        input_length = batch.src.size()[0]
        target_length = batch.trg.size()[0]
        batch_size = batch.src.size()[1]
        if batch_size != 32:
            break
        encoder_hidden = encoder.init_hidden()

        encoder_output_short, encoder_hidden = encoder(batch.src, encoder_hidden)

        encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        encoder_outputs[:input_length, :, :] = encoder_output_short

        decoder_input = Variable(torch.ones(1, batch_size).long() * SOS_token)
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        decoded_words = []
        loss = 0
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = torch.max(decoder_output, 1)
            decoded_words.append(topi)
            decoder_input = topi.view(1, -1)
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            loss += criterion(decoder_output.squeeze(0), batch.trg[di])

        total_loss += loss.data[0] / target_length
        num_batches += 1
    return total_loss / num_batches


def trainIters(encoder, decoder, training_iter, valid_iter, learning_rate=1.0, num_epochs=20):
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=1)

    for e in range(num_epochs):
        batch_len = 0
        total_loss = 0
        for batch in iter(training_iter):
            loss = train_batch(batch.src, batch.trg, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            total_loss += loss
            batch_len += 1
        train_loss = total_loss / batch_len
        print("train loss: ", train_loss)
        print("train ppl: ", np.exp(train_loss))
        val_loss = validate(encoder, decoder, valid_iter, criterion)
        print("val loss: ", val_loss)
        print("val ppl: ", np.exp(val_loss))
        torch.save(encoder.state_dict(), 'attn_encoder_model.pt')
        torch.save(decoder.state_dict(), 'attn_decoder_model.pt')


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

if __name__ == '__main__':
    # Set up
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    DE = data.Field(tokenize=tokenize_de, init_token=BOS_WORD, eos_token=EOS_WORD)
    EN = data.Field(tokenize=tokenize_en, init_token=BOS_WORD, eos_token=EOS_WORD)  # only target needs BOS/EOS

    MAX_LEN = 20
    train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN),
                                             filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                                                   len(vars(x)['trg']) <= MAX_LEN)
    print(train.fields)
    print(len(train))
    print(vars(train[0]))

    MIN_FREQ = 5
    DE.build_vocab(train.src, min_freq=MIN_FREQ)
    EN.build_vocab(train.trg, min_freq=MIN_FREQ)
    print(DE.vocab.freqs.most_common(10))
    print("Size of German vocab", len(DE.vocab))
    print(EN.vocab.freqs.most_common(10))
    print("Size of English vocab", len(EN.vocab))
    print(DE.vocab.stoi["<s>"], DE.vocab.stoi["</s>"])  # vocab index for <s>, </s>
    print(EN.vocab.stoi["<s>"], EN.vocab.stoi["</s>"])  # vocab index for <s>, </s>

    BATCH_SIZE = 32
    SOS_token = 2
    EOS_token = 3
    teacher_forcing_ratio = 0.5

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    if use_cuda:
        train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=0,
                                                          repeat=False, sort_key=lambda x: len(x.src))
    else:
        train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=-1,
                                                         repeat=False, sort_key=lambda x: len(x.src))

    hidden_size = 256
    encoder1 = EncoderLSTM(len(DE.vocab), hidden_size, batch_size=32, dropout=0.3, n_layers=2)
    decoder1 = AttnDecoderRNN(hidden_size, len(EN.vocab), batch_size=32, dropout=0.3, n_layers=2)
    if use_cuda:
        encoder1 = encoder1.cuda()
        decoder1 = decoder1.cuda()
    trainIters(encoder1, decoder1, train_iter, val_iter, num_epochs=10)
