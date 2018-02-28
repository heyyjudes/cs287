import torch
import random
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
import spacy
import numpy as np

BATCH_SIZE = 64
SOS_token = 2
EOS_token = 3
PAD_token = 1
MAX_LEN = 20 
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


class DecoderLSTM(nn.Module):
    def __init__(self, h_size, output_size, batch_size, n_layers=1, dropout=0):
        super(DecoderLSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = h_size

        self.embedding = nn.Embedding(output_size, h_size)
        self.lstm = nn.LSTM(h_size, h_size, dropout=dropout, num_layers=n_layers)
        self.out = nn.Linear(h_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input_data, hidden):
        output = self.embedding(input_data)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.out(self.dropout(output))
        output = self.softmax(output)
        return output, hidden


def train_batch(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LEN):
    encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    #useful lengths 
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    batch_size = target_variable.size()[1]
    
    # zero words and zero loss 
    loss = 0 
    total_words = 0 
    
    encoder_output_short, encoder_hidden = encoder(input_variable, encoder_hidden)
    
    encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    
    # NO ATTN: works fine 
    decoder_hidden = encoder_hidden
    decoder_output, decoder_hidden = decoder(target_variable, decoder_hidden) 

    #initialize last row as padding tokens
    last_row = torch.ones(1, batch_size).long()
    last_row = last_row.cuda() if use_cuda else last_row
    
    #shift target from 1:n + padding row
    shifted_target = Variable(torch.cat((target_variable[1:, :].data.long(), last_row)))
    m, i = torch.max(decoder_output, dim=2)

    #calculate decoder_output loss with shifted target loss

    loss = criterion(decoder_output.view(target_length*batch_size, -1), shifted_target.view(target_length*batch_size))
    # count total words
    total_words = shifted_target.ne(PAD_token).int().sum()

    loss.backward()

    torch.nn.utils.clip_grad_norm(encoder.parameters(), 3.0)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), 3.0)

    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.data[0]/total_words.data[0]


def validate(encoder, decoder, val_iter, criterion, max_length = MAX_LEN):
    encoder.eval() 
    decoder.eval() 
    total_loss = 0
    num_batches = 0
    total_words = 0 
    for batch in iter(val_iter):
        num_batches += 1 
        input_length = batch.src.size()[0]
        target_length = batch.trg.size()[0]
        batch_size = batch.src.size()[1]
        if batch_size != 64:
            break
        encoder_hidden = encoder.init_hidden()
         
        decoder_hidden = encoder_hidden
        decoder_output, decoder_hidden = decoder(batch.trg, decoder_hidden) 
           
        m, i = torch.max(decoder_output, dim=2)

        first_row = torch.ones(1, batch_size).long()
        first_row = first_row.cuda() if use_cuda else first_row
        
        shifted_target = Variable(torch.cat((batch.trg[1:, :].data.long(), first_row)))
        loss = criterion(decoder_output.view(target_length*batch_size, -1), shifted_target.view(target_length*batch_size))
        total_words += shifted_target.ne(PAD_token).int().sum()

        total_loss += loss.data[0]

    return total_loss / total_words.data[0]

def trainIters(encoder, decoder, training_iter, valid_iter, target_vocab_len, learning_rate=0.7, num_epochs=20):
    encoder.train() 
    decoder.train() 
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    #mask weight to not consider in loss 
    mask_weight = Variable(torch.FloatTensor(target_vocab_len).fill_(1))
    mask_weight[PAD_token] = 0
    mask_weight = mask_weight.cuda() if use_cuda else mask_weight
    
    #pass mask weight in to NLL Loss without size_average
    criterion = nn.NLLLoss(weight=mask_weight, size_average=False)
    
    for e in range(num_epochs):
        #initialise total loss and batch count
        batch_len = 0
        total_loss = 0
        
        for batch in iter(training_iter):
            if batch.src.size()[1] == 64: 
                loss = train_batch(batch.src, batch.trg, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
                total_loss += loss
                batch_len += 1
            
        # divide total loss by batch_length
        train_loss = total_loss / batch_len
        
        print("train loss: ", train_loss)
        print("train ppl: ", np.exp(train_loss))
        val_loss = validate(encoder, decoder, valid_iter, criterion)
        print("val loss: ", val_loss)
        print("val ppl: ", np.exp(val_loss))
        torch.save(encoder.state_dict(), 'encoder_model.pt')
        torch.save(decoder.state_dict(), 'decoder_model.pt')


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

if __name__ == '__main__':
    # Set up
    # Set up 
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    DE = data.Field(tokenize=tokenize_de)
    EN = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, eos_token = EOS_WORD) # only target needs BOS/EOS

    MAX_LEN = 20
    train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN), 
                                             filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
                                             len(vars(x)['trg']) <= MAX_LEN)


    MIN_FREQ = 5
    DE.build_vocab(train.src, min_freq=MIN_FREQ)
    EN.build_vocab(train.trg, min_freq=MIN_FREQ)
    print(DE.vocab.freqs.most_common(10))
    print("Size of German vocab", len(DE.vocab))
    print(EN.vocab.freqs.most_common(10))
    print("Size of English vocab", len(EN.vocab))
    # print(DE.vocab.stoi["<s>"], DE.vocab.stoi["</s>"]) #vocab index for <s>, </s>
    print(EN.vocab.stoi["<s>"], EN.vocab.stoi["</s>"]) #vocab index for <s>, </s>
    print(EN.vocab.stoi["<pad>"])


    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    if use_cuda:
        train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=0,
                                                          repeat=False, sort_key=lambda x: len(x.src))
    else:
        train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=-1,
                                                         repeat=False, sort_key=lambda x: len(x.src))

    hidden_size = 500
    encoder1 = EncoderLSTM(len(DE.vocab), hidden_size, batch_size=64, dropout=0.3, n_layers=2)
    decoder1 = DecoderLSTM(hidden_size, len(EN.vocab), batch_size=64, dropout=0.3, n_layers=2)
    if use_cuda:
        encoder1 = encoder1.cuda()
        decoder1 = decoder1.cuda()
    trainIters(encoder1, decoder1, train_iter, val_iter, len(EN.vocab), num_epochs=10)
