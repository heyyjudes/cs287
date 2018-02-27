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

BATCH_SIZE =64
MAX_LEN = 20
SOS_token = 2
EOS_token = 3
PAD_token = 1 
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

class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()

        self.hidden_size = hidden_size


    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S

        if use_cuda:
            attn_energies = attn_energies.cuda()
            
       # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):  
        energy = hidden.dot(encoder_output)
        return energy
        
        


def train_batch(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LEN):
    encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    encoder.train() 
    decoder.train()
    
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
    
    #pad encoder outputs for attention 
    encoder_outputs[:input_length, :, :] = encoder_output_short 

    decoder_hidden = encoder_hidden
    
    #initialize last row as padding tokens
    last_row = torch.ones(1, batch_size).long()
    last_row = last_row.cuda() if use_cuda else last_row
    
    shifted_target = Variable(torch.cat((target_variable[1:, :].data.long(), last_row)))
    output_words = Variable(torch.ones(target_length, batch_size))
    output_words = output_words.cuda() if use_cuda else output_words
    
    for t in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            target_variable[t].view(1, -1), decoder_hidden, encoder_outputs)

#         debug attention
#         v, i = torch.max(decoder_attention.squeeze(1), 1)
#         print(i)
        
        v, i = torch.max(decoder_output.squeeze(1), 1)
        output_words[t] = i
        
        loss += criterion(decoder_output.squeeze(0), shifted_target[t])
        total_words += shifted_target[t].ne(PAD_token).int().sum()

#     debug output 
#     print(output_words)
#     print(shifted_target)

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
        
        input_length = batch.src.size()[0]
        target_length = batch.trg.size()[0]
        batch_size = batch.src.size()[1]
        if batch_size != 64:
            break
        encoder_hidden = encoder.init_hidden()
        
        encoder_output_short, encoder_hidden = encoder(batch.src, encoder_hidden)
    
        encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        encoder_outputs[:input_length, :, :] = encoder_output_short      

        decoder_hidden = encoder_hidden
            #initialize last row as padding tokens
        last_row = torch.ones(1, batch_size).long()
        last_row = last_row.cuda() if use_cuda else last_row

        shifted_target = Variable(torch.cat((batch.trg[1:, :].data.long(), last_row)))
        output_words = Variable(torch.ones(target_length, batch_size))
        output_words = output_words.cuda() if use_cuda else output_words
        loss = 0 
        for t in range(target_length):
            decoder_input = batch.trg[t].view(1, -1)
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
        
            loss += criterion(decoder_output.squeeze(0), shifted_target[t])
            total_words += shifted_target[t].ne(PAD_token).int().sum()

        total_loss += loss.data[0]

    return total_loss / total_words.data[0]


def trainIters(encoder, decoder, training_iter, valid_iter, target_vocab_len, learning_rate=0.7, num_epochs=20):
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    #mask weight to not consider in loss 
    mask_weight = Variable(torch.FloatTensor(target_vocab_len).fill_(1))
    mask_weight[PAD_token] = 0
    mask_weight = mask_weight.cuda() if use_cuda else mask_weight
    
    #pass mask weight in to NLL Loss without size_average
    criterion = nn.NLLLoss(weight=mask_weight, size_average=False)
    val_loss = validate(encoder, decoder, valid_iter, criterion)
    print("val loss: ", val_loss)
    print("val ppl: ", np.exp(val_loss))
    
    for e in range(num_epochs):
        #initialise total loss and batch count
        batch_len = 0
        total_loss = 0
        for batch in iter(training_iter):
            if batch.src.size()[1] == 64: 
                loss = train_batch(batch.src, batch.trg, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
                print(loss)
                total_loss += loss
                batch_len += 1
            
        # divide total loss by batch_length
        train_loss = total_loss / batch_len
        
        print("train loss: ", train_loss)
        print("train ppl: ", np.exp(train_loss))
        val_loss = validate(encoder, decoder, valid_iter, criterion)
        print("val loss: ", val_loss)
        print("val ppl: ", np.exp(val_loss))
        torch.save(encoder.state_dict(), 'attn_encoder_model.pt')
        torch.save(decoder.state_dict(), 'attn_decoder_model.pt')
        
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
        self.attn = Attn(hidden_size)
        
        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(self.hidden_size*2, self.hidden_size, num_layers=n_layers, dropout=dropout)
        self.out = nn.Linear(self.hidden_size*2, self.output_size)

    def forward(self, input_data, hidden, encoder_outputs):
        #input_len x batch_size 
        embedded = self.embedding(input_data) #1 x batch_size x hidden dim
        
        embedded = self.dropout(embedded)
        
        #embedded[0] is 1 x batch_size x hidden_dim  
        #hidden[0] hn is 1 x batch_size x hidden_dim 
        
        # Calculate attention weights and apply to encoder outputs

        attn_weights = self.attn(hidden[0], encoder_outputs)

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        context = context.transpose(0, 1)

        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((embedded, context), 2)
        output, hidden = self.lstm(rnn_input, hidden)
        # Final output layer
        output = output.squeeze(0)
        output = self.out(torch.cat((output, context.squeeze(0)), 1))
        output = F.log_softmax(output, dim=1)

        return output, hidden, attn_weights

    def init_hidden(self):
        result = (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)),
                  Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)))
        if use_cuda:
            return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).cuda(),
                    Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).cuda())
        else:
            return result



def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

if __name__ == '__main__':
    # Set up
    # Set up 
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    DE = data.Field(tokenize=tokenize_de)
    EN = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, eos_token = EOS_WORD) # only target needs BOS/EOS

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
    hidden_size = 200
    encoder1 = EncoderLSTM(len(DE.vocab), hidden_size, batch_size=64, dropout=0.3, n_layers=1)
    decoder1 = AttnDecoderRNN(hidden_size, len(EN.vocab), batch_size=64, dropout=0.3, n_layers=1)
    if use_cuda:
        encoder1 = encoder1.cuda()
        decoder1 = decoder1.cuda()
    trainIters(encoder1, decoder1, train_iter, val_iter, len(EN.vocab), num_epochs=8)
