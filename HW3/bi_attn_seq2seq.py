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


MAX_LEN = 20
BATCH_SIZE = 64
SOS_token = 2
EOS_token = 3
PAD_token = 1 
teacher_forcing_ratio = 0.5


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, h_size, batch_size, n_layers=1, dropout=0, bidir=False):
        super(EncoderLSTM, self).__init__()
        self.num_layers = n_layers
        self.hidden_size = h_size
        self.batch_size = batch_size
        self.bidir=bidir
        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(input_size, h_size)
        self.lstm = nn.LSTM(h_size, h_size, dropout=dropout, num_layers=n_layers, bidirectional=bidir)

    def forward(self, input_src, hidden):
        embedded = self.embed(input_src)
        embedded = self.dropout(embedded)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        if self.bidir: 
           num_dir = 2
        else: 
           num_dir = 1      
        result = (Variable(torch.zeros(self.num_layers*num_dir , self.batch_size, self.hidden_size)),
                  Variable(torch.zeros(self.num_layers*num_dir, self.batch_size, self.hidden_size)))
        if use_cuda:
            return (Variable(torch.zeros(self.num_layers*num_dir, self.batch_size, self.hidden_size)).cuda(),
                    Variable(torch.zeros(self.num_layers*num_dir, self.batch_size, self.hidden_size)).cuda())
        else:
            return result


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        # hidden -> target_len x batch_size x hidden_dim
        hidden = hidden.transpose(0, 1) # batch_size x target_len x hidden_dim
        
        # encoder_outputs -> max_len x batch_size x hidden_dim
        encoder_outputs = encoder_outputs.permute(1, 2, 0)
        
        attn_energies = torch.bmm(hidden, encoder_outputs) # B x S
        

        return F.softmax(attn_energies, dim=2)
        

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, dropout=0.1, n_layers=1, max_length=MAX_LEN):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size*2
        self.output_size = output_size
        self.max_length = max_length
        self.num_layers = n_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = Attn(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=n_layers, dropout=dropout)
        self.out = nn.Linear(self.hidden_size*2, self.output_size)

    def forward(self, input_data, hidden, encoder_outputs):
        #input_len x batch_size 

        embedded = self.embedding(input_data) #batch_size x target_len x hidden dim
        embedded = F.relu(embedded)
        #lstm_output -> target_len x batch_size x hidden_dim
        lstm_output, lstm_hidden = self.lstm(embedded, hidden)

        #attn input 0 to T-1 
        if hidden[0].size()[0] != 1: 
            attn_hidden = hidden[0][-1].unsqueeze(0)
        else: 
            attn_hidden = hidden[0]
        
        if(lstm_output.size()[0] > 1):  
            attn_input = torch.cat((attn_hidden, lstm_output[:-1]))
        else: 
            attn_input = attn_hidden
        # encoder_outputs -> max_len x batch_size x hidden_dim
        attn_weights = self.attn(attn_input, encoder_outputs)
        
        # context = batch_size x target_length x hidden_dim 
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) 
        
        context = context.transpose(1, 0) #target_length x batch_size x hidden_dim
        
        output = torch.cat((lstm_output, context), 2)

        # Final output layer
        final_output = F.log_softmax(self.out(output), dim=2)
        final_output = self.dropout(final_output)
        return final_output, lstm_hidden, attn_weights



    def init_hidden(self):
        result = (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)),
                  Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)))
        if use_cuda:
            return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).cuda(),
                    Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).cuda())
        else:
            return result

def train_batch(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LEN):
    encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    #useful lengths 
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    batch_size = target_variable.size()[1]
    layers = encoder.num_layers
    # zero words and zero loss 
    loss = 0 
    total_words = 0 
    
    encoder_output_short, encoder_hidden = encoder(input_variable, encoder_hidden)
    
    encoder_outputs = Variable(torch.zeros(max_length, batch_size, 2*encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    
    #ATTENTION
    encoder_outputs[:input_length, :, :] = encoder_output_short 

    if layers != 1: 
        decoder_hidden = (torch.cat((encoder_hidden[0][-layers:], encoder_hidden[0][-layers:]), dim=2) , torch.cat((encoder_hidden[1][-layers:], encoder_hidden[1][-layers:]), dim=2)) 
    else: 
        decoder_hidden = (torch.cat((encoder_hidden[0][0].unsqueeze(0), encoder_hidden[0][1].unsqueeze(0)), dim=2) , torch.cat((encoder_hidden[1][0].unsqueeze(0), encoder_hidden[1][1].unsqueeze(0)), dim=2)) 
         
    #using encoder output and target variable 

    decoder_output, decoder_hidden, decoder_attention = decoder(target_variable, decoder_hidden, encoder_outputs)
    
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
    layers = encoder.num_layers
    for batch in iter(val_iter):
        num_batches += 1 
        input_length = batch.src.size()[0]
        target_length = batch.trg.size()[0]
        batch_size = batch.src.size()[1]
        if batch_size != 64:
            break
        encoder_hidden = encoder.init_hidden()
         
        encoder_output_short, encoder_hidden = encoder(batch.src, encoder_hidden)
        encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size*2))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        encoder_outputs[:input_length, :, :] = encoder_output_short      
        if layers != 1: 
            decoder_hidden = (torch.cat((encoder_hidden[0][-layers:], encoder_hidden[0][-layers:]), dim=2) , torch.cat((encoder_hidden[1][-layers:], encoder_hidden[1][-layers:]), dim=2)) 
        else: 
            decoder_hidden = (torch.cat((encoder_hidden[0][0].unsqueeze(0), encoder_hidden[0][1].unsqueeze(0)), dim=2) , torch.cat((encoder_hidden[1][0].unsqueeze(0), encoder_hidden[1][1].unsqueeze(0)), dim=2)) 
         
        decoder_output, decoder_hidden, decoder_attention = decoder(batch.trg, decoder_hidden, encoder_outputs)
        
        m, i = torch.max(decoder_output, dim=2)
                    
        first_row = torch.ones(1, batch_size).long()
        first_row = first_row.cuda() if use_cuda else first_row
        
        shifted_target = Variable(torch.cat((batch.trg[1:, :].data.long(), first_row)))
        loss = criterion(decoder_output.view(target_length*batch_size, -1), shifted_target.view(target_length*batch_size))
        total_words += shifted_target.ne(PAD_token).int().sum()

        total_loss += loss.data[0]

    return total_loss / total_words.data[0]


def trainIters(encoder, decoder, training_iter, valid_iter, target_vocab_len, learning_rate=1.0, num_epochs=20):
    encoder.train() 
    decoder.train() 
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    schedule = optim.lr_scheduler.ReduceLROnPlateau(
        decoder_optimizer, patience=1, factor=0.5, threshold=1e-4)

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
        
        torch.save(encoder.state_dict(), 'f_attn_encoder_model.pt')
        torch.save(decoder.state_dict(), 'f_attn_decoder_model.pt')


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
    hidden_size = 512
    encoder1 = EncoderLSTM(len(DE.vocab), hidden_size, batch_size=64, dropout=0.3, n_layers=2, bidir=True)
    decoder1 = AttnDecoderRNN(hidden_size, len(EN.vocab), batch_size=64, dropout=0.3, n_layers=2) 
    if use_cuda:
        encoder1 = encoder1.cuda()
        decoder1 = decoder1.cuda()
    trainIters(encoder1, decoder1, train_iter, val_iter, len(EN.vocab), num_epochs=10)

