n_embedding = 30
n_hidden = 30
seq_len = 32
batch_size = 12

# Our input $x$
TEXT = torchtext.data.Field()
# Data distributed with the assignment
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path=".", 
    train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)
TEXT.build_vocab(train)
print('len(TEXT.vocab)', len(TEXT.vocab))

lstm_model = LSTM(len(TEXT.vocab), n_embedding, n_hidden, seq_len, batch_size)
lstm_model.load_state_dict(torch.load('LSTM_small_model.pt'))
with open("sample.txt", "w") as fout: 
    print("id,word", file=fout)
    for i, l in enumerate(open("input.txt"), 1):
        print(l)
        input_tokens = l.split(" ")[:-1]
        input_index = torch.LongTensor([TEXT.vocab.stoi[t] for t in input_tokens]).unsqueeze(1)
        lstm_model.hidden = lstm_model.init_hidden()
        output = lstm_model(torch.autograd.Variable(input_index))
        clean_output = output[-1, 0, :].view(-1, lstm_model.vocab_dim)
        max_values, max_indices = torch.topk(clean_output, 20)
        print(" ".join([TEXT.vocab.itos[int(i)] for i in max_indices.data[0, :]]))

