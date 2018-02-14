import torchtext
import math
from collections import Counter
from nltk.util import ngrams
from torchtext.vocab import Vectors


def preprocessing_counting(t_iter):
    count_uni = Counter()
    count_bi = Counter()
    count_tri = Counter()
    
    total_uni = 0
    
    
    for batch in t_iter:
        for i in range(batch.text.size()[1]):
            sentence = [TEXT.vocab.itos[i] for i in batch.text[:, i].data]
            count_uni += Counter(sentence)
            
            bigrams = ngrams(sentence,2,pad_right=True)
            count_bi += Counter(bigrams)
            
            trigrams = ngrams(sentence,3,pad_right=True)
            count_tri += Counter(trigrams)
            
            total_uni += len(sentence)
            
            
    return count_uni, count_bi, count_tri, total_uni


def preprocessing_calc_p(count_uni, count_bi, count_tri, total_uni):
    
    prob_uni = {}
    prob_bi = {}
    prob_tri = {}
    running_sum = 0 
    
    total_uni
    
    for word in count_uni:
        prob_uni[word] = count_uni[word]/total_uni
        
    for bigram in count_bi:
        w1 = bigram[0]
        w2 = bigram[1]
        prob_bi[(w1, w2)] = count_bi[(w1, w2)]/count_uni[w1]
        
    for trigram in count_tri:
        w1 = trigram[0]
        w2 = trigram[1]
        w3 = trigram[2]
        prob_tri[(w1, w2, w3)] = count_tri[(w1, w2, w3)]/count_bi[(w1, w2)]
            
    return prob_uni, prob_bi, prob_tri


def generate_next_words(test_iter, n_words, a1, a2, prob_uni, prob_bi, prob_tri):
    
    predictions_file = open("predictions_trigrammodel.txt", "w") 
    print("id,word", file=predictions_file)
    
    count = -1
    
    for batch in test_iter:
        
        for i in range(batch.text.size()[1]):
            count += 1
            sentence = [TEXT.vocab.itos[i] for i in batch.text[:, i].data]
            w_n_minus_1 = sentence[-1]
            w_n_minus_2 = sentence[-2]
            
            
            prob = {}
     
            #p(yt|y1:t−1)=α1p(yt|yt−2,yt−1)+α2p(yt|yt−1)+(1−α1−α2)p(yt)
            for word in TEXT.vocab.freqs.keys():
                try:
                    part_1 = a1 * prob_tri[(w_n_minus_2, w_n_minus_1, word)]
                except:
                    part_1 = 0
                
                try:
                    part_2 = a2 * prob_bi[(w_n_minus_1, word)]
                except:
                    part_2 = 0
                    
                try:
                    part_3 = (1-a1-a2) * prob_uni[word]
                except:
                    part_3 = 0
                
                prob[word] = part_1 + part_2 + part_3
                if prob[word] > 1:
                    print (word)
                
            
            top_words = sorted(prob.items(), key=lambda x:-x[1])[:n_words]
            
            
            print("%d,%s"%(count, " ".join([w[0] for w in top_words])), file=predictions_file)


def calculate_perplexity(text_iter, prob_uni, prob_bi, prob_tri, a1, a2):
    
    N = 0
    running_prob = 1
    
    for batch in text_iter:
        
        for i in range(batch.text.size()[1]):
            sentence = [TEXT.vocab.itos[i] for i in batch.text[:, i].data]
            
            N_sentence = len(sentence)-2
            
            for word_i in range(N_sentence):
                N += 1
                word = sentence[word_i+2]
                w_n_minus_1 = sentence[word_i+1]
                w_n_minus_2 = sentence[word_i]
                
                #p(yt|y1:t−1)=α1p(yt|yt−2,yt−1)+α2p(yt|yt−1)+(1−α1−α2)p(yt)
                try:
                    part_1 = a1 * prob_tri[(word, w_n_minus_1, w_n_minus_2)]
                except:
                    part_1 = 0
                
                try:
                    part_2 = a2 * prob_bi[(word, w_n_minus_1)]
                except:
                    part_2 = 0
                    
                try:
                    part_3 = (1-a1-a2) * prob_uni[word]
                except:
                    part_3 = 0

                probability = part_1 + part_2 + part_3
                
                running_prob += math.log(probability)
                
    return math.exp(-1/N * running_prob)



count_uni, count_bi, count_tri, total_uni = preprocessing_counting(train_iter)
prob_uni, prob_bi, prob_tri = preprocessing_calc_p(count_uni, count_bi, count_tri, total_uni)

generate_next_words(test_iter, 20, .25, .25, prob_uni, prob_bi, prob_tri)

print (calculate_perplexity(val_iter, prob_uni, prob_bi, prob_tri, 0.25, 0.25))


