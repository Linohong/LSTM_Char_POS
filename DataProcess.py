import nltk
import torch
from torch.autograd import Variable
from nltk.corpus import brown

# Loading the Data from Brown Corpus
print('Loading the Brown Corpus ...')
numSent = 0
training_data = []

#for sent in brown.tagged_sents(categories=['news', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned', 'lore', 'mystery', 'religion', 'reviews', 'romance', 'science_fiction']) :
for sent in brown.tagged_sents(categories=['news', 'editorial']) :
    training_data.append(sent)

print('Done Loading !')

# Indexing the Brown Corpus
print('Indexing Every Word and Tag ...')
word_to_idx = {}
tag_to_idx = {}
char_to_idx = {}
for sent in training_data :
    for word, tag in sent :
        if word not in word_to_idx :
            word_to_idx[word] = len(word_to_idx)
        if tag not in tag_to_idx :
            tag_to_idx[tag] = len(tag_to_idx)

for key in word_to_idx :
    for char in key :
        if char not in char_to_idx :
            char_to_idx[char] = len(char_to_idx)
print('Done Indexing !')

# make hyper parameters
char_size = len(char_to_idx)
vocab_size = len(word_to_idx)
tag_size = len(tag_to_idx)

print(len(training_data))

'''
print("char size : " + str(char_size))
print("vocab size : " + str(vocab_size))
print("tag size : " + str(tag_size))
'''

WORD_EMBEDDING_DIM = 32
CHAR_EMBEDDING_DIM = 32

EMB_HIDDEN_DIM = 32 # embedding lstm hidden dim
HIDDEN_DIM = 32

# etc. functions
def make_index_list(sentence, word_or_tag_or_char) :
    if word_or_tag_or_char == 'word' :
        return Variable(torch.cuda.LongTensor([word_to_idx[word] for word, _ in sentence]))
    elif word_or_tag_or_char == 'tag':
        return Variable(torch.cuda.LongTensor([tag_to_idx[tag] for _, tag in sentence]))
    else :
        word_into_char_list = []
        for word, _ in sentence :
            word_into_char_list.append(Variable(torch.cuda.LongTensor([char_to_idx[char] for char in word])))
        return word_into_char_list