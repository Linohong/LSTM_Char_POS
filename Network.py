import torch
import torch.nn as nn
import torch.nn.functional as F
import DataProcess as D
from torch.autograd import Variable

class LSTMChar(nn.Module) :
    def __init__ (self) :
        super(LSTMChar, self).__init__()
        # Embedding Layer for Characters and Words respectively
        self.char_embeddings = nn.Embedding(D.char_size, D.CHAR_EMBEDDING_DIM)
        self.word_embeddings = nn.Embedding(D.vocab_size, D.WORD_EMBEDDING_DIM)

        # LSTM Layer : input as a character into this layer outputs embedding for word, Cw
        self.EMB_lstm = nn.LSTM(D.CHAR_EMBEDDING_DIM, D.EMB_HIDDEN_DIM)
        self.POS_lstm = nn.LSTM(D.EMB_HIDDEN_DIM + D.WORD_EMBEDDING_DIM, D.HIDDEN_DIM)

        # hidden layer
        self.hidden2tag = nn.Linear(D.HIDDEN_DIM, D.tag_size) # this is a linear function
        self.EMB_hidden = self.init_hidden('emb')
        self.POS_hidden = self.init_hidden('pos')

    def init_hidden(self, emb_or_pos) :
        if emb_or_pos == 'emb' :
            # EL stands for Embedding_LSTM
            return (Variable(torch.zeros(1, 1, D.EMB_HIDDEN_DIM).cuda()), Variable(torch.zeros(1, 1, D.EMB_HIDDEN_DIM).cuda()))
        else :
            return (Variable(torch.zeros(1, 1, D.HIDDEN_DIM).cuda()), Variable(torch.zeros(1, 1, D.HIDDEN_DIM).cuda()))

    def forward(self, sentence, char_list) : # this is list of indexes into Variable
        # firstly, make embeddings for characters for each word in a sentence
        char_embed_list = None
        for i, _ in enumerate(sentence) :
            char_embeds = self.char_embeddings(char_list[i]).view(len(char_list[i]), 1, -1) # input to the PyTorch embedding class should be Variable which encloses torches of indices
            EMB_lstm_out, self.EMB_hidden = self.EMB_lstm(char_embeds, self.EMB_hidden)
            EMB_lstm_out = EMB_lstm_out[-1].view(1, -1)
            if i == 0 :
                char_embed_list = EMB_lstm_out
            else :
                char_embed_list = torch.cat((char_embed_list, EMB_lstm_out), 0)
            self.EMB_hidden = self.init_hidden('emb')

        # (Concatenate) word_embeds = Word embedding + Character-based embedding
        word_embeds = self.word_embeddings(sentence)
        word_embeds = torch.cat((word_embeds, char_embed_list), 1)

        # Now Start the POS Tagging
        POS_lstm_out, self.POS_hidden = self.POS_lstm(word_embeds.view(len(sentence), 1, -1), self.POS_hidden)
        tag_space = self.hidden2tag(POS_lstm_out.view(len(sentence), -1)) # affine layer
        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores





