import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


tagged_sentence=[]
#Construction du texte global

first=0
for ligne in open("corpus-kab.txt",encoding='utf-8'):
    if (first!=0):
        sentence=[]
        line=ligne.split()

        for i in line:
            j=i.split('/')
            couple=(j[0],j[1])
            sentence.append(couple)
        tagged_sentence.append(sentence)
    first=1


def word_to_ix(word, ix):
    return torch.tensor(ix[word], dtype = torch.long)

def char_to_ix(char, ix):
    return torch.tensor(ix[char], dtype= torch.long)

def tag_to_ix(tag, ix):
    return torch.tensor(ix[tag], dtype= torch.long)

def sequence_to_idx(sequence, ix):
    return torch.tensor([ix[s] for s in sequence], dtype=torch.long)


word_to_idx = {}
tag_to_idx = {}
char_to_idx = {}
for sentence in tagged_sentence:
    for word, pos_tag in sentence:
        if word not in word_to_idx.keys():
            word_to_idx[word] = len(word_to_idx)
        if pos_tag not in tag_to_idx.keys():
            tag_to_idx[pos_tag] = len(tag_to_idx)
        for char in word:
            if char not in char_to_idx.keys():
                char_to_idx[char] = len(char_to_idx)

word_vocab_size = len(word_to_idx)
tag_vocab_size = len(tag_to_idx)
char_vocab_size = len(char_to_idx)

print("Unique words: {}".format(len(word_to_idx)))
print("Unique tags: {}".format(len(tag_to_idx)))
print("Unique characters: {}".format(len(char_to_idx)))


import random

tr_random = random.sample(list(range(len(tagged_sentence))), int(0.95 * len(tagged_sentence)))

train = [tagged_sentence[i] for i in tr_random]
test = [tagged_sentence[i] for i in range(len(tagged_sentence)) if i not in tr_random]

WORD_EMBEDDING_DIM = 1024
CHAR_EMBEDDING_DIM = 128
WORD_HIDDEN_DIM = 1024
CHAR_HIDDEN_DIM = 1024
EPOCHS = 5

class DualLSTMTagger(nn.Module):
    def __init__(self, word_embedding_dim, word_hidden_dim, char_embedding_dim, char_hidden_dim, word_vocab_size, char_vocab_size, tag_vocab_size):
        super(DualLSTMTagger, self).__init__()
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)

        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim)
        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim)

        self.lstm = nn.LSTM(word_embedding_dim + char_hidden_dim, word_hidden_dim)
        self.hidden2tag = nn.Linear(word_hidden_dim, tag_vocab_size)

    def forward(self, sentence, words):
        embeds = self.word_embedding(sentence)
        char_hidden_final = []
        for word in words:
            char_embeds = self.char_embedding(word)
            _, (char_hidden, char_cell_state) = self.char_lstm(char_embeds.view(len(word), 1, -1))
            word_char_hidden_state = char_hidden.view(-1)
            char_hidden_final.append(word_char_hidden_state)
        char_hidden_final = torch.stack(tuple(char_hidden_final))

        combined = torch.cat((embeds, char_hidden_final), 1)

        lstm_out, _ = self.lstm(combined.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))

        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
model = DualLSTMTagger(WORD_EMBEDDING_DIM, WORD_HIDDEN_DIM, CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM, word_vocab_size, char_vocab_size, tag_vocab_size)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if use_cuda:
    model.cuda()

# Define the loss function as the Negative Log Likelihood loss (NLLLoss)
loss_function = nn.NLLLoss()

# We will be using a simple SGD optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Asluɣmu
print("asluɣmu yebda")
accuracy_list = []
loss_list = []
interval = round(len(train) / 100.)
epochs = EPOCHS
e_interval = round(epochs / 10.)
for epoch in range(epochs):
    acc = 0 #to keep track of accuracy
    loss = 0 # To keep track of the loss value
    i = 0
    for sentence_tag in train:
        i += 1
        words = [torch.tensor(sequence_to_idx(s[0], char_to_idx), dtype=torch.long).to(device) for s in sentence_tag]
        sentence = [s[0] for s in sentence_tag]
        sentence = torch.tensor(sequence_to_idx(sentence, word_to_idx), dtype=torch.long).to(device)
        targets = [s[1] for s in sentence_tag]
        targets = torch.tensor(sequence_to_idx(targets, tag_to_idx), dtype=torch.long).to(device)

        model.zero_grad()

        tag_scores = model(sentence, words)

        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
        loss += loss.item()
        _, indices = torch.max(tag_scores, 1)
#         print(indices == targets)
        acc += torch.mean(torch.tensor(targets == indices, dtype=torch.float))
        if i % interval == 0:
            print("Epoch {} Iteddu;\t{}% Yemmed".format(epoch + 1, i / interval), end = "\r", flush = True)
    loss = loss / len(train)
    acc = acc / len(train)
    loss_list.append(float(loss))
    accuracy_list.append(float(acc))
    if (epoch + 1) % e_interval == 0:
        print("Epoch {} Yemmed,\tasṛuḥu {}\tTiseddi: {}".format(epoch + 1, np.mean(loss_list[-e_interval:]), np.mean(accuracy_list[-e_interval:])))



import matplotlib.pyplot as plt
plt.plot(accuracy_list, c="red", label ="Tiseddi")
plt.xlabel("Epochs")
plt.ylabel("Azal")
plt.legend()
plt.show()

plt.plot(loss_list, c="blue", label ="Asṛuḥu")
plt.xlabel("Epochs")
plt.ylabel("Azal")
plt.legend()
plt.show()


with torch.no_grad():
    words = [torch.tensor(sequence_to_idx(s[0], char_to_idx), dtype=torch.long).to(device) for s in seq]
    sentence = torch.tensor(sequence_to_idx(seq, word_to_idx), dtype=torch.long).to(device)

    tag_scores = model(sentence, words)
    _, indices = torch.max(tag_scores, 1)
    ret = []
    for i in range(len(indices)):
        for key, value in tag_to_idx.items():
            if indices[i] == value:
                ret.append((seq[i], key))
    print(ret)
