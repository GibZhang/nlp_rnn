#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @Time    : 2020-04-15 17:09
# @Author  : zhangjingbo
# @mail    : jingbo.zhang@yooli.com
# @File    : imdb_lstm.py
# @Description: 从keras加载imdb的数据，并用pytorch模型进行训练
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
import numpy as np

NUMBER_WORDS = 10000
MAX_LEN = 200
BATCH_SIZE = 256
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 128
DROPOUT = 0.3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path='./imdb.npz', num_words=NUMBER_WORDS)
    x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding='post', truncating='post')
    x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding='post', truncating='post')
    train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
    test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))
    train_sample = RandomSampler(train_data)
    train_loader = DataLoader(train_data, BATCH_SIZE, sampler=train_sample)
    test_sample = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, BATCH_SIZE, sampler=test_sample)
    return train_loader, test_loader


class ImdbBinary(nn.Module):
    def __init__(self, voc_size, embedding_size, hidden_size, dropout):
        super(ImdbBinary, self).__init__()
        self.embedding = nn.Embedding(voc_size, embedding_size)
        self.bilstm = nn.LSTM(embedding_size, hidden_size, num_layers=2, bias=True, batch_first=True,
                              bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        out, (hid, cn) = self.bilstm(embedded)
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = F.avg_pool2d(out, (out.shape[1], 1)).squeeze()
        return out


def train():
    n_iterators = 10
    model = ImdbBinary(NUMBER_WORDS, EMBEDDING_SIZE, HIDDEN_SIZE, DROPOUT)
    optimizer = optim.Adam(model.parameters())
    train_loader, test_loader = load_data()
    criterition = nn.CrossEntropyLoss()
    for i in range(n_iterators):
        for batch_index, (train_x, train_y) in enumerate(train_loader):
            optimizer.zero_grad()
            train_x, train_y = train_x.to(device), train_y.to(device)
            out = model(train_x)
            loss = criterition(out, train_y)
            loss.backward()
            optimizer.step()
            if batch_index % 10 == 1:
                print('epoch {} , batch_index {}, loss {}'.format(i, batch_index, loss.item()))
        correct = 0
        total = 0
        for batch_index, (test_x, test_y) in enumerate(test_loader):
            with torch.no_grad():
                test_x, test_y = test_x.to(device), test_y.to(device)
                pred = model(test_x)
                _, y_pred = torch.max(pred, 1)
                total += test_y.shape[0]
                correct += (y_pred == test_y).sum().item()
        print('test acc {}'.format(correct / total))


if __name__ == '__main__':
    train()
