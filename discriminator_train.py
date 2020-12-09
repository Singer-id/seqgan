# -*- coding:utf-8 -*-

import os
import random
import math

import argparse
import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from data_iter import DisDataIter
from discriminator import Discriminator
from utils import load_word_emb
from embedding import WordEmbedding

# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
parser.add_argument('--load', default=False , action='store_true', help='The suffix at the end of saved model name.')
parser.add_argument('--test', default=False , action='store_true', help='test.')
opt = parser.parse_args()
print(opt)


if opt.cuda:
    GPU = True
else:
    GPU = False

if opt.load:
    loading = True
else:
    loading = False


POSITIVE_FILE = 'true.data'
NEGATIVE_FILE = 'fake.data'
EVAL_FILE = 'eval.data'


BATCH_SIZE = 64
TOTAL_BATCH = 1

# Discriminator Parameters

VOCAB_SIZE = 5000
d_emb_dim = 300
# d_filter_sizes = [1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5]
# d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
d_filter_sizes = [1, 2, 3]
d_num_filters = [100, 120, 160]
d_dropout = 0.75
d_num_class = 2


N_word = 300
B_word = 42
USE_SMALL = True
GPU = False
SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND',
           'EQL', 'GT', 'LT', '<BEG>', 'None', 'max', 'min', 'count', 'sum', 'avg', 'SELECT']

word_emb = load_word_emb('glove/glove.%dB.%dd.txt' % (B_word, N_word), \
                         load_used=False, use_small=USE_SMALL)

cond_embed_layer = WordEmbedding(word_emb, N_word, GPU,
                                 SQL_TOK, our_model=False,
                                 trainable=False)


# define train
def train_epoch(model, data_iter, criterion, optimizer):
    total_loss = 0.
    total_words = 0.1
    count = 0
    for (data, target) in data_iter:#tqdm(
        #data_iter, mininterval=2, desc=' - Training', leave=False):
        data, length = cond_embed_layer.str_list_to_batch(data)
        target = Variable(target)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        for i in range(len(target)):
            count +=1
        pred = model.forward(data)
        loss = criterion(pred, target)
        total_loss += loss.data.item()
        # total_words += data.size(0) * data.size(1)
        total_words += data.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    data_iter.reset()
    # return math.exp(total_loss / total_words)
    return total_loss / total_words

def acc_epoch(model, data_iter ):
    error = 0.0
    nums = 0
    for (data, target) in data_iter:#tqdm(
        #data_iter, mininterval=2, desc=' - Training', leave=False):
        data, length = cond_embed_layer.str_list_to_batch(data)
        target = Variable(target)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        # print(len(pred))
        _,  choice = pred.topk(1);
        choices = choice.squeeze(1)
        for i in range(len(target)):
            if opt.test:
                print(target[i],choices[i])
            nums += 1
            if target[i]!=choices[i]:
                error += 1
    return (nums-error)/nums



#pretrain Dsicriminator

dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
if opt.test:
    POSITIVE_FILE_1 = 'real.data'
    NEGATIVE_FILE_1 = 'gene.data'
    dis_data_iter = DisDataIter(POSITIVE_FILE_1, NEGATIVE_FILE_1, BATCH_SIZE)

discriminator = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout, GPU)
# dis_criterion = nn.NLLLoss(size_average=False)
dis_criterion = nn.CrossEntropyLoss()
dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001, weight_decay=0)
if opt.cuda:
    dis_criterion = dis_criterion.cuda()
print('Pretrain Dsicriminator ...')


if loading:
    discriminator.load_state_dict(torch.load('discriminator.pkl'))
    acc = acc_epoch(discriminator, dis_data_iter)
    print ('acc: %f' %  acc)
else:
    best_acc = 0.0
    for epoch in range(5):
        # generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)
        dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
        #for _ in range(3):
        for _ in range(3):
            loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer)
            print('Epoch %d, loss: %f' % (epoch+1, loss))
        acc = acc_epoch(discriminator, dis_data_iter)
        if acc > best_acc:
            torch.save(discriminator.state_dict(), 'discriminator.pkl')
        print ('Epoch %d, acc: %f' % (epoch+1, acc))



