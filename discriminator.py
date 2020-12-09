# -*- coding: utf-8 -*-

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from word_embedding import WordEmbedding
from torch.autograd import Variable


class Discriminator(nn.Module):
    """A CNN for text classification

    architecture: Embedding >> Convolution >> Max-pooling >> Softmax
    """

    def __init__(self, num_classes, vocab_size, emb_dim, filter_sizes, num_filters, dropout, gpu):
        super(Discriminator, self).__init__()
        #self.emb = nn.Embedding(vocab_size, emb_dim)
        self.gpu = gpu
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f,emb_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(sum(num_filters), num_classes)
        self.softmax = nn.Softmax()
        self.init_parameters()
        if gpu:
            self.cuda()

    def forward(self, emb):
        """
        Args:
            x: (batch_size * seq_len)
        """
        # word_emb = load_word_emb('glove/glove.%dB.%dd.txt' % (B_word, N_word), \
        #                          load_used=self.train_embed, use_small=USE_SMALL)
        # cond_embed_layer = WordEmbedding(word_emb, N_word, GPU,
        #                                  SQL_TOK, our_model=False,
        #                                  trainable=False)
        if self.gpu:
            emb = emb.cuda()
        emb = emb.unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim
        #emb = WordEmbedding.str_list_to_batch(x)

        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        highway = self.highway(pred)
        pred = F.sigmoid(highway) *  F.relu(highway) + (1. - F.sigmoid(highway)) * pred
        # pred = F.sigmoid(highway)
        score = self.lin(self.dropout(pred))

        return score

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)


