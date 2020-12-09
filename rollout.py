# -*- coding:utf-8 -*-

from utils import load_dataset, to_batch_seq, load_word_emb
from word_embedding import WordEmbedding
import copy
from grammar_score import *
from rouge import *
import numpy as np
from net_utils import run_lstm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
N_word = 300
B_word = 42
USE_SMALL = True
GPU = False
SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND',
           'EQL', 'GT', 'LT', '<BEG>', 'None', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG', 'SELECT']
word_emb = load_word_emb('glove/glove.%dB.%dd.txt' % (B_word, N_word), load_used=False, use_small=USE_SMALL)

cond_embed_layer = WordEmbedding(word_emb, N_word, GPU,
                                 SQL_TOK, our_model=False,
                                 trainable=False)
class Rollout(object):
    """Roll-out policy"""

    def __init__(self, model, update_rate):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate
        self.softmax = nn.Softmax()


    def get_reward(self, samples, num, discriminator, all_inp_var, all_name_len, all_len, q_embedding, q_len, type_embedding ,type_len, type_new_embedding , type_new_len, col_seq, cell_tok, gt_seq, all_tokens, all_type,all_cell):
        """
        Args:
            x : (batch_size, seq_len) input data
            num : roll-out number
            discriminator : discrimanator model
        """
        rewards = []
        max_len = samples.size(1)
        col = []
        cell = []
        for (one_col, one_cell) in zip(col_seq, cell_tok):
            one_co = []
            one_ce = []
            for val in one_col:
                one_co.append(' '.join(val))
            for vall in one_cell:
                one_ce.append(' '.join(vall))
            col.append(one_co)
            cell.append(one_ce)



        for i in range(num):
            for l in range(1, max_len+1):
                data = samples[:, 0:l, :]
                MCsamples = self.own_model.MCsample(all_inp_var, all_name_len, all_len, q_embedding, q_len, type_embedding ,type_len, type_new_embedding , type_new_len, all_tokens, data)
                # jisuan ROUGE and grammar:
                #rouge_score = []
                grammar_score = []
                for k in range(len(MCsamples)):
                    #rouge_score.append(rouge_2(gt_seq[i],MCsamples[i],1))
                    grammar_score.append(grammar(MCsamples[k], col[k], cell[k], all_tokens[k],all_type[k],all_cell[k]))
                    # print(MCsamples[k],grammar(MCsamples[k], col[k], cell[k]))
                # print grammar_score
                MCsamples, length = cond_embed_layer.str_list_to_batch(MCsamples)
                pred = self.softmax(discriminator(MCsamples))
                pred = pred.cpu().data[:, 1].numpy()
                for p in range(len(pred)):
                    pred[p] = 0.2*pred[p] + 0.8*grammar_score[p]

                if i == 0:
                    rewards.append(pred)

                else:
                    rewards[l - 1] += pred

            # for the last token
            # pred = discriminator(x)
            # pred = pred.cpu().data[:, 1].numpy()
            # if i == 0:
            #     rewards.append(pred)
            # else:
            #     rewards[x_len[i]-1] += pred
        rewards = np.transpose(np.array(rewards)) / (1.0 * num)  # batch_size * seq_len
        #print(rewards.shape)
        #print(max_len)
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
