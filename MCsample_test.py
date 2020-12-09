# -*- coding:utf-8 -*-
import sys
from importlib import reload
reload(sys)
sys.setdefaultencoding('utf8')
import random
import torch
from utils import *
from dbengine import DBEngine
from SeqtoSQL import Seq2SQL
import numpy as np
import datetime
import matplotlib
from embedding import WordEmbedding
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import operator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #data
    #parser.add_argument('--condition', default=False, action='store_true',  help='If set, use conditionn data.')
    #gpu
    parser.add_argument('--gpu', default=False, action='store_true', help='Enable gpu')
    parser.add_argument('--load', default=False , action='store_true', help='The suffix at the end of saved model name.')
    args = parser.parse_args()
    dataset = 0
    N_word = 300
    N_h = 100
    N_depth = 2
    B_word = 42
    BATCH_SIZE = 64
    max_col_num = 45
    max_tok_num = 200
    USE_SMALL = True
    load_used = False
    SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>', 'None', 'max', 'min', 'count', 'sum',
               'avg', 'SELECT']

    if args.gpu:
        GPU = True
    else:
        GPU = False


    if args.load:
        loading = True
    else:
        loading = False


    sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, \
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset(
                    dataset, use_small=USE_SMALL)

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
            load_used=load_used, use_small=USE_SMALL)

    model = Seq2SQL(word_emb, N_word=N_word, gpu=GPU,trainable_emb=False)
    model.load_state_dict(torch.load('generater.pkl'))

    embed_layer = WordEmbedding(word_emb, N_word, GPU, SQL_TOK, our_model=False, trainable=False)

    loss_list = []
    epoch_num = []


    def genetator_train(model, batch_size, sql_data, table_data):
        perm = np.random.permutation(len(sql_data))
        cum_loss = 0.0
        st = 0
        while st < len(sql_data):
            ed = st + batch_size if st + batch_size < len(perm) else len(perm)

            q_seq, col_seq, col_num, cell, cell_num, gt_cell, cell_id, cell_tok, input_id, sql_query_seq, gt_seq, raw_data = to_batch_seq(
                sql_data, table_data, perm, st, ed)

            gt_tok, gt_len = generate_gt_seq(col_num, cell_id, gt_seq, cell_tok, input_id)

            # embedding dividely
            sql_seq = []
            for val in SQL_TOK:
                sql_seq.append([val])
            all_seq = []
            for i in range(len(q_seq)):
                if len(cell[i]) != 0:
                    all_seq_one = sql_seq + col_seq[i] + cell[i]
                else:
                    all_seq_one = sql_seq + col_seq[i]
                all_seq.append(all_seq_one)
            q_embedding, q_len = embed_layer.gen_q_embedding(q_seq)
            all_batch = embed_layer.gen_col_embedding(all_seq)
            all_inp_var, all_name_len, all_len = all_batch

            # generate typr seq
            type_seq = []
            for i in range(len(q_seq)):
                type_seq_one = []
                for j in range(len(SQL_TOK)):
                    type_seq_one.append('SQL')
                for k in range(col_num[i]):
                    type_seq_one.append('column')
                for p in range(all_len[i] - col_num[i] - len(SQL_TOK)):
                    type_seq_one.append('cell')
                type_seq.append(type_seq_one)

            type_embedding, type_len = embed_layer.gen_q_embedding(type_seq)

            score = model.forward(all_inp_var, all_name_len, all_len, q_embedding, q_len, type_embedding, type_len,
                                  gt_seq=None, reinforce=False, gt_sel=None)

            score, choices = score.topk(1)
            # print(score.shape)
            choices = choices.squeeze(2)
            choices = choices.view(-1, 1)
            # print(choices)

            samples = torch.zeros(choices.size(0), max_tok_num).scatter_(1, choices, 1)
            samples = samples.view(BATCH_SIZE, -1, max_tok_num)

            all_tokens = gen_all_tokens(col_seq, cell_tok)

            MCsamples = model.MCsample(all_inp_var, all_name_len, all_len, q_embedding, q_len, type_embedding,
                                       type_len, all_tokens, samples)
            for b in range(len(q_seq)):
                print(MCsamples[b])
        return MCsamples


    genetator_train(model, BATCH_SIZE, sql_data, table_data)
