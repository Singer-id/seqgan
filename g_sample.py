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


    def generate_real_samples(generated_num, output_file):
        samples = []
        # sql_data, table_data, val_sql_data, val_table_data ,test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB = load_dataset(dataset_id=0, use_small=False)
        for i in range(int(generated_num)):
            sql = sql_data[i]
            sample = sql['query_tok']
            sample.append('<END>')
            #sample.append(sql['question'])
            samples.append(sample)
        # print ("samples=",samples[0][0])
        with open(output_file, 'w') as fout:
            for sample in samples:
                # print("current sample=",sample)
                string = ' '.join([s for s in sample])
                fout.write('%s\n' % string)


    def generate_samples(model, batch_size, generated_num, output_file, sql_data, table_data):
        """利用模型自带sample函数，生成数据样本"""
        # samples = []
        perm = np.random.permutation(len(sql_data))
        st = 0
        all_samples = []
        while st < generated_num:
            ed = st + batch_size if st + batch_size < generated_num else generated_num

            q_seq, col_seq, col_num, cell, cell_num, gt_cell, cell_id, cell_tok, input_id, sql_query_seq, gt_seq, raw_data = to_batch_seq(
                sql_data, table_data, perm, st, ed)

            # embedding dividely
            sql_seq = []
            for val in SQL_TOK:
                sql_seq.append([val])
            all_seq = []
            for i in range(len(q_seq)):
                all_seq_one = sql_seq + col_seq[i] + cell[i]
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

            all_tokens = gen_all_tokens(col_seq, cell_tok)
            samples_list = model.sample(all_inp_var, all_name_len, all_len, q_embedding, q_len, type_embedding,
                                        type_len, all_tokens)
            st = ed
            samples = []
            for sample_list in samples_list:
                if '<END>' in sample_list:
                    sample_list = sample_list[: sample_list.index('<END>') + 1]
                samples.append(sample_list)
            for sample in samples:
                if len(sample) < 3:
                    samples.remove(sample)
            all_samples.append(samples)
        with open(output_file, 'w') as fout:
            for samples in all_samples:
                for sample in samples:
                    string = ' '.join([s for s in sample])
                    fout.write('%s\n' % string)


    generator = Seq2SQL(word_emb, N_word=N_word, gpu=GPU, trainable_emb=False)
    if loading:
        generator.load_state_dict(torch.load('generater.pkl'))
    GENERATED_NUM = 1000
    POSITIVE_FILE = 'real.data'
    NEGATIVE_FILE = 'gene.data'
    generate_real_samples(GENERATED_NUM, POSITIVE_FILE)
    generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE, sql_data, table_data)
