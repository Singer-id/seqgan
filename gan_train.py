# -*- coding:utf-8 -*-

import os
import random
import math
import datetime
import argparse
import tqdm
import operator
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import sys
from importlib import reload
reload(sys)
from torch.autograd import Variable

from sql_type import Seq2SQL
from discriminator import Discriminator
from rollout import Rollout
from data_iter import DisDataIter
from utils_1 import *
from rouge import *
from embedding import WordEmbedding
# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
opt = parser.parse_args()
print(opt)

USE_SMALL = True
# GPU = False
# if opt.cuda is not None and opt.cuda >= 0:
#     torch.cuda.set_device(opt.cuda)
#     opt.cuda = True
if opt.cuda:
    GPU = True
else:
    GPU = False

# Basic Training Paramters
SEED = 88
BATCH_SIZE = 64
TOTAL_BATCH = 80
GENERATED_NUM = 1000
POSITIVE_FILE = 'real.data'
NEGATIVE_FILE = 'gene.data'
EVAL_FILE = 'eval.data'
VOCAB_SIZE = 5000
dataset = 0
pred_entry = (True, True, True)
# Genrator Parameters
N_word=300
N_h=10
N_depth = 2
B_word=42
max_col_num = 45
max_tok_num = 460
PRE_EPOCH_NUM = 10
SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND',
           'EQL', 'GT', 'LT', '<BEG>', 'None', 'max', 'min', 'count', 'sum', 'avg', 'SELECT']

sql_data, table_data, val_sql_data, val_table_data, \
test_sql_data, test_table_data, \
TRAIN_DB, DEV_DB, TEST_DB = load_dataset(
    dataset, use_small=USE_SMALL)

word_emb = load_word_emb('glove/glove.%dB.%dd.txt' % (B_word, N_word), \
                         load_used=False, use_small=USE_SMALL)

embed_layer = WordEmbedding(word_emb, N_word, GPU, SQL_TOK, our_model=False, trainable=False)

# Discriminator Parameters
d_emb_dim = 300
# d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
d_filter_sizes = [1, 2, 3]
d_num_filters = [100, 120, 160]
dis_pre_epoch = 1
d_dropout = 0.75
d_num_class = 2

# perm = np.random.permutation(len(sql_data))

def generate_real_samples(generated_num, output_file):
    samples = []
    # sql_data, table_data, val_sql_data, val_table_data ,test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB = load_dataset(dataset_id=0, use_small=False)
    for i in range(int(generated_num)):
        sql = sql_data[i]
        sample = sql['query_tok']
        sample.append('<END>')
        samples.append(sample)
    # print ("samples=",samples[0][0])
    with open(output_file, 'w',encoding= 'utf-8') as fout:
        for sample in samples:
            #print("current sample=",sample)
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

        q_seq, col_seq, col_num, cell, cell_num, cell_id, cell_tok, input_id, sql_query_seq, gt_seq, raw_data, type = to_batch_seq(sql_data, table_data, perm, st, ed)

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

        # generate add_type
        type_new = []
        for i in range(len(q_seq)):
            nnum = ['min', 'sum', 'avg', 'max', 'GT', 'LT']
            type_new_one = []
            for j in range(len(SQL_TOK)):
                if SQL_TOK[j] in nnum:
                    type_new_one.append('number')
                else:
                    type_new_one.append('none')
            type_new_one += type[i]
            type_new.append(type_new_one)

        type_new_embedding, type_new_len = embed_layer.gen_q_embedding(type_new)

        all_tokens = gen_all_tokens(col_seq, cell_tok)
        samples_list = model.sample(all_inp_var, all_name_len, all_len, q_embedding, q_len, type_embedding,
                                    type_len, type_new_embedding , type_new_len, all_tokens)
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


    # define ge_train


def genetator_train(model, optimizer, batch_size, sql_data, table_data, qz):
    model.train()
    perm = np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0
    while st < len(sql_data):
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, cell, cell_num, cell_id, cell_tok, input_id, sql_query_seq, gt_seq, raw_data, type = to_batch_seq( sql_data, table_data, perm, st, ed)

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

        # generate type seq
        type_seq = []
        for i in range(len(q_seq)):
            type_seq_one = []
            for j in range(len(SQL_TOK)):
                type_seq_one.append('SQL')
            for k in range(col_num[i]):
                type_seq_one.append('column')
            for p in range(cell_num[i]):
                # for p in range(all_len[i] - col_num[i] - len(SQL_TOK)):
                type_seq_one.append('cell')
            type_seq.append(type_seq_one)
        type_embedding, type_len = embed_layer.gen_q_embedding(type_seq)

        # generate add_type
        type_new = []
        for i in range(len(q_seq)):
            nnum = ['min', 'sum', 'avg', 'max', 'GT', 'LT']
            type_new_one = []
            for j in range(len(SQL_TOK)):
                if SQL_TOK[j] in nnum:
                    type_new_one.append('number')
                else:
                    type_new_one.append('none')
            type_new_one += type[i]
            type_new.append(type_new_one)

        type_new_embedding, type_new_len = embed_layer.gen_q_embedding(type_new)
        # print(type_len, type_new_len)
        # for i in range(len(q_len)):
        #     print(cell_num[i]+col_num[i], len(type[i]))
        score = model.forward(all_inp_var, all_name_len, all_len, q_embedding, q_len, type_embedding, type_len,
                              type_new_embedding, type_new_len, gt_seq=gt_tok, reinforce=False, gt_sel=None)
        loss = model.loss(score, gt_tok, qz)
        cum_loss += loss.data.cpu().item() * (ed - st)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        st = ed

    return cum_loss / len(sql_data)



#define dis_train
def dis_train(model, data_iter, criterion, optimizer):
    model.train()
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:#tqdm(
        #data_iter, mininterval=2, desc=' - Training', leave=False):
        data, length = embed_layer.str_list_to_batch(data)
        target = Variable(target)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        # print(pred.shape)
        # print(target.shape)
        loss = criterion(pred, target)
        total_loss += loss.data.item()
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    data_iter.reset()
    return math.exp(total_loss / total_words)


def epoch_acc(model, batch_size, sql_data, table_data, pred_entry):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    error = 0.0
    while st < len(sql_data):
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, cell, cell_num, cell_id, cell_tok, input_id, sql_query_seq, gt_seq, raw_data, type = to_batch_seq(
            sql_data, table_data, perm, st, ed)

        # append fake_cell
        gt_tok, gt_len = generate_gt_seq(col_num, cell_id, gt_seq, cell_tok, input_id)
        all_tokens = gen_all_tokens(col_seq, cell_tok)

        raw_q_seq = [x[0] for x in raw_data]  # question
        # raw_col_seq = [x[1] for x in raw_data]  # col headers
        raw_col_seq = []
        for col in col_seq:
            co = []
            for val in col:
                co.append(' '.join(val))
            raw_col_seq.append(co)
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)

        # sel_index = agg_index = con_index = -1
        for b in range(len(q_seq)):
            # print(all_tokens[b], cell_tok[b])
            gt_seq_one = gt_tok[b][1:]
            gt_sample = []
            for g in gt_seq_one:
                gt_sample.append(all_tokens[b][g])
                # print(gt_sample, sql_query_seq[b])

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

        # generate add_type
        type_new = []
        for i in range(len(q_seq)):
            nnum = ['min', 'sum', 'avg', 'max', 'GT', 'LT']
            type_new_one = []
            for j in range(len(SQL_TOK)):
                if SQL_TOK[j] in nnum:
                    type_new_one.append('number')
                else:
                    type_new_one.append('none')
            type_new_one += type[i]
            type_new.append(type_new_one)

        type_new_embedding, type_new_len = embed_layer.gen_q_embedding(type_new)

        score = model.forward(all_inp_var, all_name_len, all_len, q_embedding, q_len, type_embedding, type_len,
                              type_new_embedding, type_new_len,
                              gt_seq=None, reinforce=False, gt_sel=None)

        # zhijie jisuan

        _, choice = score.topk(1)
        choice = choice.squeeze(2)
        B = len(choice)
        for b in range(B):
            gt_seq_one = gt_tok[b][1:]
            choice_one = choice[b].cpu().numpy().tolist()
            for i in range(len(choice[b])):
                if choice[b][i] == 1:
                    choice_one = choice[b].cpu()[: i + 1].numpy().tolist()
                    break
            if not operator.eq(gt_seq_one, choice_one):
                error += 1

        pred_queries = model.gen_query(score, q_seq, col_seq, cell_tok, raw_q_seq, raw_col_seq, pred_entry)
        one_err, tot_err = model.check_acc(raw_data, pred_queries, query_gt, pred_entry)

        one_acc_num += (ed - st - one_err)
        tot_acc_num += (ed - st - tot_err)

        st = ed
    acc = len(sql_data) - error
    return tot_acc_num / len(sql_data), one_acc_num / len(sql_data), acc / len(sql_data)

class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, target, reward):
        reward = reward.contiguous().view(-1,1)
        loss = target.contiguous().view(-1,1)
        loss = loss * reward
        loss =  - torch.sum(loss)/BATCH_SIZE
        return loss


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Define Networks
    generator = Seq2SQL(word_emb, N_word=N_word, gpu=GPU,trainable_emb=False)
    discriminator = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout, GPU)
    if opt.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    # Generate real data

    print('Generating data ...')
    generate_real_samples(GENERATED_NUM, POSITIVE_FILE)


    # Pretrain Generator
    loss_list = []
    epoch_num = []
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.001, weight_decay=0)
    agg_m, sel_m, cond_m = best_model_name()
    # for i in range(PRE_EPOCH_NUM):
    for i in range(1):
        print('Epoch %d @ %s' % (i + 1, datetime.datetime.now()))
        epoch_num.append(i + 1)
        b = genetator_train(generator, optimizer,BATCH_SIZE, sql_data, table_data,0)
        print(' Loss = %s' % b)
        loss_list.append(b)

        # train_acc = epoch_acc(generator, BATCH_SIZE, sql_data, table_data, pred_entry)
        # print ' Train acc_qm: %s\n   word acc_qm: %s' % train_acc
        # val_acc = epoch_acc(generator, BATCH_SIZE, val_sql_data, val_table_data, pred_entry)
        # print ' Dev acc_qm: %s\n   word acc_qm: %s ' % (val_acc)
    #generator.load_state_dict(torch.load('generater.pkl'))

    # pretrain Dsicriminator

    dis_criterion = nn.NLLLoss(size_average=False)
    dis_optimizer = optim.Adam(discriminator.parameters())
    if opt.cuda:
        dis_criterion = dis_criterion.cuda()
    print('Pretrain Dsicriminator ...')
    # for epoch in range(1):
    for epoch in range(1):
        generate_samples(generator,BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE, sql_data, table_data)
        dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
        for _ in range(dis_pre_epoch):
            loss = dis_train(discriminator, dis_data_iter, dis_criterion, dis_optimizer)
            print('Epoch %d, loss: %f' % (epoch+1, loss))

    #discriminator.load_state_dict(torch.load('discriminator.pkl'))
    # Adversarial Training 
    rollout = Rollout(generator, 0.8)

    print('#####################################################')
    print('Start Adeversatial Training...\n')
    gen_gan_loss = GANLoss()

    gen_gan_optm = optim.Adam(generator.parameters())
    if opt.cuda:
        gen_gan_loss = gen_gan_loss.cuda()
    gen_criterion = nn.NLLLoss(size_average=False)
    if opt.cuda:
        gen_criterion = gen_criterion.cuda()
    dis_criterion = nn.NLLLoss(size_average=False)
    dis_optimizer = optim.Adam(discriminator.parameters())
    if opt.cuda:
        dis_criterion = dis_criterion.cuda()
    for total_batch in range(TOTAL_BATCH):
        ## Train the generator for one step
        for it in range(1):
            perm = np.random.permutation(len(sql_data))
            st = 1
            ed = st + BATCH_SIZE if st + BATCH_SIZE < len(perm) else len(perm)
            q_seq, col_seq, col_num, cell, cell_num, cell_id, cell_tok, input_id, sql_query_seq, gt_seq, raw_data,  type = to_batch_seq(
                sql_data, table_data, perm, st, ed)
            # gt_tok, gt_len = generate_gt_seq(col_num, cell_id, gt_seq, cell_tok, input_id)

            # embedding dividely
            sql_seq = []
            for val in SQL_TOK:
                sql_seq.append([val])
            all_seq = []
            for i in range(len(q_seq)):
                all_seq_one = sql_seq+col_seq[i]+cell[i]
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
                for p in range(all_len[i]-col_num[i]-len(SQL_TOK)):
                    type_seq_one.append('cell')
                type_seq.append(type_seq_one)

            #generate add_type
            type_new = []
            for i in range(len(q_seq)):
                nnum = ['min','sum','avg','max','GT','LT']
                type_new_one = []
                for j in range(len(SQL_TOK)):
                    if SQL_TOK[j] in nnum:
                        type_new_one.append('number')
                    else:
                        type_new_one.append('none')
                type_new_one += type[i]
                type_new.append(type_new_one)

            type_new_embedding , type_new_len = embed_layer.gen_q_embedding(type_new)

            type_embedding ,type_len = embed_layer.gen_q_embedding(type_seq)
            all_tokens = gen_all_tokens(col_seq, cell_tok)
            prob_1 = generator.forward(all_inp_var, all_name_len, all_len, q_embedding, q_len, type_embedding ,type_len, type_new_embedding , type_new_len, gt_seq = None, reinforce=False, gt_sel=None)

            prob = torch.nn.functional.softmax(prob_1,dim = 2)
            #print(prob_1.size(),prob.size())
            score, choices = prob.topk(1)
            # print(score)
            choices = choices.squeeze(2)
            choices = choices.view(-1,1)
            #print(choices)

            samples = torch.zeros(choices.size(0), max_tok_num).scatter_(1, choices, 1)
            samples = samples.view(BATCH_SIZE, -1, max_tok_num)
            # print(samples.shape)
            # print(max(x_len))

            rewards = rollout.get_reward(samples, 1, discriminator, all_inp_var, all_name_len, all_len, q_embedding, q_len, type_embedding ,type_len, type_new_embedding , type_new_len,col_seq, cell_tok, gt_seq, all_tokens, type_new, all_cell)
            rewards = Variable(torch.Tensor(rewards))
            #print(rewards.size())
            if opt.cuda:
                rewards = rewards.cuda().contiguous().view((-1,))
            #prob = generator.forward(input_var)
            #print(prob.size())
            loss = gen_gan_loss(score, rewards)
            gen_gan_optm.zero_grad()
            loss.backward()
            gen_gan_optm.step()

        # if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
        #     generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
        #     eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)
        #     loss = eval_epoch(target_lstm, eval_iter, gen_criterion)
        #     print('Batch [%d] True Loss: %f' % (total_batch, loss))
        rollout.update_params()
        
        # for _ in range(4):
        # # for _ in range(1):
        #     generate_samples(generator, GENERATED_NUM, NEGATIVE_FILE, sql_data, table_data)
        #     dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
        #     for _ in range(2):
        #     # for _ in range(1):
        #         loss = dis_train(discriminator, dis_data_iter, dis_criterion, dis_optimizer)
        #
        # train_acc = epoch_acc(generator, BATCH_SIZE, sql_data, table_data, pred_entry)
        # print ' Train acc_qm: %s\n   break down acc_qm: %s' % train_acc
        # val_acc = epoch_acc(generator, BATCH_SIZE, val_sql_data, val_table_data, pred_entry)
        # print ' Dev acc_qm: %s\n  break down acc_qm：%s' % val_acc
        # generator.train()



if __name__ == '__main__':
    main()
