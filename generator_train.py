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

    embed_layer = WordEmbedding(word_emb, N_word, GPU, SQL_TOK, our_model=False, trainable=False)


    loss_list = []
    epoch_num = []



    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    # Adam default lr = 0.001, Can also try 0.01/0.05
    agg_m, sel_m, cond_m = best_model_name()



    def genetator_train(model, optimizer, batch_size, sql_data, table_data):
        model.train()
        perm = np.random.permutation(len(sql_data))
        cum_loss = 0.0
        st = 0
        while st < len(sql_data):
            ed = st + batch_size if st + batch_size < len(perm) else len(perm)

            q_seq, col_seq, col_num, cell, cell_num, gt_cell, cell_id, cell_tok, input_id, sql_query_seq, gt_seq, raw_data = to_batch_seq(sql_data, table_data, perm, st, ed)

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
                for p in range(all_len[i]-col_num[i]-len(SQL_TOK)):
                    type_seq_one.append('cell')
                type_seq.append(type_seq_one)

            type_embedding ,type_len = embed_layer.gen_q_embedding(type_seq)

            score = model.forward(all_inp_var, all_name_len, all_len, q_embedding, q_len, type_embedding ,type_len, gt_seq = gt_tok, reinforce=False, gt_sel=None)
            loss = model.loss(score, gt_tok)
            cum_loss += loss.data.cpu().item() * (ed - st)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            st = ed

        return cum_loss / len(sql_data)


    def epoch_acc(model, batch_size, sql_data, table_data, pred_entry):
        model.eval()
        perm = list(range(len(sql_data)))
        st = 0
        one_acc_num = 0.0
        tot_acc_num = 0.0
        error = 0.0
        while st < len(sql_data):
            ed = st + batch_size if st + batch_size < len(perm) else len(perm)

            q_seq, col_seq, col_num, cell, cell_num, gt_cell, cell_id,  cell_tok, input_id, sql_query_seq, gt_seq, raw_data = to_batch_seq(sql_data, table_data, perm, st, ed)


            # append fake_cell
            gt_tok, gt_len = generate_gt_seq(col_num, cell_id, gt_seq, cell_tok, input_id)
            all_tokens = gen_all_tokens(col_seq, cell_tok)


            raw_q_seq = [x[0] for x in raw_data]  # question
            raw_col_seq = [x[1] for x in raw_data]  # col headers
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
                for p in range(all_len[i]-col_num[i]-len(SQL_TOK)):
                    type_seq_one.append('cell')
                type_seq.append(type_seq_one)

            type_embedding ,type_len = embed_layer.gen_q_embedding(type_seq)

            score = model.forward(all_inp_var, all_name_len, all_len, q_embedding, q_len, type_embedding, type_len,
                                  gt_seq=None, reinforce=False, gt_sel=None)

            #zhijie jisuan

            _, choice = score.topk(1)
            choice = choice.squeeze(2)
            B = len(choice)
            for b in range(B):
                gt_seq_one = gt_tok[b][1:]
                choice_one = choice[b].cpu().numpy().tolist()
                for i in range(len(choice[b])):
                    if choice[b][i] == 1:
                        choice_one = choice[b].cpu()[: i+1].numpy().tolist()
                        break
                if not operator.eq(gt_seq_one,choice_one):
                    error +=1


            pred_queries = model.gen_query(score, q_seq, col_seq, cell_tok, raw_q_seq, raw_col_seq, pred_entry)
            one_err, tot_err = model.check_acc(raw_data, pred_queries, query_gt, pred_entry)

            one_acc_num += (ed - st - one_err)
            tot_acc_num += (ed - st - tot_err)

            st = ed
        acc = len(sql_data) - error
        return tot_acc_num / len(sql_data), one_acc_num / len(sql_data), acc/len(sql_data)


    def epoch_exec_acc(model, batch_size, sql_data, table_data, db_path):
        engine = DBEngine(db_path)

        model.eval()
        perm = list(range(len(sql_data)))
        tot_acc_num = 0.0
        acc_of_log = 0.0
        st = 0
        while st < len(sql_data):
            ed = st + batch_size if st + batch_size < len(perm) else len(perm)
            q_seq, col_seq, col_num, cell, cell_num, gt_cell, cell_id, cell_tok, input_id, sql_query_seq, gt_seq, raw_data = to_batch_seq(
                sql_data, table_data, perm, st, ed)

            raw_q_seq = [x[0] for x in raw_data]  # question
            raw_col_seq = [x[1] for x in raw_data]  # col headers
            query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)

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

            score = model.forward(all_inp_var, all_name_len, all_len, q_embedding, q_len, type_embedding, type_len, gt_seq=None, reinforce=False, gt_sel=None)

            pred_queries = model.gen_query(score, q_seq, col_seq, cell_tok, raw_q_seq, raw_col_seq, (True, True, True))

            for idx, (sql_gt, sql_pred, tid) in enumerate(
                    zip(query_gt, pred_queries, table_ids)):
                ret_gt = engine.execute(tid,
                                        sql_gt['sel'], sql_gt['agg'], sql_gt['conds'])
                try:
                    ret_pred = engine.execute(tid,
                                              sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
                except:
                    ret_pred = None
                tot_acc_num += (ret_gt == ret_pred)

            st = ed

        return tot_acc_num / len(sql_data)


    # def generate_samples(model, generated_num, output_file, sql_data, table_data):
    #     """利用模型自带sample函数，生成数据样本"""
    #     # samples = []
    #     perm = np.random.permutation(len(sql_data))
    #     st = 0
    #     while st < generated_num:
    #         ed =  generated_num
    #
    #         q_seq, col_seq, col_num, cell, cell_num, cell_num_1, cell_id, cell_tok, sql_query_seq, gt_seq = to_batch_seq(sql_data,
    #                                                                                                          table_data,
    #                                                                                                          perm, st,
    #                                                                                                          ed)
    #         x_emb_var, x_len = embed_layer.gen_x_batch(q_seq, col_seq)
    #         col_batch = embed_layer.gen_col_embedding(col_seq)
    #         col_inp_var, col_name_len, col_len = col_batch
    #         cell_batch = embed_layer.gen_col_embedding(cell)
    #         cell_inp_var, cell_name_len, cell_len = cell_batch
    #         q_embedding, q_len = embed_layer.gen_q_embedding(q_seq)
    #         sql_embedding, sql_len = embed_layer.gen_sql_embedding(len(x_len))
    #
    #         all_tokens = gen_all_tokens(col_seq, cell_tok)
    #         samples_list = model.sample(col_inp_var, col_name_len, col_len, cell_inp_var, cell_name_len, cell_len,
    #                                q_embedding, q_len, sql_embedding, sql_len, all_tokens)
    #         st = ed
    #     samples = []
    #     for sample_list in samples_list:
    #         if '<END>' in sample_list:
    #             sample_list = sample_list[: sample_list.index('<END>')+1]
    #         samples.append(sample_list)
    #         for sample in samples:
    #             if (len(sample) < 3):
    #                 samples.remove(sample)
    #     # print ("samples=",samples[0][0])
    #     with open(output_file, 'w') as fout:
    #         for sample in samples:
    #             # print("current sample=",sample)
    #             string = ' '.join([s for s in sample])
    #             fout.write('%s\n' % string)

    def random_fake_data(generated_num, output_file, sql_data, table_data):
        perm = np.random.permutation(len(sql_data))
        q_seq, col_seq, col_num, cell, cell_num, cell_num_1, cell_id, cell_tok, sql_query_seq, gt_seq = to_batch_seq(
                    sql_data,
                    table_data,
                    perm, 0, generated_num)
        all_tokens = gen_all_tokens(col_seq, cell_tok)
        all_len = []
        for b in all_tokens:
            all_len.append(len(b))
        min_len = min(all_len)
        fake_samples = []
        for i in range(generated_num):
            fake_sample = []
            random_choice = random.sample(range(min_len), 5)

            for j in range(5):
                fake_sample.append(all_tokens[i][random_choice[j]])
            fake_samples.append(fake_sample)

        with open(output_file, 'w') as fout:
            for sample in fake_samples:
                # print("current sample=",sample)
                string = ' '.join([s for s in sample])
                fout.write('%s\n' % string)


    TRAIN_ENTRY = (True, True, True)

    if loading:
        model.load_state_dict(torch.load('generater.pkl'))


        val_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY)
        print (' Dev acc_qm: %s\n   breakdown result: %s\n  acc: %s' % val_acc)

    else:
        best_acc = 0.0
        for i in range(80):
            print ('Epoch %d @ %s' % (i + 1, datetime.datetime.now()))
            b = genetator_train(model, optimizer, BATCH_SIZE, sql_data, table_data)
            print(' Loss = %s' % b)
            loss_list.append(b)

            print (' Train acc_qm: %s\n   breakdown result: %s\n  acc: %s'  % epoch_acc(model, BATCH_SIZE, sql_data, table_data,TRAIN_ENTRY))
            val_acc = epoch_acc(model,BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY)
            print (' Dev acc_qm: %s\n   breakdown result: %s\n  acc: %s' % val_acc)
            if val_acc[0] > best_acc:
                best_acc = val_acc[0]
                torch.save(model.state_dict(), 'generater.pkl')
            # exec_acc = epoch_exec_acc(model, BATCH_SIZE, val_sql_data, val_table_data, DEV_DB)
            # print ' dev acc_ex: %s', exec_acc
            # if exec_acc[0] > best_acc:
            #     best_acc = val_acc[0]
            #     torch.save(model.state_dict(), 'generater.pkl')



    #
    # output_file = 'test_random_fake.data'
    # generated_num = 1000
    # random_fake_data(generated_num, output_file, sql_data, table_data)
    # NEGATIVE_FILE = 'test_gene.data'
    # generate_samples(model, 1000, NEGATIVE_FILE, sql_data, table_data)








