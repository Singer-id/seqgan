import json
import torch
from utils import *
from model import Seq2SQL
# from condition_prediction import Seq2SQLCondPredictor
import numpy as np
import datetime
import matplotlib
from word_embedding import WordEmbedding
from torch.autograd import Variable
import torch.nn.functional as F
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #data
    #parser.add_argument('--condition', default=False, action='store_true',  help='If set, use conditionn data.')
    #gpu
    parser.add_argument('--gpu', default=False, action='store_true', help='Enable gpu')

    args = parser.parse_args()
    dataset = 1
    N_word = 300
    N_h = 100
    N_depth = 2
    B_word = 42
    BATCH_SIZE = 64
    max_col_num = 45
    max_tok_num = 200
    USE_SMALL = True
    load_used = False

    if args.gpu:
        GPU = True
    else:
        GPU = False


    sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, \
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset(
                    dataset, use_small=USE_SMALL)

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
            load_used=load_used, use_small=USE_SMALL)
    model = Seq2SQL(word_emb, N_word=N_word, gpu=GPU,trainable_emb=False)

    # COND_OPS = ['EQL', 'GT', 'LT']


    loss_list = []
    epoch_num = []



    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    # Adam default lr = 0.001, Can also try 0.01/0.05
    agg_m, sel_m, cond_m = best_model_name()




    def genetator_train(model, optimizer, batch_size, sql_data, table_data):
        model.train()
        perm = np.random.permutation(len(sql_data))
        cum_loss = 0.0
        st = 0
        while st < len(sql_data):
            ed = st + batch_size if st + batch_size < len(perm) else len(perm)

            # q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq = \
            #     to_batch_seq(sql_data, table_data, perm, st, ed)
            q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, cell_seq, cell_num = \
                to_batch_seq(sql_data, table_data, perm, st, ed)
            gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
            gt_sel_seq = [x[1] for x in ans_seq]
            score = model.forward(q_seq, col_seq, col_num, cell_seq, cell_num, gt_where=gt_where_seq, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq)
            loss = model.loss(score, gt_where_seq)
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
        while st < len(sql_data):
            ed = st + batch_size if st + batch_size < len(perm) else len(perm)

            q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = to_batch_seq(sql_data, table_data,
                                                                                              perm, st, ed,
                                                                                              ret_vis_data=True)
            raw_q_seq = [x[0] for x in raw_data]  # question
            raw_col_seq = [x[1] for x in raw_data]  # col headers
            query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
            gt_sel_seq = [x[1] for x in ans_seq]
            score = model.forward(q_seq, col_seq, col_num, gt_where=None, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq)
            pred_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq, raw_col_seq, pred_entry)
            one_err, tot_err = model.check_acc(raw_data, pred_queries, query_gt, pred_entry)

            one_acc_num += (ed - st - one_err)
            tot_acc_num += (ed - st - tot_err)

            st = ed
        return tot_acc_num / len(sql_data), one_acc_num / len(sql_data)

    # def epoch_acc(model, batch_size, sql_data, table_data, pred_entry):
    #
    #     model.eval()
    #     perm = list(range(len(sql_data)))
    #     st = 0
    #
    #     error = 0
    #     while st < len(sql_data):
    #         ed = st + batch_size if st + batch_size < len(perm) else len(perm)
    #
    #         q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = to_batch_seq(sql_data, table_data,
    #                                                                                           perm, st, ed,
    #                                                                                           ret_vis_data=True)
    #
    #         gt_sel_seq = [x[1] for x in ans_seq]
    #         gt_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
    #         score = model.forward(q_seq, col_seq, col_num, gt_where=None, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq)
    #         _, choice = score.topk(1)
    #         choice = choice.squeeze(2)
    #         B = len(gt_seq)
    #         for b in range(B):
    #             if gt_seq[b][1:]!=choice[b][: len(gt_seq[b])-1]:
    #                 error +=1
    #
    #         st = ed
    #     acc =  len(sql_data) - error
    #     return acc / len(sql_data)

    TRAIN_ENTRY = (True, True, True)
    for i in range(100):
        print ('Epoch %d @ %s' % (i + 1, datetime.datetime.now()))
        b = genetator_train(model, optimizer, BATCH_SIZE, sql_data, table_data)
        print(' Loss = %s' % b)
        loss_list.append(b)

        print (' Train acc_qm: %s\n   breakdown result: %s' % epoch_acc(model, BATCH_SIZE, sql_data, table_data, TRAIN_ENTRY))
        # val_acc = epoch_token_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY)
        val_acc = epoch_acc(model,BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY)
        print (' Dev acc_qm: %s\n   breakdown result: %s' % val_acc)





