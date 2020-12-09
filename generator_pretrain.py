import json
import torch
from utils import *
from seq2sql import Seq2SQL
import numpy as np
import datetime
import matplotlib
from word_embedding import WordEmbedding
from torch.autograd import Variable
import torch.nn.functional as F
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# This is the training model - where there is no loading of model state.
# Training on CE losses first followed by RL
# best model is saved for test.
# saved model attached along with the code
# is generated from this model.
N_word=300
N_h=100
N_depth = 2
B_word=42
BATCH_SIZE=64
max_col_num = 45
max_tok_num = 250
use_small=True

SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND',
           'EQL', 'GT', 'LT', '<BEG>', '', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG', 'SELECT']

COND_OPS = ['EQL', 'GT', 'LT']
GPU = False

loss_list = []
epoch_num = []
#TRAIN_ENTRY=(True, True, True)  # (AGG, SEL, COND)
#reinforce = True


sql_data, table_data, val_sql_data, val_table_data, \
             test_sql_data, test_table_data, \
             TRAIN_DB, DEV_DB, TEST_DB = load_dataset(1,use_small)

word_emb = load_word_emb('glove/glove.%dB.%dd.txt' % (B_word, N_word), \
                         load_used=False, use_small=True)

cond_embed_layer = WordEmbedding(word_emb, N_word, GPU, SQL_TOK, our_model=False)

model = Seq2SQL(N_word, N_h, N_depth, max_col_num, max_tok_num, GPU)

optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay = 0)
#Adam default lr = 0.001, Can also try 0.01/0.05
agg_m, sel_m, cond_m = best_model_name()


# define train
def genetator_train(model, optimizer, batch_size, sql_data, table_data):
    model.train()
    perm=np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0
    #while st < len(sql_data):
    while st < batch_size:
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq = \
                to_batch_seq(sql_data, table_data, perm, st, ed)
        x_emb_var, x_len = cond_embed_layer.gen_x_batch(q_seq, col_seq)

        gt_seq = model.generate_gt_seq(q_seq, col_seq, query_seq)
        print(gt_seq)

        score, choice = model.forward(x_emb_var, x_len, gt_seq, gt_cond=None, reinforce=None, hidden=None)

        loss = model.loss(score, gt_seq)
        cum_loss += loss.data.cpu().item()*(ed - st)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        st = ed

    return cum_loss / batch_size




def epoch_acc(model, batch_size, sql_data, table_data):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0
    tot_err_num = 0.0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        x_emb_var, x_len = cond_embed_layer.gen_x_batch(q_seq, col_seq)

        gt_seq = model.generate_gt_seq(q_seq,col_seq,query_seq)
        pred_seq, choices = model.forward(x_emb_var, x_len, gt_seq, gt_cond=None, reinforce=None, hidden=None)
        for i in range(len(gt_seq)):
            gt_seq[i] = gt_seq[i][1:]

            for j in range(len(gt_seq[i])):
                if gt_seq[i][j] != choices[i][j]:
                    tot_err_num +=1
                    break
        st = ed
    tot_acc_num = len(sql_data) - tot_err_num
    return tot_acc_num / len(sql_data)


#init_acc = epoch_acc(model, BATCH_SIZE,val_sql_data, val_table_data)

#best_acc = init_acc
best_idx = 0
#print('Init dev acc_qm: %s\n  breakdown on (agg, sel, where): %s'%\ init_acc)
    
for i in range(100):
    print('Epoch %d @ %s'%(i+1, datetime.datetime.now()))
    epoch_num.append(i+1)
    b = genetator_train(model, optimizer, BATCH_SIZE, sql_data, table_data)
    print(' Loss = %s'%b)
    loss_list.append(b)

    train_acc = epoch_acc(model, BATCH_SIZE, sql_data, table_data)
    print(' Train acc_qm: %s ' % (train_acc))
    val_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data)
    print(' Dev acc_qm: %s ' % (val_acc))
    # print(' Train acc_qm: %s\n   breakdown result: %s'%epoch_acc(
    #                     model, BATCH_SIZE, sql_data, table_data))
    # val_acc = epoch_acc(model,
    #                     BATCH_SIZE, val_sql_data, val_table_data)
    # print(' Dev acc_qm: %s\n   breakdown result: %s'%val_acc)
    #
    # if val_acc > best_acc:
    #     best_acc = val_acc
    #     best_cond_idx = i+1
    #     torch.save(model.cond_pred.state_dict(), cond_m)
    #
    # print(' Best val acc = %s, on epoch %s individually'%(best_acc,best_cond_idx))

#bestince = best_acc
#plt.plot(epoch_num,loss_list, 'ro')
#plt.xlabel('Epoch_Num')
#plt.ylabel('Loss')
#plt.show()
#plt.savefig('lossdropmixed_curve.png')

# best_acc = 0.0
# best_idx = -1
# print("Init dev acc_qm: %s\n  breakdown on (agg, sel, where): %s"% \
#                 epoch_acc(model, BATCH_SIZE, val_sql_data,\
#                 val_table_data, TRAIN_ENTRY))
# print("Init dev acc_ex: %s"%epoch_exec_acc(
#                 model, BATCH_SIZE, val_sql_data, val_table_data, DEV_DB))
# reinforce = True
# for i in range(500):
#
#     print('Epoch in RL train %d @ %s'%(i+1, datetime.datetime.now()))
#
#     print(' Avg reward = %s'%epoch_reinforce_train(
#                 model, optimizer, BATCH_SIZE, sql_data, table_data, TRAIN_DB))
#     print(' Train acc_qm: %s\n   breakdown result: %s'%epoch_acc(
#                         model, BATCH_SIZE, sql_data, table_data, TRAIN_ENTRY))
#     print(' dev acc_qm: %s\n   breakdown result: %s'% epoch_acc(
#                 model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY))
#     exec_acc = epoch_exec_acc(
#                     model, BATCH_SIZE, val_sql_data, val_table_data, DEV_DB)
#     print(' dev acc_ex: %s'%exec_acc)
#     if exec_acc > best_acc:
#         best_acc = exec_acc
#         best_idx = i+1
#         torch.save(model.agg_pred.state_dict(), agg_m)
#         torch.save(model.sel_pred.state_dict(), sel_m)
#         torch.save(model.cond_pred.state_dict(), cond_m)
#     print(' Best exec acc = %s, on epoch %s'%(best_acc, best_idx))
