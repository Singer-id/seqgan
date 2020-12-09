# -*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import json
import random
from utils_right import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from embedding import WordEmbedding
from net_utils import run_lstm, col_name_encode



# This is a re-implementation based on the following paper:

# Victor Zhong, Caiming Xiong, and Richard Socher. 2017.
# Seq2SQL: Generating Structured Queries from Natural Language using
# Reinforcement Learning. arXiv:1709.00103

class Seq2SQL(nn.Module):
    def __init__(self, word_emb, N_word, N_h=100, N_depth=2,gpu=False, trainable_emb=False):
        super(Seq2SQL, self).__init__()
        self.trainable_emb = trainable_emb

        self.gpu = gpu
        self.N_h = N_h
        self.N_depth = N_depth

        self.max_col_num = 45
        self.max_tok_num = 200
        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>', 'None', 'max', 'min', 'count', 'sum', 'avg', 'SELECT']
        self.COND_OPS = ['EQL', 'GT', 'LT']

        self.cond_lstm = nn.LSTM(input_size=3*N_h, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)


        self.encoder = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.encoder_1 = nn.LSTM(input_size=2*N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.att = nn.Linear(N_h, N_h)
        self.att_type = nn.Linear(N_h, 2 * N_h)
        self.decoder = nn.LSTM(input_size=self.max_tok_num,
                hidden_size=N_h, num_layers=N_depth,
                batch_first=True, dropout=0.3)

        self.cond_out_g = nn.Linear(N_h, N_h)
        self.cond_out_h = nn.Linear(N_h, N_h)
        self.cond_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.embed_layer = WordEmbedding(word_emb, N_word, gpu,self.SQL_TOK, our_model=False,trainable=trainable_emb)
        # self.cond_pred = Seq2SQLCondPredictor(
        #     N_word, N_h, N_depth, self.max_col_num, self.max_tok_num, gpu)


        self.CE = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()
        self.bce_logit = nn.BCEWithLogitsLoss()
        if gpu:
            self.cuda()


    def gen_gt_batch(self, tok_seq, gen_inp=True):
        # If gen_inp: generate the input token sequence (removing <END>)
        # Otherwise: generate the output token sequence (removing <BEG>)
        B = len(tok_seq)
        ret_len = np.array([len(one_tok_seq)-1 for one_tok_seq in tok_seq])
        max_len = max(ret_len)
        ret_array = np.zeros((B, max_len, self.max_tok_num), dtype=np.float32)
        for b, one_tok_seq in enumerate(tok_seq):
            out_one_tok_seq = one_tok_seq[:-1] if gen_inp else one_tok_seq[1:]
            for t, tok_id in enumerate(out_one_tok_seq):
                ret_array[b, t, tok_id] = 1

        ret_inp = torch.from_numpy(ret_array)
        if self.gpu:
            ret_inp = ret_inp.cuda()
        ret_inp_var = Variable(ret_inp) #[B, max_len, max_tok_num]

        return ret_inp_var, ret_len


    # def forward(self, col_inp_var, col_name_len, col_len, cell_inp_var, cell_name_len, cell_len, q_embedding, q_len, sql_embedding, sql_len, gt_seq = None, reinforce=False, gt_sel=None):
    #
    #     B = len(q_len)
    #     # embedding dividely
    #     # x_emb_var, x_len = self.embed_layer.gen_x_batch(q, col)
    #     # col_batch = self.embed_layer.gen_col_embedding(col)
    #     # col_inp_var, col_name_len, col_len = col_batch
    #     # cell_batch = self.embed_layer.gen_col_embedding(cell)
    #     # cell_inp_var, cell_name_len, cell_len = cell_batch
    #     #
    #     # q_embedding, q_len = self.embed_layer.gen_q_embedding(q)
    #     # sql_embedding, sql_len = self.embed_layer.gen_sql_embedding(len(x_len))
    #     all_len = []
    #     for i in range(B):
    #         sum = sql_len[i]+col_len[i]+cell_len[i]
    #         all_len.append(sum)
    #     max_x_len = max(all_len)
    #     # encode dividely
    #     e_col, h_col = col_name_encode(col_inp_var, col_name_len,col_len, self.encoder)
    #     e_cell, h_cell = col_name_encode(cell_inp_var, cell_name_len, cell_len, self.encoder)
    #     e_q, h_q = run_lstm(self.encoder, q_embedding, q_len)
    #     e_sql, h_sql = run_lstm(self.encoder, sql_embedding, sql_len)
    #
    #     # attention between q and col:
    #     att_val = torch.bmm(e_col, self.att(e_q).transpose(1, 2))
    #     max_q_len = max(q_len)
    #     for idx, num in enumerate(q_len):
    #         if num < max_q_len:
    #             att_val[idx, :, num:] = -100
    #     att = self.softmax(att_val.view((-1, max_q_len))).view(B, -1, max_q_len)
    #     e_col_expand = (e_q.unsqueeze(1) * att.unsqueeze(3)).sum(2)
    #
    #     # attention between q and cell:
    #     att_val = torch.bmm(e_cell, self.att(e_q).transpose(1, 2))
    #     max_q_len = max(q_len)
    #     for idx, num in enumerate(q_len):
    #         if num < max_q_len:
    #             att_val[idx, :, num:] = -100
    #     att = self.softmax(att_val.view((-1, max_q_len))).view(B, -1, max_q_len)
    #     e_cell_expand = (e_q.unsqueeze(1) * att.unsqueeze(3)).sum(2)
    #
    #     #cat
    #     e_all = torch.cat((e_sql, e_col_expand, e_cell_expand), 1)
    #     h_enc, hidden = run_lstm(self.cond_lstm, e_all, all_len)
    #     decoder_hidden = tuple(torch.cat((hid[:2], hid[2:]),dim=2) for hid in hidden)
    #     if gt_seq is not None:
    #         gt_tok_seq, gt_tok_len = self.gen_gt_batch(gt_seq, gen_inp=True)
    #         g_s, _ = run_lstm(self.decoder,
    #                 gt_tok_seq, gt_tok_len, decoder_hidden)
    #
    #         h_enc_expand = h_enc.unsqueeze(1)
    #         g_s_expand = g_s.unsqueeze(2)
    #         cond_score = self.cond_out( self.cond_out_h(h_enc_expand ) +
    #                 self.cond_out_g(g_s_expand) ).squeeze()
    #         for idx, num in enumerate(all_len):
    #             if num < max_x_len:
    #                 cond_score[idx, :, num:] = -100
    #     else:
    #         h_enc_expand = h_enc.unsqueeze(1)
    #         scores = []
    #         choices = []
    #         done_set = set()
    #
    #         t = 0
    #         init_inp = np.zeros((B, 1, self.max_tok_num), dtype=np.float32)
    #         init_inp[:, 0, 7] = 1  #Set the <BEG> token
    #         if self.gpu:
    #             cur_inp = Variable(torch.from_numpy(init_inp).cuda())
    #         else:
    #             cur_inp = Variable(torch.from_numpy(init_inp))
    #         cur_h = decoder_hidden
    #         while len(done_set) < B and t < 100:
    #             g_s, cur_h = self.decoder(cur_inp, cur_h)
    #             g_s_expand = g_s.unsqueeze(2)
    #
    #             cur_cond_score = self.cond_out(self.cond_out_h(h_enc_expand) +
    #                     self.cond_out_g(g_s_expand)).squeeze()
    #             for b, num in enumerate(all_len):
    #                 if num < max_x_len:
    #                     cur_cond_score[b, num:] = -100
    #             scores.append(cur_cond_score)
    #
    #             if not reinforce:
    #                 _, ans_tok_var = cur_cond_score.view(B, max_x_len).max(1)
    #                 ans_tok_var = ans_tok_var.unsqueeze(1)
    #             else:
    #                 ans_tok_var = self.softmax(cur_cond_score).multinomial()
    #                 choices.append(ans_tok_var)
    #             ans_tok = ans_tok_var.data.cpu()
    #             if self.gpu:  #To one-hot
    #                 cur_inp = Variable(torch.zeros(
    #                     B, self.max_tok_num).scatter_(1, ans_tok, 1).cuda())
    #             else:
    #                 cur_inp = Variable(torch.zeros(
    #                     B, self.max_tok_num).scatter_(1, ans_tok, 1))
    #             cur_inp = cur_inp.unsqueeze(1)
    #
    #             for idx, tok in enumerate(ans_tok.squeeze()):
    #                 if tok == 1:  #Find the <END> token
    #                     done_set.add(idx)
    #             t += 1
    #
    #         cond_score = torch.stack(scores, 1)
    #
    #     if reinforce:
    #         return cond_score, choices
    #     else:
    #         return cond_score


    def forward(self, all_inp_var, all_name_len, all_len, q_embedding, q_len, type_embedding ,type_len, gt_seq = None, reinforce=False, gt_sel=None):

        B = len(q_len)
        # embedding dividely

        max_x_len = max(all_len)
        # encode dividely
        e_q, h_q = run_lstm(self.encoder, q_embedding, q_len)
        e_type, h_type = run_lstm(self.encoder, type_embedding ,type_len)
        e_sql, h_sql = col_name_encode(all_inp_var, all_name_len, all_len, self.encoder)


        #e_all_type = torch.cat((e_type, e_sql), 2)


        # attention between q and all:

        att_val = torch.bmm(e_sql, self.att(e_q).transpose(1, 2))
        max_q_len = max(q_len)
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val[idx, :, num:] = -100
        att = self.softmax(att_val.view((-1, max_q_len))).view(B, -1, max_q_len)
        e_all_tt = (e_q.unsqueeze(1) * att.unsqueeze(3)).sum(2)
        #cat
        e_all_q = torch.cat((e_sql,e_all_tt),2)

        #cat typr
        e_all_expand = torch.cat((e_all_q, e_type), 2)

        h_enc, hidden = run_lstm(self.cond_lstm, e_all_expand, all_len)
        decoder_hidden = tuple(torch.cat((hid[:2], hid[2:]),dim=2) for hid in hidden)
        if gt_seq is not None:
            gt_tok_seq, gt_tok_len = self.gen_gt_batch(gt_seq, gen_inp=True)
            g_s, _ = run_lstm(self.decoder,
                    gt_tok_seq, gt_tok_len, decoder_hidden)

            h_enc_expand = h_enc.unsqueeze(1)
            g_s_expand = g_s.unsqueeze(2)
            cond_score = self.cond_out( self.cond_out_h(h_enc_expand ) +
                    self.cond_out_g(g_s_expand) ).squeeze()
            for idx, num in enumerate(all_len):
                if num < max_x_len:
                    cond_score[idx, :, num:] = -100
        else:
            h_enc_expand = h_enc.unsqueeze(1)
            scores = []
            choices = []
            done_set = set()

            t = 0
            init_inp = np.zeros((B, 1, self.max_tok_num), dtype=np.float32)
            init_inp[:, 0, 7] = 1  #Set the <BEG> token
            if self.gpu:
                cur_inp = Variable(torch.from_numpy(init_inp).cuda())
            else:
                cur_inp = Variable(torch.from_numpy(init_inp))
            cur_h = decoder_hidden
            while len(done_set) < B and t < 100:
                g_s, cur_h = self.decoder(cur_inp, cur_h)
                g_s_expand = g_s.unsqueeze(2)

                cur_cond_score = self.cond_out(self.cond_out_h(h_enc_expand) +
                        self.cond_out_g(g_s_expand)).squeeze()
                for b, num in enumerate(all_len):
                    if num < max_x_len:
                        cur_cond_score[b, num:] = -100
                scores.append(cur_cond_score)

                if not reinforce:
                    _, ans_tok_var = cur_cond_score.view(B, max_x_len).max(1)
                    ans_tok_var = ans_tok_var.unsqueeze(1)
                else:
                    ans_tok_var = self.softmax(cur_cond_score).multinomial()
                    choices.append(ans_tok_var)
                ans_tok = ans_tok_var.data.cpu()
                if self.gpu:  #To one-hot
                    cur_inp = Variable(torch.zeros(
                        B, self.max_tok_num).scatter_(1, ans_tok, 1).cuda())
                else:
                    cur_inp = Variable(torch.zeros(
                        B, self.max_tok_num).scatter_(1, ans_tok, 1))
                cur_inp = cur_inp.unsqueeze(1)

                for idx, tok in enumerate(ans_tok.squeeze()):
                    if tok == 1:  #Find the <END> token
                        done_set.add(idx)
                t += 1

            cond_score = torch.stack(scores, 1)

        if reinforce:
            return cond_score, choices
        else:
            return cond_score


    def loss(self, score, gt_seq):
        # pred_agg, pred_sel, pred_cond = pred_entry
        cond_score = score
        loss = 0
        for b in range(len(gt_seq)):
            if self.gpu:
                cond_truth_var = Variable(
                        torch.from_numpy(np.array(gt_seq[b][1:])).cuda())
            else:
                cond_truth_var = Variable(
                        torch.from_numpy(np.array(gt_seq[b][1:])))
            cond_pred_score = cond_score[b, :len(gt_seq[b])-1]

            loss += ( self.CE(
                    cond_pred_score, cond_truth_var) / len(gt_seq) )

        return loss

    def step(self, all_inp_var, all_name_len, all_len, q_embedding, q_len, type_embedding ,type_len):

        B = len(q_len)
        # embedding dividely

        max_x_len = max(all_len)
        # encode dividely
        e_q, h_q = run_lstm(self.encoder, q_embedding, q_len)
        e_type, h_type = run_lstm(self.encoder, type_embedding ,type_len)
        e_sql, h_sql = col_name_encode(all_inp_var, all_name_len, all_len, self.encoder)

        # attention between q and all:

        att_val = torch.bmm(e_sql, self.att(e_q).transpose(1, 2))
        max_q_len = max(q_len)
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val[idx, :, num:] = -100
        att = self.softmax(att_val.view((-1, max_q_len))).view(B, -1, max_q_len)
        e_all_expand = (e_q.unsqueeze(1) * att.unsqueeze(3)).sum(2)
        #cat
        # e_all_expand = torch.cat((e_all_type, e_all_cat), 2)

        h_enc, hidden = run_lstm(self.cond_lstm, e_all_expand, all_len)
        decoder_hidden = tuple(torch.cat((hid[:2], hid[2:]),dim=2) for hid in hidden)

        scores = []
        choices = []
        done_set = set()
        h_enc_expand = h_enc.unsqueeze(1)

        t = 0
        init_inp = np.zeros((B, 1, self.max_tok_num), dtype=np.float32)
        init_inp[:, 0, 7] = 1  #Set the <BEG> token
        if self.gpu:
            cur_inp = Variable(torch.from_numpy(init_inp).cuda())
        else:
            cur_inp = Variable(torch.from_numpy(init_inp))
        cur_h = decoder_hidden
        while len(done_set) < B and t < 100:
            g_s, cur_h = self.decoder(cur_inp, cur_h)
            g_s_expand = g_s.unsqueeze(2)

            cur_cond_score = self.cond_out(self.cond_out_h(h_enc_expand) +
                        self.cond_out_g(g_s_expand)).squeeze()
            for b, num in enumerate(all_len):
                if num < max_x_len:
                    cur_cond_score[b, num:] = -100
            scores.append(cur_cond_score)


            _, ans_tok_var = cur_cond_score.view(B, max_x_len).max(1)
            ans_tok_var = ans_tok_var.unsqueeze(1)

            ans_tok = ans_tok_var.data.cpu()
            if self.gpu:  #To one-hot
                cur_inp = Variable(torch.zeros(B, self.max_tok_num).scatter_(1, ans_tok, 1).cuda())
            else:
                cur_inp = Variable(torch.zeros(
                        B, self.max_tok_num).scatter_(1, ans_tok, 1))
            cur_inp = cur_inp.unsqueeze(1)

            for idx, tok in enumerate(ans_tok.squeeze()):
                if tok == 1:  #Find the <END> token
                    done_set.add(idx)
            t += 1

        cond_score = torch.stack(scores, 1)
        return cond_score


    def sample(self, all_inp_var, all_name_len, all_len, q_embedding, q_len, type_embedding ,type_len, all_seq):
        """

        :param batch_size:
        :param seq_len:
        :param x_emb_var:
        :param x_len:
        :param gt_where:
        :param hidden:
        :param reinforce:
        :param flag: whether sample from zero
        :return:
        """


        output = self.step(all_inp_var, all_name_len, all_len, q_embedding, q_len, type_embedding ,type_len)
        # todo 序列下标到词库下标的映射,否则判别器无法奏效

        _, choice = output.topk(1)
        choices = choice.squeeze(2)
        #choices = choices.cpu().numpy()
        result = []

        for q, c in zip(all_seq, choices):
            result.append([q[c_i] for c_i in c])
        return result

    def MCsample(self, all_inp_var, all_name_len, all_len, q_embedding, q_len, type_embedding ,type_len, all_tok, data):
        x = data
        s, index = x.topk(1)
        index = index.squeeze(2)

        B = len(q_len)
        # embedding dividely

        max_x_len = max(all_len)
        # encode dividely
        e_q, h_q = run_lstm(self.encoder, q_embedding, q_len)
        e_type, h_type = run_lstm(self.encoder, type_embedding ,type_len)
        e_sql, h_sql = col_name_encode(all_inp_var, all_name_len, all_len, self.encoder)

        # attention between q and all:

        att_val = torch.bmm(e_sql, self.att(e_q).transpose(1, 2))
        max_q_len = max(q_len)
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val[idx, :, num:] = -100
        att = self.softmax(att_val.view((-1, max_q_len))).view(B, -1, max_q_len)
        e_all_expand = (e_q.unsqueeze(1) * att.unsqueeze(3)).sum(2)
        #cat
        # e_all_expand = torch.cat((e_all_type, e_all_cat), 2)

        h_enc, hidden = run_lstm(self.cond_lstm, e_all_expand, all_len)
        decoder_hidden = tuple(torch.cat((hid[:2], hid[2:]),dim=2) for hid in hidden)

        h_enc_expand = h_enc.unsqueeze(1)
        # B = len(x_len)
        # max_x_len = max(x_len)
        choices = []

        done_set = set()
        cur_h = decoder_hidden
        l = x.size(1)
        t = l
        #print(l)
        # while len(done_set) < B and t < 100:
        for j in range(l):
            cur_inp = x[:, j, :].unsqueeze(1)
            g_s, cur_h = self.decoder(cur_inp, cur_h)
        while len(done_set) < B and l-1 < t < 50:

            #cur_inp = x[:, l-1, :].unsqueeze(1)
            #print(cur_inp.shape)

            g_s, cur_h = self.decoder(cur_inp, cur_h)
            g_s_expand = g_s.unsqueeze(2)

            cur_cond_score = self.cond_out(self.cond_out_h(h_enc_expand) +
                                           self.cond_out_g(g_s_expand)).squeeze()
            for b, num in enumerate(all_len):
                if num < max_x_len:
                    cur_cond_score[b, num:] = -100
            # scores.append(cur_cond_score)
            _, ans_tok_var = cur_cond_score.view(B, max_x_len).max(1)
            ans_tok_var = ans_tok_var.unsqueeze(1)
            cur_inp = torch.zeros(ans_tok_var.size(0), self.max_tok_num).scatter_(1, ans_tok_var, 1)
            cur_inp = cur_inp.view(B,1,self.max_tok_num)
            #print(cur_inp)
            #print(ans_tok_var)

            choices.append(ans_tok_var)
            ans_tok = ans_tok_var.data.cpu()
            for idx, tok in enumerate(ans_tok.squeeze()):
                if tok == 1:  # Find the <END> token
                    done_set.add(idx)
            t+=1
        # after l
        choices = torch.cat(choices, 1)

        # all
        choices = torch.cat((index,choices),1)
        #print(choices.shape)

        samples_list=[]
        # for q, c in zip(q_seq, choices):
        #     sample += [q[c_i] for c_i in c]
        #     samples.append(sample)
        # print(len(samples[0]))
        for i in range(B):
            sample = []
            for c in choices[i]:
                sample.append(all_tok[i][c])
            samples_list.append(sample)

        samples = []
        for sample_list in samples_list:
            if '<END>' in sample_list:
                sample_list = sample_list[: sample_list.index('<END>') + 1]
            samples.append(sample_list)
        for sample in samples:
            if len(sample) < 3:
                samples.remove(sample)
            # print(sample)
        return samples


    def check_acc(self, vis_info, pred_queries, gt_queries, pred_entry):
        def pretty_print(vis_data):
            print ('question:', vis_data[0])
            print ('headers: (%s)'%(' || '.join(vis_data[1])))
            print ('query:', vis_data[2])

        def gen_cond_str(conds, header):
            if len(conds) == 0:
                return 'None'
            cond_str = []
            for cond in conds:
                cond_str.append(
                    header[cond[0]] + ' ' + self.COND_OPS[cond[1]] + \
                    ' ' + str(cond[2]).lower())
            return 'WHERE ' + ' AND '.join(cond_str)

        pred_agg, pred_sel, pred_cond = pred_entry


        tot_err = agg_err = sel_err = cond_err = cond_num_err = \
                  cond_col_err = cond_op_err = cond_val_err = 0.0
        for b, (pred_qry, gt_qry) in enumerate(zip(pred_queries, gt_queries)):
            good = True
            if pred_agg:
                agg_pred = pred_qry['agg']
                agg_gt = gt_qry['agg']
                if agg_pred != agg_gt:
                    agg_err += 1
                    good = False

            if pred_sel:
                sel_pred = pred_qry['sel']
                sel_gt = gt_qry['sel']
                if sel_pred != sel_gt:
                    sel_err += 1
                    good = False

            if pred_cond:
                cond_pred = pred_qry['conds']
                cond_gt = gt_qry['conds']
                flag = True
                if len(cond_pred) != len(cond_gt):
                    flag = False
                    cond_num_err += 1

                if flag and set(
                        x[0] for x in cond_pred) != set(x[0] for x in cond_gt):
                    flag = False
                    cond_col_err += 1

                for idx in range(len(cond_pred)):
                    if not flag:
                        break
                    gt_idx = tuple(x[0] for x in cond_gt).index(cond_pred[idx][0])
                    if flag and cond_gt[gt_idx][1] != cond_pred[idx][1]:
                        flag = False
                        cond_op_err += 1

                for idx in range(len(cond_pred)):
                    if not flag:
                        break
                    gt_idx = tuple(x[0] for x in cond_gt).index(cond_pred[idx][0])
                    if flag and str(cond_gt[gt_idx][2]).lower() != \
                       str(cond_pred[idx][2]).lower():
                        flag = False
                        cond_val_err += 1

                if not flag:
                    cond_err += 1
                    good = False

            if not good:
                tot_err += 1

        return np.array((agg_err, sel_err, cond_err)), tot_err

    def gen_query(self, scores, q, col, cell_tok, raw_q, raw_col, pred_entry):
        def merge_tokens(tok_list, raw_tok_str):
            tok_str = raw_tok_str.lower()
            alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$('
            special = {'-LRB-': '(', '-RRB-': ')', '-LSB-': '[', '-RSB-': ']',
                           '``': '"', '\'\'': '"', '--': u'\u2013'}
            ret = ''
            double_quote_appear = 0
            for raw_tok in tok_list:
                if not raw_tok:
                    continue
                tok = special.get(raw_tok, raw_tok)
                if tok == '"':
                    double_quote_appear = 1 - double_quote_appear

                if len(ret) == 0:
                    pass
                elif len(ret) > 0 and ret + ' ' + tok in tok_str:
                    ret = ret + ' '
                elif len(ret) > 0 and ret + tok in tok_str:
                    pass
                elif tok == '"':
                    if double_quote_appear:
                        ret = ret + ' '
                elif tok[0] not in alphabet:
                    pass
                elif (ret[-1] not in ['(', '/', u'\u2013', '#', '$', '&']) and \
                        (ret[-1] != '"' or not double_quote_appear):
                    ret = ret + ' '
                ret = ret + tok
            return ret.strip()

        pred_agg, pred_sel, pred_cond = pred_entry

        ret_queries = []
        B = len(scores)
        all_tokens = gen_all_tokens(col, cell_tok)
        for b in range(B):
            all_toks = all_tokens[b]
            agg_ops = ['max', 'min', 'count', 'sum', 'avg']
            cur_query = {}
            out_toks = []
            # sel_index = agg_index = con_index = -1
            con_index = 0
            for score in scores[b].data.cpu().numpy():
                tok = np.argmax(score)
                val = all_toks[tok]
                if val == '<END>':
                    break
                out_toks.append(val)
            # out_toks = []
            # for val in out_tok:
            #     out_toks += val.split()
            # print(out_toks)
            cur_query['agg'] = 0
            for i in range(len(out_toks)):
                if out_toks[i] == 'SELECT':
                    sel_index = i
                if out_toks[i] in agg_ops:
                    agg_index = i
                    cur_query['agg'] = agg_ops.index(out_toks[i]) + 1
                if out_toks[i] == 'WHERE':
                    con_index = i

            # if agg_index != -1:
            #     col_tok = "".join(out_toks[agg_index + 1: con_index])
            # else:
            #     col_tok = "".join(out_toks[sel_index + 1: con_index])
            # # pre_col = merge_tokens(col_tok, raw_q[b] + ' || ' + \
            # #                            ' || '.join(raw_col[b]))
            col_tok = "".join(out_toks[con_index-1])
            to_idx = [x.lower() for x in raw_col[b]]
            if col_tok in to_idx:
                cur_query['sel'] = to_idx.index(col_tok)
            else:
                cur_query['sel'] = 0

            cond_toks = out_toks[con_index:]
            cur_query['conds'] = []

            if len(cond_toks) > 0:
                cond_toks = cond_toks[1:]
            st = 0
            while st < len(cond_toks):
                cur_cond = [None, None, None]
                ed = len(cond_toks) if 'AND' not in cond_toks[st:] \
                    else cond_toks[st:].index('AND') + st
                if 'EQL' in cond_toks[st:ed]:
                    op = cond_toks[st:ed].index('EQL') + st
                    cur_cond[1] = 0
                elif 'GT' in cond_toks[st:ed]:
                    op = cond_toks[st:ed].index('GT') + st
                    cur_cond[1] = 1
                elif 'LT' in cond_toks[st:ed]:
                    op = cond_toks[st:ed].index('LT') + st
                    cur_cond[1] = 2
                else:
                    op = st
                    cur_cond[1] = 0
                sel_col = "".join(cond_toks[op-1])
                to_idx = [x.lower() for x in raw_col[b]]
                # pred_col = merge_tokens(sel_col, raw_q[b] + ' || ' + \
                #                             ' || '.join(raw_col[b]))
                if sel_col in to_idx:
                    cur_cond[0] = to_idx.index(sel_col)
                else:
                    cur_cond[0] = 0
                cur_cond[2] = "".join(cond_toks[op + 1:ed]).split()
                cur_query['conds'].append(cur_cond)
                st = ed + 1
            ret_queries.append(cur_query)
        return ret_queries




