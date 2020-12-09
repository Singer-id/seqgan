import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net_utils import run_lstm

class Seq2SQLCondPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_col_num, max_tok_num, gpu):
        super(Seq2SQLCondPredictor, self).__init__()
        print ("Seq2SQL where prediction")
        self.N_h = N_h
        self.max_col_num = max_col_num
        self.max_tok_num = max_tok_num
        #self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>','none', 'max', 'min', 'count','sum', 'avg', 'SELECT']
        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']
        self.COND_OPS = ['EQL', 'GT', 'LT']
        self.gpu = gpu

        self.cond_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.cond_decoder = nn.LSTM(input_size=self.max_tok_num,
                hidden_size=N_h, num_layers=N_depth,
                batch_first=True, dropout=0.3)

        self.cond_out_g = nn.Linear(N_h, N_h)
        self.cond_out_h = nn.Linear(N_h, N_h)
        self.cond_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.CE = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()
        if gpu:
            self.cuda()

    # def generate_gt_where_seq(self, q, col, query):
    #     agg = ['max', 'min', 'count', 'sum', 'avg']
    #     for b in range(len(query)):
    #         if query[b][1] in agg and 'WHERE' in query[b]:
    #             con_index = query[b].index('WHERE')
    #             query[b].pop(con_index - 1)
    #             query[b].pop(2)
    #     #     for i in range(len(query[b])):
    #     #         if query[b][i] in agg:
    #     #              query[b][i] = query[b][i].upper()
    #     #         if query[b][i]=='(' and query[b][i-1] in agg:
    #     #             query[b][i]=''
    #     #         if query[b][i]=='WHERE' and query[b][i-1] == ')':
    #     #             query[b][i-1] = ''
    #
    #     ret_seq = []
    #     print(query)
    #     for cur_q, cur_col, cur_query in zip(q, col, query):
    #         connect_col = [tok for col_tok in cur_col for tok in col_tok+[',']]
    #         all_toks = self.SQL_TOK + connect_col + [None] + cur_q + [None]
    #         cur_seq = [all_toks.index('<BEG>')]  # initialize cur_seq
    #         cur_seq = cur_seq + map(lambda tok: all_toks.index(tok) if tok in all_toks else 0, cur_query)
    #         cur_seq.append(all_toks.index('<END>'))  # append <END> to cur_seq
    #         ret_seq.append(cur_seq)
    #
    #     return ret_seq

    def generate_gt_where_seq(self, q, col, query):
        # data format
        # <BEG> WHERE cond1_col cond1_op cond1
        #         AND cond2_col cond2_op cond2
        #         AND ... <END>

        ret_seq = []
        for cur_q, cur_col, cur_query in zip(q, col, query):
            connect_col = [tok for col_tok in cur_col for tok in col_tok+[',']]
            all_toks = self.SQL_TOK + connect_col + [None] + cur_q + [None]
            cur_seq = [all_toks.index('<BEG>')]
            if 'WHERE' in cur_query:
                cur_where_query = cur_query[cur_query.index('WHERE'):]
                cur_seq = cur_seq + map(lambda tok:all_toks.index(tok)
                                        if tok in all_toks else 0, cur_where_query)
            cur_seq.append(all_toks.index('<END>'))
            ret_seq.append(cur_seq)
        return ret_seq




    # ground truth to onehot
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


    def forward(self, x_emb_var, x_len, gt_where, gt_cond, reinforce):
        max_x_len = max(x_len)
        B = len(x_len)

        h_enc, hidden = run_lstm(self.cond_lstm, x_emb_var, x_len)
        decoder_hidden = tuple(torch.cat((hid[:2], hid[2:]),dim=2)
                for hid in hidden)

        if gt_where is not None:
            gt_tok_seq, gt_tok_len = self.gen_gt_batch(gt_where, gen_inp=True)
            g_s, _ = run_lstm(self.cond_decoder, gt_tok_seq, gt_tok_len, decoder_hidden)

            h_enc_expand = h_enc.unsqueeze(1)
            g_s_expand = g_s.unsqueeze(2)
            cond_score = self.cond_out( self.cond_out_h(h_enc_expand) +
                    self.cond_out_g(g_s_expand) ).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    cond_score[idx, :, num:] = -100
        else:
            h_enc_expand = h_enc.unsqueeze(1)
            scores = []
            choices = []
            done_set = set()

            t = 0
            init_inp = np.zeros((B, 1, self.max_tok_num), dtype=np.float32)
            init_inp[:,0,7] = 1  #Set the <BEG> token
            if self.gpu:
                cur_inp = Variable(torch.from_numpy(init_inp).cuda())
            else:
                cur_inp = Variable(torch.from_numpy(init_inp))
            cur_h = decoder_hidden
            while len(done_set) < B and t < 100:
                g_s, cur_h = self.cond_decoder(cur_inp, cur_h)
                g_s_expand = g_s.unsqueeze(2)

                cur_cond_score = self.cond_out(self.cond_out_h(h_enc_expand) +
                        self.cond_out_g(g_s_expand)).squeeze()
                for b, num in enumerate(x_len):
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

    def loss(self, cond_score, gt_where):
        loss =0
        for b in range(len(gt_where)):
            if self.gpu:
                cond_truth_var = Variable(
                    torch.from_numpy(np.array(gt_where[b][1:])).cuda())
            else:
                cond_truth_var = Variable(
                    torch.from_numpy(np.array(gt_where[b][1:])))
            cond_pred_score = cond_score[b, :len(gt_where[b]) - 1]

            loss += (self.CE(
                cond_pred_score, cond_truth_var) / len(gt_where))
        return loss

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

        B = len(gt_queries)

        tot_err = agg_err = sel_err = cond_err = cond_num_err = \
                  cond_col_err = cond_op_err = cond_val_err = 0.0
        agg_ops = ['None', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
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


    def gen_query(self, scores, q, col, raw_q, raw_col, pred_entry, reinforce=False, verbose=False):
        def merge_tokens(tok_list, raw_tok_str):
            tok_str = raw_tok_str.lower()
            alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$('
            special = {'-LRB-':'(', '-RRB-':')', '-LSB-':'[', '-RSB-':']',
                       '``':'"', '\'\'':'"', '--':u'\u2013'}
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
        for b in range(B):
            all_toks = self.SQL_TOK + \
                       [x for toks in col[b] for x in
                        toks + [',']] + [''] + q[b] + ['']
            agg_ops = ['max', 'min', 'count', 'sum', 'avg']
            cur_query = {}
            out_toks = []
            sel_index = agg_index =con_index =-1
            for score in scores[b].data.cpu().numpy():
                tok = np.argmax(score)
                val = all_toks[tok]
                if val == '<END>':
                    break
                out_toks.append(val)
            #print(out_toks)
            cur_query['agg'] = 0
            for i in range(len(out_toks)):
                if out_toks[i] == 'SELECT':
                    sel_index = i
                if out_toks[i] in agg_ops:
                    agg_index = i
                    cur_query['agg'] = agg_ops.index(out_toks[i])+1
                if out_toks[i] =='WHERE':
                    con_index = i

            if agg_index != -1:
                col_tok = out_toks[agg_index+1:con_index]
            else: col_tok = out_toks[sel_index+1:con_index]
            pre_col = merge_tokens(col_tok, raw_q[b] + ' || ' + \
                                        ' || '.join(raw_col[b]))
            to_idx = [x.lower() for x in raw_col[b]]
            if pre_col in to_idx:
                cur_query['sel'] = to_idx.index(pre_col)
            else:
                cur_query['sel'] = 0

            cond_toks = out_toks[con_index:]
            cur_query['conds'] = []
            if verbose:
                print (cond_toks)
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
                sel_col = cond_toks[st:op]
                to_idx = [x.lower() for x in raw_col[b]]
                pred_col = merge_tokens(sel_col, raw_q[b] + ' || ' + \
                                            ' || '.join(raw_col[b]))
                if pred_col in to_idx:
                    cur_cond[0] = to_idx.index(pred_col)
                else:
                    cur_cond[0] = 0
                cur_cond[2] = merge_tokens(cond_toks[op+1:ed], raw_q[b])
                cur_query['conds'].append(cur_cond)
                st = ed + 1
            ret_queries.append(cur_query)
            #print(cur_query)
        return ret_queries

    # def gen_query(self, score, q, col, raw_q, raw_col, pred_entry,
    #               reinforce=False, verbose=False):
    #     def merge_tokens(tok_list, raw_tok_str):
    #         tok_str = raw_tok_str.lower()
    #         alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$('
    #         special = {'-LRB-':'(', '-RRB-':')', '-LSB-':'[', '-RSB-':']',
    #                    '``':'"', '\'\'':'"', '--':u'\u2013'}
    #         ret = ''
    #         double_quote_appear = 0
    #         for raw_tok in tok_list:
    #             if not raw_tok:
    #                 continue
    #             tok = special.get(raw_tok, raw_tok)
    #             if tok == '"':
    #                 double_quote_appear = 1 - double_quote_appear
    #
    #             if len(ret) == 0:
    #                 pass
    #             elif len(ret) > 0 and ret + ' ' + tok in tok_str:
    #                 ret = ret + ' '
    #             elif len(ret) > 0 and ret + tok in tok_str:
    #                 pass
    #             elif tok == '"':
    #                 if double_quote_appear:
    #                     ret = ret + ' '
    #             elif tok[0] not in alphabet:
    #                 pass
    #             elif (ret[-1] not in ['(', '/', u'\u2013', '#', '$', '&']) and \
    #                  (ret[-1] != '"' or not double_quote_appear):
    #                 ret = ret + ' '
    #             ret = ret + tok
    #         return ret.strip()
    #
    #     pred_agg, pred_sel, pred_cond = pred_entry
    #     cond_score = score
    #
    #     ret_queries = []
    #     # if pred_agg:
    #     #     B = len(agg_score)
    #     # elif pred_sel:
    #     #     B = len(sel_score)
    #     # elif pred_cond:
    #     B = len(cond_score[0]) if reinforce else len(cond_score)
    #     for b in range(B):
    #         cur_query = {}
    #         # if pred_agg:
    #         #     cur_query['agg'] = np.argmax(agg_score[b].data.cpu().numpy())
    #         # if pred_sel:
    #         #     cur_query['sel'] = np.argmax(sel_score[b].data.cpu().numpy())
    #         if pred_cond:
    #             cur_query['conds'] = []
    #             all_toks = self.SQL_TOK + \
    #                        [x for toks in col[b] for x in
    #                         toks+[',']] + [''] + q[b] + ['']
    #             cond_toks = []
    #             if reinforce:
    #                 for choices in cond_score[1]:
    #                     if choices[b].data.cpu().numpy()[0] < len(all_toks):
    #                         cond_val = all_toks[choices[b].data.cpu().numpy()[0]]
    #                     else:
    #                         cond_val = '<UNK>'
    #                     if cond_val == '<END>':
    #                         break
    #                     cond_toks.append(cond_val)
    #             else:
    #                 for where_score in cond_score[b].data.cpu().numpy():
    #                     cond_tok = np.argmax(where_score)
    #                     cond_val = all_toks[cond_tok]
    #                     if cond_val == '<END>':
    #                         break
    #                     cond_toks.append(cond_val)
    #
    #             if verbose:
    #                 print cond_toks
    #             if len(cond_toks) > 0:
    #                 cond_toks = cond_toks[1:]
    #             st = 0
    #             while st < len(cond_toks):
    #                 cur_cond = [None, None, None]
    #                 ed = len(cond_toks) if 'AND' not in cond_toks[st:] \
    #                      else cond_toks[st:].index('AND') + st
    #                 if 'EQL' in cond_toks[st:ed]:
    #                     op = cond_toks[st:ed].index('EQL') + st
    #                     cur_cond[1] = 0
    #                 elif 'GT' in cond_toks[st:ed]:
    #                     op = cond_toks[st:ed].index('GT') + st
    #                     cur_cond[1] = 1
    #                 elif 'LT' in cond_toks[st:ed]:
    #                     op = cond_toks[st:ed].index('LT') + st
    #                     cur_cond[1] = 2
    #                 else:
    #                     op = st
    #                     cur_cond[1] = 0
    #                 sel_col = cond_toks[st:op]
    #                 to_idx = [x.lower() for x in raw_col[b]]
    #                 pred_col = merge_tokens(sel_col, raw_q[b] + ' || ' + \
    #                                         ' || '.join(raw_col[b]))
    #                 if pred_col in to_idx:
    #                     cur_cond[0] = to_idx.index(pred_col)
    #                 else:
    #                     cur_cond[0] = 0
    #                 cur_cond[2] = merge_tokens(cond_toks[op+1:ed], raw_q[b])
    #                 cur_query['conds'].append(cur_cond)
    #                 st = ed + 1
    #         ret_queries.append(cur_query)
    #         # print(cur_query)
    #     return ret_queries
