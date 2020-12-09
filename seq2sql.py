# -*- coding: UTF-8 -*-
from torch import nn
import torch
from torch.autograd import Variable
import numpy as np

from net_utils import run_lstm


class Seq2SQL(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_col_num, max_tok_num, gpu):
        """

        :param N_word: 字典大小
        :param N_h:隐状态维度
        :param N_depth: lstm的层数
        :param max_col_num:生成的sql最多的column个数
        :param max_tok_num:生成的sql最多的token 个数
        :param gpu:
        """
        super(Seq2SQL, self).__init__()
        print("Seq2SQL  prediction")
        self.N_h = N_h
        self.max_tok_num = max_tok_num
        self.max_col_num = max_col_num
        self.gpu = gpu

        self.cond_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h / 2,
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
        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND',
            'EQL', 'GT', 'LT', '<BEG>', '', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG', 'SELECT']

        #COND_OPS = ['EQL', 'GT', 'LT']
        # if gpu:
        #     self.cuda()

    def generate_gt_seq(self, q, col, query):
        # This generates the ground truth where sequence of tokens
        """
        :param q: question tokens = the question is tokenized as a list
                    E.g: ['what', 'station', 'aired', 'a', 'game', 'at', '9:00', 'pm', '?']
        :param col: Column header tokens = each head is a list of words it
                    contains (small letters)
                    E.g: ['Time', 'Big Ten Team'] => [['time'], ['big', 'ten', 'team']]
        :param query: the Query in tokenized form
                    e.g ['SELECT', 'television', 'WHERE', 'time', 'EQL', '9:00', 'pm']

        :constant: self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']

        :return ret_seq: List of indices of tokens of WHERE clauses within the token-space of the query (='all_toks') for each question
                        Start of each list is 7, end of each list is 1, any token not present in the set union(SQL_TOK, q, col) is 0
                        E.g.[[7, 2, 27, 4, 34, 1], [7, 2, 14, 15, 16, 17, 4, 38, 1]]
        """
        ret_seq = []
        for cur_q, cur_col, cur_query in zip(q, col, query):
            connect_col = [tok for col_tok in cur_col for tok in col_tok + [',']]
            all_toks = self.SQL_TOK + connect_col + [None] + cur_q + [None]  # Get list of all tokens (all SQL tokens + col_names + question_tokens )
            cur_seq = [all_toks.index('<BEG>')]  # initialize cur_seq
            #cur_seq = []
            # if 'WHERE' in cur_query:
            #     cur_where_query = cur_query[cur_query.index('WHERE'):]  # extract the condition part of the query
            #     cur_seq = cur_seq + [all_toks.index(tok) if tok in all_toks else 0
            #                          for tok in
            #                          cur_where_query]  # append token indices of WHERE part of query from 'all_toks'
            # cur_seq.append(all_toks.index('<END>'))  # append <END> to cur_seq
            cur_seq = cur_seq + [all_toks.index(tok) if tok in all_toks else 0 for tok in cur_query]  # append token indices of WHERE part of query from 'all_toks'
            cur_seq.append(all_toks.index('<END>'))  # append <END> to cur_seq
            ret_seq.append(cur_seq)
        return ret_seq

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

    # def forward(self, x_emb_var, x_len, col_inp_var, col_name_len, col_len,
    #             col_num, gt_where, gt_cond=None, reinforce=None, hidden=None):
    def forward(self, x_emb_var, x_len, gt_where, gt_cond=None, reinforce=None, hidden=None):
        max_x_len = max(x_len)
        B = len(x_len)

        h_enc, hidden = run_lstm(self.cond_lstm, x_emb_var, x_len)
        decoder_hidden = tuple(torch.cat((hid[:2], hid[:2]),dim=2)
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
            score, choices = torch.topk(cond_score, 1, 2)
            # print(score)
            choices = choices.squeeze(2)
        else:
            choices = []
            h_enc_expand = h_enc.unsqueeze(1)
            scores = []

            done_set = set()

            t = 0
            init_inp = np.zeros((B, 1, self.max_tok_num), dtype=np.float32)
            init_inp[:, 0, 7] = 1  # Set the <BEG> token

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
                    ans_tok_var = self.softmax(cur_cond_score).multinomial(1)
                    # print ("ans_tok_var=",ans_tok_var)
                    choices.append(ans_tok_var)
                ans_tok = ans_tok_var.data.cpu()
                if self.gpu:  # To one-hot
                    cur_inp = Variable(torch.zeros(
                        B, self.max_tok_num).scatter_(1, ans_tok, 1).cuda())
                else:
                    cur_inp = Variable(torch.zeros(
                        B, self.max_tok_num).scatter_(1, ans_tok, 1))
                cur_inp = cur_inp.unsqueeze(1)

                for idx, tok in enumerate(ans_tok.squeeze()):
                    if tok == 1:  # Find the <END> token
                        done_set.add(idx)
                t += 1

            cond_score = torch.stack(scores, 1)
        #print (choices)
        return self.softmax(cond_score), choices
    def loss(self, score, gt_seq):
        loss = 0
        # Objective/cost function as mentioned in the paper
        # is the sum of all three losses.
        # During No Rl training, CE losses for all three are considered
        # For RL training - CE losses from the first two and rewards
        # from execution are used

        for b in range(len(gt_seq)): #for each training example
            truth_var = Variable(torch.from_numpy(np.array(gt_seq[b][1:]))) #target value of condition after removing <BEG> token. <END> is still present as last entry.

            pred_score = score[b, :len(gt_seq[b])-1] #extract scores for the condition. the tensor is padded with garbage values
            #print(pred_score)
            #print(truth_var)
            loss += ( self.CE(pred_score, truth_var) / len(gt_seq) )

        return loss





    def step(self, x_emb_var, x_len, gt_where, hidden=None, reinforce=True):
        max_x_len = max(x_len)
        B = len(x_len)
        # 隐藏参数hidden
        h_enc, hidden = run_lstm(self.cond_lstm, x_emb_var, x_len, hidden=hidden)
        decoder_hidden = tuple(torch.cat((hid[:2], hid[2:]), dim=2)
                               for hid in hidden)
        if gt_where is not None:
            gt_tok_seq, gt_tok_len = self.gen_gt_batch(gt_where, gen_inp=True)
            g_s, _ = run_lstm(self.cond_decoder,
                              gt_tok_seq, gt_tok_len, decoder_hidden)

            h_enc_expand = h_enc.unsqueeze(1)
            g_s_expand = g_s.unsqueeze(2)
            cond_score = self.cond_out(self.cond_out_h(h_enc_expand) +
                                       self.cond_out_g(g_s_expand)).squeeze()
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
            init_inp[:, 0, 7] = 1  # Set the <BEG> token
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
                    ans_tok_var = self.softmax(cur_cond_score).multinomial(1)
                    choices.append(ans_tok_var)
                ans_tok = ans_tok_var.data.cpu()
                if self.gpu:  # To one-hot
                    cur_inp = Variable(torch.zeros(
                        B, self.max_tok_num).scatter_(1, ans_tok, 1).cuda())
                else:
                    cur_inp = Variable(torch.zeros(
                        B, self.max_tok_num).scatter_(1, ans_tok, 1))
                cur_inp = cur_inp.unsqueeze(1)

                for idx, tok in enumerate(ans_tok.squeeze()):
                    if tok == 1:  # Find the <END> token
                        done_set.add(idx)
                t += 1

            cond_score = torch.stack(scores, 1)
            choices = torch.cat(choices, 1)
        return self.softmax(cond_score), choices, hidden

    def sample(self, x_emb_var, x_len, q_seq, hidden=None):
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


        output, choices, hidden = self.step(x_emb_var, x_len, None, hidden=hidden)
        # todo 序列下标到词库下标的映射,否则判别器无法奏效

        choices = choices.numpy()
        #choices = choices.cpu().numpy()
        result = []

        for q, c in zip(q_seq, choices):
            result += [q[c_i] for c_i in c]
        return result



    def MCsample(self, x_emb_var, x_len, q_seq, data, hidden=None):
        x = data
        s, index = x.topk(1)
        index = index.squeeze(2)

        h_enc, hidden = run_lstm(self.cond_lstm, x_emb_var, x_len)
        decoder_hidden = tuple(torch.cat((hid[:2], hid[2:]), dim=2)
                               for hid in hidden)
        h_enc_expand = h_enc.unsqueeze(1)
        B = len(x_len)
        max_x_len = max(x_len)
        choices = []

        done_set = set()
        cur_h = decoder_hidden
        l = x.size(1)
        t = l
        #print(l)
        # while len(done_set) < B and t < 100:
        for j in range(l):
            cur_inp = x[:, j, :].unsqueeze(1)
            g_s, cur_h = self.cond_decoder(cur_inp, cur_h)
        while len(done_set) < B and l-1 < t < 50:

            #cur_inp = x[:, l-1, :].unsqueeze(1)
            #print(cur_inp.shape)

            g_s, cur_h = self.cond_decoder(cur_inp, cur_h)
            g_s_expand = g_s.unsqueeze(2)

            cur_cond_score = self.cond_out(self.cond_out_h(h_enc_expand) +
                                           self.cond_out_g(g_s_expand)).squeeze()
            for b, num in enumerate(x_len):
                if num < max_x_len:
                    cur_cond_score[b, num:] = -100
            # scores.append(cur_cond_score)
            _, ans_tok_var = cur_cond_score.view(B, max_x_len).max(1)
            ans_tok_var = ans_tok_var.unsqueeze(1)
            cur_inp = torch.zeros(ans_tok_var.size(0), 250).scatter_(1, ans_tok_var, 1)
            cur_inp = cur_inp.view(B,1,250)
            #print(cur_inp)
            #print(ans_tok_var)

            choices.append(ans_tok_var)
            ans_tok = ans_tok_var.data.cpu()
            for idx, tok in enumerate(ans_tok.squeeze()):
                if tok == 1:  # Find the <END> token
                    done_set.add(idx)
            t+=1
        #print(choices)
        choices = torch.cat(choices, 1)
        choices = torch.cat((index,choices),1)
        #print(choices.shape)


        sample = []
        #sampless = []
        samples=[]
        # for q, c in zip(q_seq, choices):
        #     sample += [q[c_i] for c_i in c]
        #     samples.append(sample)
        # print(len(samples[0]))
        for i in range(B):
            for c in choices[i]:
                sample.append(q_seq[i][c])
            samples.append(sample)
            sample = []
        #print (len(samples[0]))
        return samples


        # for i in range(B):
        #     sample_1 = sampless[i*len(sample)/B: (i+1)*len(sampless)/B]
        #     samples.append(sample_1)
        # print(len(samples))












