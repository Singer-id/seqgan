# simple grammar score
    # * represents 0 or more times, + represents 1 or more times, ? represents 0 or 1 times
# Sent_S = Sel_S (SUB_REL Sel_S)*, SUB_REL = {intersect, union, except}
# Sel_S = (A_S)+ F_S (W_S)?
# A_S = (agg)+ column
# F_S = from table
# W_S = C_S (CON_REL C_S)*, CON_REL = {and}
# C_S = column OP cell, OP = {EQL, GT, LT}

# a stack s, a buffer b, score, unit_num
    # type = {sql_kw, agg, op, table, column, cell}
    # string_type = {number, text, none}, {min, max, sum, avg, LT, GT} = number, {count, EQL} = none
import random



def get_type(sample, col, cell):

    agg = ['None', 'max', 'min', 'count', 'sum', 'avg']
    op = ['EQL', 'GT', 'LT']
    sql_kw = ['<UNK>', '<END>', 'WHERE', 'AND', '<BEG>', 'SELECT']
    # for i in range(len(sample)):
    if sample in sql_kw:
        type = 'sql_kw'
    elif sample in agg:
        type = 'agg'
    elif sample in op:
        type = 'OP'
    elif sample in col:
        type = 'column'
    elif sample in cell:
        type = 'cell'
    else:
        type = 'UNK'
    return type

def get_col(ge_col, ge_cell, col, all_cell):
    index = col.index(ge_col)
    col_cell = all_cell[index]
    input_cell = []
    for val in col_cell:
        if '||' in val:
            input_cell.append(' '.join(val[val.index('||') + 1:]))
    if ge_cell in input_cell:
        return True
    else:
        return False

def string_type(sample, all_tokens, all_type):
    index = all_tokens.index(sample)
    str_type = all_type[index]
    return str_type


def grammar(samples, col, cell, all_tokens, all_type, all_cell):
    s = []
    #s_iter = -1
    b = samples
    #b_iter = 0
    if len(b)>0:
        s.append(b[0])
    s_iter = 0
    b_iter = 1
    score = 0
    unit_num = 0
    while len(b)>0:
        unit_score = 0
        unit_prescore = 0
        if s_iter < 0 and b_iter >= len(b):
            break
        elif s_iter < 0 and b_iter < len(b):
            s.append(b[b_iter])
            b_iter +=1
            s_iter +=1
        elif b_iter >= len(b):
            # if s[s_iter] == 'F_S':
            #     s.pop()
            #     s_iter -=1
            #     continue
            if s_iter - 1 >= 0:
                if s[s_iter - 1] == 'SELECT' and s[s_iter] == 'A_S' or s[s_iter - 1] == 'WHERE' and s[s_iter] == 'C_S' or s[s_iter - 1] == 'AND' and s[s_iter] == 'C_S':
                    score += 1
                    s.pop()
                    s.pop()
                    s_iter -= 2
                else:
                    s.pop()
                    s_iter -=1

            else:
                s.pop()
                s_iter -=1
            unit_num += 1
        elif get_type(s[s_iter], col, cell) == 'agg':  # process A_S
            unit_num += 1
            unit_score = 2
            unit_prescore = 0
            if get_type(b[b_iter], col, cell) == 'column':
                unit_prescore +=1
                if string_type(s[s_iter],all_tokens,all_type) == string_type(b[b_iter],all_tokens,all_type) or string_type(s[s_iter],all_tokens,all_type) == 'none':
                    unit_prescore +=1
            score += unit_prescore * 1.0 / unit_score
            s.pop()
            s.append('A_S')
            b_iter +=1
        # elif s[s_iter] == 'from':  # process F_S
        #     unit_num +=1
        #     unit_score = 1
        #     unit_prescore = 0
        #     if type(b[b_iter]) == 'table':
        #         unit_prescore +=1
        #     score += unit_prescore * 1.0 / unit_score
        #     s.pop()
        #     s.append('F_S')
        #     b_iter +=1
        elif get_type(s[s_iter], col, cell) == 'column' and get_type(b[b_iter], col, cell) == 'OP' and b_iter + 1 < len(b) and get_type(b[b_iter + 1], col, cell) == 'cell':  # process C_S
            unit_num +=1
            unit_score = 3
            unit_prescore = 0
            if string_type(s[s_iter],all_tokens,all_type) == string_type(b[b_iter],all_tokens,all_type) or string_type(b[b_iter],all_tokens,all_type) == 'none':
                unit_prescore +=1

            if string_type(b[b_iter],all_tokens,all_type) == string_type(b[b_iter + 1],all_tokens,all_type) or string_type(b[b_iter],all_tokens,all_type) == 'none':
                unit_prescore +=1

            if get_col(s[s_iter], b[b_iter + 1], col, all_cell):
                unit_prescore +=1

            score += unit_prescore * 1.0 / unit_score
            s.pop()
            s.append('C_S')
            b_iter += 2
        elif s[s_iter] == 'SELECT' and get_type(b[b_iter], col, cell) == 'column':
            unit_num +=1
            score += 1
            s.append('A_S')
            s_iter +=1
            b_iter +=1
        elif b_iter < len(b):
            s.append(b[b_iter])
            s_iter +=1
            b_iter +=1
        else :
            break

    if unit_num != 0:
        grammar_score = score * 1.0 / unit_num
    else:
        grammar_score = 0
    return grammar_score

