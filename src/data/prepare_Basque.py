


def create_test_data(path, out, CR=4):
    with open(out, 'w') as wf:
        wf.write('{}\t{}\t{}\t{}\t{}\n'.format('nr', 'amb', 'align', 'sent', 'corr'))
        with open(path, 'r') as rf:
            next(rf)
            for l in rf:
                l = l.strip()
                line = l.split(',')
                if len(line[6]) < 1:
                    continue
                cond_item = '{}_{}'.format(line[2], line[1])
                amb = line[3]
                align = line[5]
                sent = line[6:13]
                cr = sent[CR]
                sent_masked = sent
                sent_masked[CR] = '***[MASK]***'
                sent_str = ' '.join(sent_masked)
                wf.write('{}\t{}\t{}\t{}\t{}\n'.format(cond_item, amb, align, sent_str, cr))
                
def create_test_data_lstm(path, out):
    print('here')
    #nr      amb     align   sent    corr    wrong1  wrong2  wrong
    with open(out, 'w') as wf:
        #wf.write('{}\t{}\t{}\n'.format('nr', 'cond', 'sent'))
        with open(path, 'r') as rf:
            next(rf)
            for l in rf:
                l = l.strip()
                line = l.split('\t')
                nr = line[0]
                amb = line[1]
                align = line[2]
                sent = line[3]
                corr = line[4]
                wrong1 = line[5]
                wrong2 = line[6]
                wrong3 = line[7]
                corr_s = sent.replace('***[MASK]***', corr)
                wr1_s = sent.replace('***[MASK]***', wrong1)
                wr2_s = sent.replace('***[MASK]***', wrong2)
                wr3_s = sent.replace('***[MASK]***', wrong3)
                cond_c = "{}{}_{}".format(amb, align, 'TRUE')
                cond_w1 =  "{}{}_{}".format(amb, align, 'FALSE1')
                cond_w2 =  "{}{}_{}".format(amb, align, 'FALSE2')
                cond_w3 =  "{}{}_{}".format(amb, align, 'FALSE3')
                wf.write('{}\t{}\t{}\n'.format(nr, cond_c, corr_s))
                wf.write('{}\t{}\t{}\n'.format(nr, cond_w1, wr1_s))
                wf.write('{}\t{}\t{}\n'.format(nr, cond_w2, wr2_s))
                wf.write('{}\t{}\t{}\n'.format(nr, cond_w3, wr3_s))
                
                



def create_LSTM_data(path, out):
    with open(out, 'w') as wf:
        with open(path, 'r') as rf:
            next(rf)
            for l in rf:
                l = l.strip()
                line = l.split(',')
                if len(line[6]) < 1:
                    continue
                cond = line[1]
                item = line[2]
                amb = line[3]
                align = line[5]
                sent = line[6:13]
                sent = ' '.join(sent)
                cr = sent[5]
                wf.write('{}\t{}\t{}\n'.format(item, cond, sent))

def create_aux_test(path, out):
    dict_stim = dict()
    with open(path, 'r') as rf:
        next(rf)
        for l in rf:
            l = l.strip()
            line = l.split(',')
            if len(line[6]) < 1:
                continue
            cond_item = '{}_{}'.format(line[2], line[1])
            amb = line[3]
            align = line[5]
            sent = line[6:13]
            cr = sent[5]
            sent_masked = sent
            sent_masked[5] = '***[MASK]***'
            sent_str = ' '.join(sent_masked)
            dict_stim[cond_item] = {'amb': amb, 'align': align, 'sent': sent_str, 'correct': cr}

    with open(out, 'w') as wf:
        
        wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('nr', 'amb', 'align', 'sent', 'corr', 'wrong1', 'wrong2', 'wrong3'))
        for k,v in dict_stim.items():
            sent_id, cond = k.split('_')
            to_include = []
            for i in range(1,5):
                if str(i) != str(cond):
                    print(i, cond)
                    to_include.append(dict_stim['{}_{}'.format(sent_id, i)]['correct'])
            print(to_include)
            wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(k, v['amb'], v['align'], v['sent'], v['correct'], to_include[0], to_include[1], to_include[2]))

#with open(out, 'w') as wf:
#wf.write('{}\t{}\t{}\t{}\n'.format(line[0], amb, align, sent_str))

if __name__ == '__main__':
    #path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/BASQUE/input/Basque_INTRs_transformers.csv'
    path =  '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/BASQUE/input/Basque_INTRs_transformer_aux.csv'
    #out = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/BASQUE/dummy.txt' 
    out = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/BASQUE/Basque_INTRs_LSTM_test.csv'
    dict_conds = {'dira':'Amb_Abs', 'dute':'Amb_Erg', 'da':'Unamb_Abs', 'du': 'Unamb_Erg'}
    
    with open(out, 'w') as wf:
        wf.write('{}\t{}\t{}\t{}\t{}\n'.format('sent_id', 'cond', 'sent', 'correct', 'cond_verb'))
        with open(path, 'r') as rf:
            for l in rf:
                l = l.strip()
                line = l.split('\t')
                sent_id = line[0]
                cond = line[1]
                sent = line[2].replace('***[MASK]***', line[3])
                corr = line[4]
                cond_verb = line[5]
                wf.write('{}\t{}\t{}\t{}\t{}\n'.format(sent_id, cond, sent, corr, cond_verb))

    """
    with open(out, 'w') as wf:
        wf.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format('sent_id', 'cond', 'sent','form', 'correct', 'cond_verb'))
        with open(path, 'r') as rf:
            next(rf)
            for l in rf:
                l = l.strip()
                line = l.split('\t')
                sent_id = line[0]
                cond = line[1] + '_' + line[2]
                sent = line[3]
                sent = sent.replace('  ', ' ')
                sent_new = sent.replace('***[MASK]***', line[4])
                sent_new = sent_new.split(' ')
                correct_form = sent_new[5]
                sent_new[5] = '***[MASK]***'
                sent_new = ' '.join(sent_new) 
                wf.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(sent_id, cond, sent_new, correct_form, 'True', cond))
                ## incorrect
                for form, condition in dict_conds.items():
                    if form == correct_form:
                        continue
                    else:
                        wf.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(sent_id, cond, sent_new, form, 'False', condition))
        
                       
    """ 

