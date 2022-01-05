


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
    path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/BASQUE/Basque_INTRs_sentences.csv'
    out = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/BASQUE/Basque_INTRs_transformers.csv' 
    create_test_data(path, out) 