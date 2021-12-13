def create_masked_sents(sent_dict, cat_to_mask = 'V', mask_tok='[MASK]'):
    sent = []
    target = ''
    mask_tok = '***' + mask_tok + '***'
    for cat, word in sent_dict.items():
        if len(word) < 1: continue
        if cat == cat_to_mask:
            sent.append(mask_tok)
            target = word
        else:
            sent.append(word)
    return ' '.join(sent), target

    
path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/Plos_stimuli_separated.csv'
out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/Plos_stimuli_test_transformers.csv'

conds_dict = {'AMI':'AI', 'AFI': 'AI', 'UMI':'UI', 'UFI': 'UI', 'AMP': 'AP', 'AFP': 'AP', 'UMP':'UP', 'UFP': 'UP'}
stimuli_dict = dict()
with open(path, 'r') as rf:
    next(rf)
    for l in rf:
        l = l.strip()
        line = l.split(',')
        sent_id = line[0]
        if line[0] not in stimuli_dict:
            stimuli_dict[sent_id] = dict() #'AI', 'AP', 'UP', 'UI'}
        cond = conds_dict[line[1]]
        ag = line[6]
        erg = line[7]
        pat = line[2]
        acc = line[3]
        verb = line[4]
        aux = line[5]
        punct = line[8]
        stimuli_dict[sent_id][cond] =  {'A':ag, 'P':pat, 'ERG': erg, 'ACC': acc, 'V': verb, 'AUX': aux, 'punct': punct}
test_stimuli = dict()
with open(out_path, 'w') as wf:
    wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('sent_id', 'masked_sent', 'correct', 'typec', 'wrong1', 'wrong2', 'type1', 'type2'))
    for sent_id, sent in stimuli_dict.items():
        test_stimuli[sent_id] = dict()
        cond_AI = sent['AI']
        cond_AP = sent['AP']
        cond_UI = sent['UI']
        cond_UP = sent['UP']
        #### Ambiguous IPFV
        #sent_AI = [s for s in cond_AI.values() if len(s) > 0 ]
        #str_AI = ' '.join(sent_AI)
        str_AI, target_AI = create_masked_sents(cond_AI)

        wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(sent_id, str_AI, target_AI, 'AI', cond_AP['V'], cond_UP['V'], 'AP', 'UP')) 
 #       wf.write('{},{},{},{},{},{},{},{},{}\n'.format(sent_id, 'AI_correct', cond_AI['A'], cond_AI['ERG'], cond_AI['P'], cond_AI['ACC'], cond_AI['V'], cond_AI['AUX'], cond_AI['punct']))
        # wrong Ambiguous IPFV, verb from: UP
        #wf.write('{},{},{},{},{},{},{},{},{}\n'.format(sent_id, 'AI_UP', cond_AI['A'], cond_AI['ERG'], cond_AI['P'], cond_AI['ACC'], cond_UP['V'], cond_AI['AUX'], cond_AI['punct'])) 
        # wrong Ambiguous IPFV, verb from: AP
        #wf.write('{},{},{},{},{},{},{},{},{}\n'.format(sent_id, 'AI_AP', cond_AI['A'], cond_AI['ERG'], cond_AI['P'], cond_AI['ACC'], cond_AP['V'], cond_AI['AUX'], cond_AI['punct'])) 
 
        #### Ambiguous PFV
        str_AP, target_AP = create_masked_sents(cond_AP)
        wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(sent_id, str_AP, target_AP, 'AP', cond_AI['V'], cond_UP['V'], 'AI', 'UP')) 
        #wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(sent_id, 'AP', str_AP, cond_AI['V'], cond_UP['V'], 'AI', 'UP')) 
        #wf.write('{},{},{},{},{},{},{},{},{}\n'.format(sent_id, 'AP_correct', cond_AP['A'], cond_AP['ERG'], cond_AP['P'], cond_AP['ACC'], cond_AP['V'], cond_AP['AUX'], cond_AP['punct'])) 
        # wrong Ambiguous PFV, verb form: AI
        #wf.write('{},{},{},{},{},{},{},{},{}\n'.format(sent_id, 'AP_AI', cond_AP['A'], cond_AP['ERG'], cond_AP['P'], cond_AP['ACC'], cond_AI['V'], cond_AP['AUX'], cond_AP['punct'])) 
        # wrong Ambiguous PFV, verb form: UP
        #wf.write('{},{},{},{},{},{},{},{},{}\n'.format(sent_id, 'AP_UP', cond_AP['A'], cond_AP['ERG'], cond_AP['P'], cond_AP['ACC'], cond_UP['V'], cond_AP['AUX'], cond_AP['punct'])) 
 
        #### Unambig. IPFV
        str_UI, target_UI = create_masked_sents(cond_UI)
        wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(sent_id, str_UI, target_UI, 'UI', cond_UP['V'], cond_AP['V'], 'UP', 'AP')) 
        #wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(sent_id, 'UI', str_UI, cond_AP['V'], cond_UP['V'], 'AP', 'UP')) 

        #wf.write('{},{},{},{},{},{},{},{},{}\n'.format(sent_id, 'UI_correct', cond_UI['A'], cond_UI['ERG'], cond_UI['P'], cond_UI['ACC'], cond_UI['V'], cond_UI['AUX'], cond_UI['punct']))  
        # wrong unambig. IPFV, verb form: AP
        #wf.write('{},{},{},{},{},{},{},{},{}\n'.format(sent_id, 'UI_AP', cond_UI['A'], cond_UI['ERG'], cond_UI['P'], cond_UI['ACC'], cond_AP['V'], cond_UI['AUX'], cond_UI['punct'])) 
        # wrong unambig. IPFV, verb form: UP
        #wf.write('{},{},{},{},{},{},{},{},{}\n'.format(sent_id, 'UI_UP', cond_UI['A'], cond_UI['ERG'], cond_UI['P'], cond_UI['ACC'], cond_UP['V'], cond_UI['AUX'], cond_UI['punct'])) 
        
        #### Unambig. PFV
        str_UP, target_UP = create_masked_sents(cond_UP)
        wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(sent_id, str_UP, target_UP, 'UP', cond_AI['V'], cond_AP['V'], 'AI', 'AP'))  
        #wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(sent_id, 'UP', str_UP, cond_AI['V'], cond_AP['V'], 'AI', 'AP')) 
#        wf.write('{},{},{},{},{},{},{},{},{}\n'.format(sent_id, 'UP_correct', cond_UP['A'], cond_UP['ERG'], cond_UP['P'], cond_UP['ACC'], cond_UP['V'], cond_UP['AUX'], cond_UP['punct'])) 
        # wrong unambig. PFV, verb form: AI 
        #wf.write('{},{},{},{},{},{},{},{},{}\n'.format(sent_id, 'UP_AI', cond_UP['A'], cond_UP['ERG'], cond_UP['P'], cond_UP['ACC'], cond_AI['V'], cond_UP['AUX'], cond_UP['punct'])) 
        # wrong unambig. PFV, verb form: AP
        #wf.write('{},{},{},{},{},{},{},{},{}\n'.format(sent_id, 'UP_AP', cond_UP['A'], cond_UP['ERG'], cond_UP['P'], cond_UP['ACC'], cond_AP['V'], cond_UP['AUX'], cond_UP['punct'])) 
"""
out_path2 = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/Plos_stimuli_test_text.txt'
with open(out_path2, 'w') as wf:
    with open(out_path, 'r') as rf:
        next(rf)
        for l in rf:
            l = l.strip()
            line = l.split(',')
            sent_id = line[0]
            cond = line[1]
            sent = line[2:]
            sent_fin = []
            for s in sent:
                if len(s) >0:
                    sent_fin.append(s)
            sent_fin_str = ' '.join(sent_fin)
            wf.write('{} {} {}\n'.format(sent_id, cond, sent_fin_str))
"""


