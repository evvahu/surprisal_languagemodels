"""
Acknowledgment: 
    This implementation benefitted greatly from insights into the code from SArehalli 
    found here: https://github.com/SArehalli/NLMsCaptureSomeAgrAttr
"""

import torch
import pandas as pd
import sys
import os
sys.path.insert(0, "colorlessgreenRNNs/src/language_models")
from dictionary_corpus import Dictionary
#from colorlessgreenRNNs.src.language_models import dictionarycorpus
import fasttext 
import torch.nn.functional as F
import argparse
import numpy as np

class EvaluationSurprisal():

    def __init__(self, path_data, path_model, path_stimuli, wo, str_split = '\t', cuda=False, oov=False, map_loaction='cpu', lang='EU', ft='False'):
        self.model = torch.load(path_model, map_location=map_loaction)
        self.wo = wo
        self.ft = ft
        if self.ft:
            self.fasttext_model = fasttext.load_model('/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/FastText/cc.hi.300.bin')
        self.cuda = cuda
        self.dictionary = Dictionary(path_data)
        self.lang = lang
        #if lang == 'HI':
        #    self.cond_dict = {'AMI':'AI', 'AFI': 'AI', 'UMI':'UI', 'UFI': 'UI', 'AMP': 'AP', 'AFP': 'AP', 'UMP':'UP', 'UFP': 'UP'}
        #elif lang == 'EU':
        #    self.cond_dict = {'1': 'Amb_Unac', '2': 'Amb_Unerg', '3': 'Unamb_Unac' ,'4': 'Unamb_Unerg'}
        #else:
        #    print('invalid language:{}'.format(lang))
        self.stimuli = self.read_stimuli(path_stimuli, str_split)
        self.count = 0
        self.oov = oov
        
        if self.oov:
            self.id_dict = self.get_id_dict(wo, lang)
            print(self.id_dict)
        

    def read_stimuli(self, path_stimuli, split_str = 'hi'):
        df = pd.DataFrame(columns=['Nr', 'Condition', 'Sentence'])
        stimuli = []
        with open(path_stimuli) as f:
            next(f)
            for l in f:
                l = l.strip()
                if not l:continue
                #line = l.split(split_str)
                if split_str == 'hi':
                    line = l.split(' ', maxsplit=2)
                else:
                    line = l.split('\t')
                print(line)
                idx = line[0]
                cond = line[1]
                toks = line[2]
           #     if len(cond) > 3:
            #        cond_f = cond.split('_')[0]
                   #add_inf = cond.split('_')[1]
              #  elif len(cond) == 3:
             #       cond_f = self.cond_dict[line[1]]
              #  else:
               #     cond_f = cond
                if self.wo == 'B':
                    stimuli.append((idx, cond, toks))
                else:
                    #toks = line[2:]
                    correct = line[3]
                    if self.lang == 'DE':
                        form_of_present = toks.split(' ')[-1].strip('.')
                    else:
                        form_of_present = line[4]
                    #print('number of tokens in one stimuli: {}'.format(len(toks)))
                    stimuli.append((idx, cond, toks,correct, form_of_present))
                    #stimuli.append((idx, cond, toks, line[3], line[4]))
        return stimuli


    def indexify(self, word):
       
        if word not in self.dictionary.word2idx:
            print("Warning: {} not in vocab".format(word))
            self.count +=1
        return self.dictionary.word2idx[word] if word in self.dictionary.word2idx else self.dictionary.word2idx["<unk>"]
    
    def check_oov(self, sent, id_oov):
        oov = False
        list_oovs = []
        for i,t in enumerate(sent[:id_oov+1]):
            if t not in self.dictionary.word2idx:
                oov = True
                list_oovs.append(i)
            else:
                continue
        return oov, list_oovs
    def check_oov_de(self, sent):
        oov = False
        sent = sent[-6:]
        list_oovs = []
        for i,t in enumerate(sent):
            if t not in self.dictionary.word2idx:
                oov = True
                list_oovs.append(i)
            else:
                continue
        return oov, list_oovs
    def check_oov_de3(self,sent):
        oov = False
        list_oovs = []
        not_consider = [0, 5,6,7]
        sent = sent[:10]
        print(sent)
        for i, t in enumerate(sent):
            if i in not_consider:
                continue
            else:
                if t not in self.dictionary.word2idx:
                    oov = True
                    list_oovs.append(i)
                else:
                    continue
        return oov, list_oovs


    def get_surprisals(self):
        with torch.no_grad():
            surpss = []
            probss = []
            total_surps = []
            all_stims = []
            oovs_dict = dict()
  
            for stimmy in self.stimuli:
                if self.wo == 'B':
                    id, cond, sentence = stimmy
                else:
                    id, cond, sentence, correct, actual_form = stimmy

            #for id, cond, sentence, correct, type_form in self.stimuli:
                if type(sentence) == list:
                    sentence = sentence[0].split(' ')
                if sentence.endswith('.'):
                    sentence = sentence.replace('.', ' .')
                if self.lang == 'DE':
                    sentence = sentence.replace(',', ' ,')
                    
                sentence = sentence.split(' ')
                
                if self.oov:
                    if self.lang == 'DE':
                        print('correct lang')
                        OOV, oovs_list = self.check_oov_de(sentence)
                    elif self.lang == 'DE3':
                        OOV, oovs_list = self.check_oov_de3(sentence)
                    else:
                        if self.lang == 'DE2':
                            id_oov = 10
                        else:
                            id_oov = self.id_dict[str(cond)]
                        OOV, oovs_list = self.check_oov(sentence, id_oov)
                    oovs_dict[id] = oovs_list
                else:
                    OOV = False
                
                if not OOV: 
                    #all_stims.append([id,cond, add_inf, sentence])
                    if self.wo == 'T':
                        all_stims.append([id, cond, sentence, correct, actual_form])
                    else:
                        all_stims.append([id, cond, sentence])
                    if self.ft == True:
                        tens_input = torch.empty(size=(len(sentence), 300))
                        
                        for i, word in enumerate(sentence):
                            tens_input[i] = torch.from_numpy(self.fasttext_model.get_word_vector(word))
                        tens_input = tens_input[None]
                        print('tens, shape', tens_input.shape)
                        #input = [torch.from_numpy(self.fasttext_model.get_word_vector(word)) for word in sentence]
                        #input = torch.LongTensor(input)
                    #else:
                    input = torch.LongTensor([self.indexify(w) for w in sentence])
                    if self.cuda:
                        input = input.cuda()
                    
                    if self.ft:
                        out, _ = self.model(tens_input, self.model.init_hidden(len(sentence)))
                    else:
                        out, _ = self.model(input.view(-1,1), self.model.init_hidden(1))

                    #out, _ = self.model(tens_input.view(-1, 1), self.model.init_hidden(1))
                    # Get surprisals for all words 
                    surps = []
                    probs = []
                    if self.ft:
                        out = out[0]
                    for i, word_idx in enumerate(input):
                        surps.append(-F.log_softmax(out[i], dim=-1).view(-1)[word_idx].item())
                        probs.append(F.softmax(out[i], dim=-1).view(-1)[word_idx].item())
                    surpss.append(surps)
                    probss.append(probs)
                        # Get surprisals over the fill sentence
                    total_surps.append(sum([-F.log_softmax(out[i], dim=-1).view(-1)[word_idx].item() 
                                                for i,word_idx in enumerate(input)]))
                else:
                    print('oooooov')
                    continue
        return surpss, probss, total_surps, all_stims, oovs_dict
#def to_file(vs1, vs2, out_path):
    def oovs(self, out_path):
        oov = False
        with open(out_path, 'w') as wf:
            wf.write('{}\t{}\t{}\n'.format('sent_id', 'tok_nr', 'tok_str'))
            for id, cond, sentence in self.stimuli:
                for i,t in enumerate(sentence):
                    if t not in self.dictionary.word2idx:
                        wf.write('{}\t{}\t{}\n'.format(id, i, t))
                    else:
                        continue
    def get_id_dict(self, wo, lang):
        id_dict = dict()
        print(wo)
        assert wo == 'B' or wo == 'T', 'wrong word order name'
        if lang == 'HI':
            if wo == 'B':
                id_dict['AMI'] = 1
                id_dict['AMP'] = 1
                id_dict['UMP'] = 2
                id_dict['UMI'] = 2
                id_dict['AFI'] = 1
                id_dict['AFP'] = 1
                id_dict['UFP'] = 2
                id_dict['UFI'] = 2
            else:  # wo =='T':
                id_dict['AMI'] = 2
                id_dict['AMP'] = 3
                id_dict['UMP'] = 4
                id_dict['UMI'] = 4
                id_dict['AFI'] = 2
                id_dict['AFP'] = 3
                id_dict['UFP'] = 4
                id_dict['UFI'] = 4
        elif lang == 'HI2':
            id_dict['A'] = 2 # Unamb P IPFV
            id_dict['B'] = 1  # Amb P IPFV
            id_dict['C'] = 1  # Amb A IPFV
            id_dict['D'] = 1  # Amb S IPFV
            id_dict['E'] = 2  # Unamb P PFV
            id_dict['F'] = 2 # Unamb A PFV
            id_dict['G'] = 1 # Amb P PFV
            id_dict['H'] = 1 # Amb S PFV

        elif lang =='EU':
            if wo =='B':
                id_dict['1'] = 4
                id_dict['2'] = 4
                id_dict['3'] = 4
                id_dict['4'] = 4
            else:
                id_dict['Amb_Abs'] = 5
                id_dict['Amb_Erg'] = 5
                id_dict['Unamb_Abs'] = 5
                id_dict['Unamb_Erg'] = 5
        elif lang.startswith('DE'):
            return id_dict
        else:
            print('unknown lang')
        return id_dict


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("data_path")
    argp.add_argument("model_path")
    argp.add_argument("stimuli_path")
    argp.add_argument("out_path")
    argp.add_argument("--lang", type=str, default='EU')
    argp.add_argument("--wo", type=str, default='B') # B or T for Bickel or test 
    argp.add_argument("--seed", type=int, default=1)
    argp.add_argument("--cuda", action="store_true")
    argp.add_argument("--oov", action="store_true") 
    argp.add_argument("--ft", action="store_true")
    argp = argp.parse_args()
    # Make it reproduceable
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    torch.manual_seed(argp.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(argp.seed)
    str_split = ' '
    EvSurp = EvaluationSurprisal(argp.data_path, argp.model_path, argp.stimuli_path, argp.wo, str_split, argp.cuda, argp.oov, map_location, argp.lang, argp.ft)
    #EvSurp.oovs(argp.out_path)
    surp_values, probs_values, total_surps, allstims, oovs_dict = EvSurp.get_surprisals()

    out_path1 = argp.out_path.replace('.csv', '_oov.csv')
    #os.path.join(argp.out_path, 'out_of_VOC_stim.csv' )
    with open(out_path1, 'w') as wf:
        wf.write('{}\t{}\n'.format('sent_id', 'word_nrs'))
        for k,v in oovs_dict.items():
            wf.write('{}\t{}\n'.format(k,v))

    #print('length surprisals: {}'.format(len(surp_values)))
    #print('length stimuli: {}'.format(len(EvSurp.stimuli)))
    #print('all surprisal values', surp_values)
    #print('all stimuli', EvSurp.stimuli)
     #id, cond, sentence, correct, actual_form 
    with open(argp.out_path, 'w') as wf:
        wf.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format("sent_id", "cond", "word", "probability", "surprisal", "correct", "form"))
        for surp, prob, stim, in zip(surp_values, probs_values, allstims):
            stimuli = stim[2]
            #print('stim', stim)
            #print('length surprisal : {}, length stimuli: {}'.format(len(surp), len(stimuli)))
            verb = stimuli[-3]
            for su, pr, st in zip(surp, prob, stimuli):
                
                if argp.lang =='EU':
                    cond = stim[1]
                cond = stim[1] 
    
                    #cond, cond2 = EvSurp.cond_dict[stim[1]].split('_')
                if argp.wo == 'T':
                    if argp.lang == 'DE':
                        cond = cond + '_' + verb
                        print('condition: ', cond)
                        wf.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(stim[0], cond, st, pr, su, stim[3], stim[4])) 
                    else:
                        wf.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(stim[0], cond, st, pr, su, stim[3], stim[4])) 
                else:
                    wf.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(stim[0], cond, st, pr, su, '', '')) 

    """
    with open(argp.out_path, 'w') as wf:
        wf.write("{}\t{}\t{}\t{}\t{}\t{}\n".format("sent_id", "cond", "cond2", "word", "probability", "surprisal"))
        for surp, prob, stim, in zip(surp_values, probs_values, allstims):
            stimuli = stim[3]
            print('stim', stim)
            print('length surprisal : {}, length stimuli: {}'.format(len(surp), len(stimuli)))
            for su, pr, st in zip(surp, prob, stimuli):
                
                if argp.lang =='EU':
                    cond = stim[1]
                    cond2 = stim[2]
                    #cond, cond2 = EvSurp.cond_dict[stim[1]].split('_')
                wf.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(stim[0], cond, cond2, st, pr, su))
    
    """
    #print('length of corpus: {}'.format(len(EvSurp.dictionary)))
    #to_file(surp_values, total_surps, argp.out_path)