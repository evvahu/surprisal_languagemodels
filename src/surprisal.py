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

import torch.nn.functional as F
import argparse
import numpy as np

class EvaluationSurprisal():

    def __init__(self, path_data, path_model, path_stimuli, wo, cuda=False, oov=False, map_loaction='cpu'):
        self.model = torch.load(path_model, map_location=map_loaction)

        self.cuda = cuda
        self.dictionary = Dictionary(path_data)
        self.cond_dict = {'AMI':'AI', 'AFI': 'AI', 'UMI':'UI', 'UFI': 'UI', 'AMP': 'AP', 'AFP': 'AP', 'UMP':'UP', 'UFP': 'UP'}
        self.stimuli = self.read_stimuli(path_stimuli)
        self.count = 0
        self.oov = oov
        
        if self.oov:
            self.id_dict = self.get_id_dict(wo)
        
    def read_stimuli(self, path_stimuli):
        df = pd.DataFrame(columns=['Nr', 'Condition', 'Sentence'])
        stimuli = []
        with open(path_stimuli) as f:
            for l in f:
                l = l.strip()
                if not l:continue
                print(l)
                line = l.split(' ')
                print(line)
                idx = line[0]
                cond = line[1]
                add_inf = ''
                if len(cond) > 3:
                    cond_f = cond.split('_')[0]
                    add_inf = cond.split('_')[1]
                elif len(cond) == 3:
                    cond_f = self.cond_dict[line[1]]
                else:
                    cond_f = cond
                toks = line[2:]
                print('number of tokens in one stimuli: {}'.format(len(toks)))
                stimuli.append((idx, cond_f,add_inf, toks))
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
            print(sent,i, id_oov)
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
            for id, cond, add_inf, sentence in self.stimuli:
                if self.oov:
                    id_oov = self.id_dict[cond]
                    OOV, oovs_list = self.check_oov(sentence, id_oov)
                    oovs_dict[id] = oovs_list
                    print(id, OOV)
                else:
                    OOV = False
                if not OOV: 
                    all_stims.append([id,cond, add_inf, sentence])
                    print(OOV)
                    input = torch.LongTensor([self.indexify(w) for w in sentence])
                    
                    if self.cuda:
                        input = input.cuda()

                    out, _ = self.model(input.view(-1, 1), self.model.init_hidden(1))

                    # Get surprisals for all words 
                    surps = []
                    probs = []

                    for i, word_idx in enumerate(input):
                        surps.append(-F.log_softmax(out[i], dim=-1).view(-1)[word_idx].item())
                        probs.append(F.softmax(out[i], dim=-1).view(-1)[word_idx].item())
                    surpss.append(surps)
                    probss.append(probs)
                        # Get surprisals over the fill sentence
                    total_surps.append(sum([-F.log_softmax(out[i], dim=-1).view(-1)[word_idx].item() 
                                                for i,word_idx in enumerate(input)]))
                else:
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
    def get_id_dict(self, wo):
        id_dict = dict()
        print(wo)
        assert wo == 'B' or wo == 'T', 'wrong word order name'
        if wo == 'B':
            id_dict['AI'] = 1
            id_dict['AP'] = 1
            id_dict['UP'] = 2
            id_dict['UI'] = 2
        else:  # wo =='T':
            id_dict['AI'] = 2
            id_dict['AP'] = 3
            id_dict['UP'] = 4
            id_dict['UI'] = 4
        return id_dict


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("data_path")
    argp.add_argument("model_path")
    argp.add_argument("stimuli_path")
    argp.add_argument("out_path")
    argp.add_argument("--wo", type=str, default='B') # B or T for Bickel or test 
    argp.add_argument("--seed", type=int, default=1)
    argp.add_argument("--cuda", action="store_true")
    argp.add_argument("--oov", action="store_true") 
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

    EvSurp = EvaluationSurprisal(argp.data_path, argp.model_path, argp.stimuli_path, argp.wo, argp.cuda, argp.oov, map_location)
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

    with open(argp.out_path, 'w') as wf:
        wf.write("{}\t{}\t{}\t{}\t{}\t{}\n".format("sent_id", "cond", "cond2", "word", "probability", "surprisal"))
        for surp, prob, stim, in zip(surp_values, probs_values, allstims):
            stimuli = stim[3]
            print('length surprisal : {}, length stimuli: {}'.format(len(surp), len(stimuli)))
            for su, pr, st in zip(surp, prob, stimuli):
                cond2 = stim[2]
                wf.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(stim[0], stim[1],stim[2], st, pr, su))
    

    #print('length of corpus: {}'.format(len(EvSurp.dictionary)))
    #to_file(surp_values, total_surps, argp.out_path)