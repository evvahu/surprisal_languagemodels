from transformers import AutoTokenizer, AutoModel, ElectraTokenizer, ElectraForMaskedLM, BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM, ElectraModel, ElectraForPreTraining
import torch
import numpy as np
from transformers import pipeline
device = torch.device('cpu')


#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class TransformerSurprisal():

    def __init__(self, model_name, stimuli_path, out_path, transf = 'bert',task='MLM', lang='Hi'):
        self.s_path = stimuli_path
        self.task = task
        if transf == 'electra':
            if self.task == 'MLM': 
                self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
                self.model = ElectraForMaskedLM.from_pretrained(model_name)#.to(device)
                self.mask_tok = '[MASK]'
            elif self.task == 'discriminator':
                self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
                self.model = ElectraForPreTraining.from_pretrained(model_name)
                self.mask_tok = '[MASK]'
            else:
                print('invalid task {} selected for electra'.format(self.task))
        elif transf == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForMaskedLM.from_pretrained(model_name)
            self.mask_tok = '[MASK]'
        elif transf == 'roberta':
            #self.get_preds_roberta(model_name)
            #self.model_name = model_name
            #print('call method: get_preds_roberta')
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForMaskedLM.from_pretrained(model_name)
            self.mask_tok = '<mask>'
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.mask_tok = '[MASK]'
        #if self.model:
        #self.model.eval()
        self.write = False
        if len(out_path) > 0:
            self.write = True
            
        self.out_path = out_path

    def _get_discriminator_preds(self, sent):
        print(sent)
        pre, target, post = sent.split("***")
        tokens = ['[CLS]'] + self.tokenizer.tokenize(pre)
        target_idx = len(tokens)
        target = [target]
        tokens += target + self.tokenizer.tokenize(post) + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        oov_id = self.tokenizer.convert_tokens_to_ids('[oov]')
        print(oov_id, input_ids[target_idx])
        if oov_id == input_ids[target_idx]:
            return None
        inp = torch.LongTensor(input_ids).unsqueeze(0)
        out = self.model(inp) #, labels=labels)
        probs = torch.sigmoid(out.logits)
        return probs[0][target_idx]

    def get_scores_eu(self):
        if self.write:
            wf = open(self.out_path, 'w')
            wf.write('sent_id\tamb\talign\tscore_correct\n')
        for id, c1, c2, sent, wc in self.read_stimuli_eu(self.s_path):

            print('sent input', sent)
            score = self._get_preds(sent, [wc])
            if not score:
                continue
            else:
                if self.write:
                    wf.write('{}\t{}\t{}\t{}\n'.format(id, c1,c2, score[0]))

    def get_all_scores_eu_aux(self):
        all_scores = dict()
        lt = 0
        if self.write:
            wf = open(self.out_path, 'w')
            #wf.write('sent_id\tamb\talign\tscore_correct\tscore_incorr1\tscore_incorrec2\tscore_inccorect3\tlargerthan\n')
            wf.write('{}\t{}\t{}\t{}\t{}\n'.format('sent_id', 'amb', 'align', 'score', 'word'))
        for sent, wc, w1, w2, w3, s_id, c1, c2 in self.read_stimuli_eu_aux(self.s_path):
            scores = self._get_preds(sent, [wc, w1, w2, w3])
            if not scores:
                continue
            if self.write:
                larger_than = False
                if (scores[0] > scores[1]) and (scores[0]>scores[2]):
                    larger_than = True
                    lt +=1
                for indi, (wo, s) in enumerate(zip([wc,w1, w2,w3], scores)):
                    if indi == 0:
                        wo = wo + '_corr'
                    wf.write('{}\t{}\t{}\t{}\t{}\n'.format(s_id, c1, c2, s, wo))
                    #wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(s_id, c1, c2, scores[0], scores[1], scores[2], scores[3], larger_than))
            all_scores[s_id] = scores

        print('largest: {}'.format(lt))
        return all_scores

    def get_all_scores_hi_verb(self):
        writer = open(self.out_path, 'w') 
        writer.write('{}\t{}\t{}\t{}\n'.format('id', 'ambig','aspect', 'score'))
        for id, cond, sent, corr in self.read_stimuli_hi_verb(self.s_path):
            if self.task == 'MLM':
                score = self._get_preds(sent, [corr])
            else:
                curr_sent = sent.replace('***[MASK]***', corr)
                score = self._get_discriminator_preds(curr_sent)
            if score:
                score = score[0]
            cond1 = cond[0]
            cond2 = cond[1]
            writer.write('{}\t{}\t{}\t{}\n'.format(id, cond1,cond2, score))
        
        writer.close()

    def get_all_scores_hi_aux(self):
        wf = open(self.out_path, 'w')
        wf.write('{}\t{}\t{}\t{}\t{}\n'.format('sent_id','cond', 'score', 'correct_form', 'cond_verb'))
        for stim_dict in self.read_stimuli(self.s_path):  #sent, wc, w1, w2, w3, s_id, c1, c2
            if self.task == 'MLM':
                score = self._get_preds(stim_dict['sent'], stim_dict['form'])
            else:
                curr_sent = stim_dict['sent'].replace(self.mask_tok, stim_dict['form'])
                score = self._get_discriminator_preds(curr_sent)

            wf.write('{}\t{}\t{}\t{}\t{}\n'.format(stim_dict['sent_id'], stim_dict['cond'], score, stim_dict['correct'], stim_dict['cond_verb']))


    def _get_preds(self, sent,word, test=True):
        #for sent, wc, w1, w2 in self.read_stimuli(self.s.path):
        pre, target, post = sent.split('***')
        tokens = ['[CLS]'] + self.tokenizer.tokenize(pre)
        target_idx = len(tokens)
        target= [self.mask_tok]
        if test:
            tokens += target
        else:
            tokens += target + self.tokenizer.tokenize(post) + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        oov = self.tokenizer.convert_tokens_to_ids('[oov]')
        try:
            word_id = self.tokenizer(word.strip())['input_ids']
            
            if oov in word_id:
                return None
        except:
            print('skipping',word)
            return None 

        #print(word_ids)
        tens = torch.LongTensor(input_ids).unsqueeze(0)
        #print(target_idx)
        print(target_idx)
        res = self.model(tens).logits[0, target_idx]
        #print(res)
        sm = torch.nn.Softmax(dim=0)
        res = sm(res)
        score = res[word_id]
        print(word_id, word, score)
        return float(torch.mean(score[1:-2]))

   
    def read_stimuli(self, stimuli_path):
        with open(stimuli_path, 'r') as rf:
            next(rf)
            for l in rf:
                l = l.strip()
                line = l.split('\t') 
                yield {'sent_id':line[0], 'cond':line[1], 'sent': line[2], 'form': line[3], 'correct': line[4], 'cond_verb': line[5]}

    def read_stimuli_eu(self, stimuli_path):
        with open(stimuli_path, 'r') as rf:
            next(rf)
            for l in rf:
                l = l.strip()
                s_id, c1, c2, sent, wc= l.split('\t') 
                yield s_id, c1, c2, sent, wc

    def read_stimuli_eu_aux(self, stimuli_path):
        with open(stimuli_path, 'r') as rf:
            next(rf)
            for l in rf:
                l = l.strip()
                s_id, c1, c2, sent, wc, w1, w2, w3 = l.split('\t') 
                yield sent, wc, w1, w2, w3, s_id, c1, c2

    def read_stimuli_hi(self, stimuli_path):
        with open(stimuli_path, 'r', encoding='utf8') as rf:
            next(rf)
            for l in rf:
                l = l.strip()
                s_id, sent, wc, tc, w1, w2, t1, t2 = l.split('\t')
                #print('test stimuli', wc==w1) 
                yield sent, wc, w1, w2, tc, t1, t2, s_id
#        with open(stimuli_path, 'r') as rf:
    def read_stimuli_hi_verb(self, stimuli_path):
        with open(stimuli_path, 'r') as rf:
            next(rf)
            for l in rf:
                l = l.strip()
                id, cond, sent, corr = l.split('\t')
                yield id, cond, sent, corr
if __name__ == '__main__':
    path = ''
    outpath = ''
    #modelname = 'monsoon-nlp/hindi-tpu-electra'
    #modelname = 'monsoon-nlp/hindi-bert'
    #modelname = 'ixa-ehu/berteus-base-cased'
    modelname = 'surajp/RoBERTa-hindi-guj-san'
    #modelname = 'bert-base-multilingual-cased'
    #modelname = 'google/electra-small-discriminator'
    #transi = TransformerSurprisal('hindi', path, outpath)
    #stim_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/BASQUE/Basque_INTRs_transformers.csv'
    #out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/BASQUE/Basque_INTRs_transformers_res.csv'
    #stim_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/Plos_stimuli_test_transformers.csv'
    #out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/Plos_stimuli_test_transformers_electra_MLM_results.csv'
    #stim_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/Plos_stimuli_transf.csv'
    stim_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/input/Plos_stimuli_test_transformers.csv'
    out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/results/Hindi_transformer_roberta_test_results.csv'
    #out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/Plos_stimuli_transf_results.csv'
    t_surp = TransformerSurprisal(modelname, stim_path, out_path, transf='roberta', task='MLM', lang='HI')
    t_surp.get_all_scores_hi_aux()
    