from transformers import AutoTokenizer, AutoModel, ElectraTokenizer, ElectraForMaskedLM, BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM, ElectraModel, ElectraForPreTraining
import torch
import numpy as np


device = torch.device('cpu')

class SyntaxAwareFineTuner():
    def __init__(self):
        pass

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
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForMaskedLM.from_pretrained(model_name)
            self.mask_tok = '<mask>'
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.mask_tok = '[MASK]'
        self.model.eval()
        self.write = False
        if len(out_path) > 0:
            self.write = True
            
        self.out_path = out_path

    def _get_discriminator_preds(self, sent):
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

    def get_all_scores_eu(self):
        all_scores = dict()
        lt = 0
        if self.write:
            wf = open(self.out_path, 'w')
            wf.write('sent_id\tamb\talign\tscore_correct\tscore_incorr1\tscore_incorrec2\tscore_inccorect3\tlargerthan\n')
        for sent, wc, w1, w2, w3, s_id, c1, c2 in self.read_stimuli_eu(self.s_path):
            scores = self._get_preds(sent, [wc, w1, w2, w3])
            if not scores:
                continue
            if self.write:
                larger_than = False
                if (scores[0] > scores[1]) and (scores[0]>scores[2]):
                    larger_than = True
                    lt +=1
                wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(s_id, c1, c2, scores[0], scores[1], scores[2], scores[3], larger_than))
            all_scores[s_id] = scores

        print('largest: {}'.format(lt))
        return all_scores

    def get_all_scores_hi(self):
        all_scores = dict()
        lt = 0
        if self.write:
            wf = open(self.out_path, 'w')
            wf.write('sent_id\tscore_correct\tscore_incorr1\tscore_incorrec2\ttypec\ttypew1\ttypew2\tlargerthan\n')
        for sent, wc, w1, w2, tc, t1, t2, s_id in self.read_stimuli_hi(self.s_path):  #sent, wc, w1, w2, w3, s_id, c1, c2
            if self.task == 'MLM':
                scores = self._get_preds(sent, [wc, w1, w2])
            else:
                scores = []
                na = float('inf')
                for i, word in enumerate([wc, w1, w2]):
                    print('word', i, word)
                    curr_sent = sent.replace(self.mask_tok, word)
                    score = self._get_discriminator_preds(curr_sent)
                    if score: scores.append(score)
                    else:
                        na = i
                if na == 0:
                    scores = None
            if not scores or len(scores)<1:
                continue
            if self.write:
                larger_than = False
                try:
                    if (scores[0] > scores[1]) and (scores[0]>scores[2]) and (scores[0]> scores[3]):
                        larger_than = True
                        lt +=1
                except:
                    continue
                score_list = scores
                if len(scores) < 3:
                    score_list = scores + list(np.zeros(3-len(score_list)))
                wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(s_id, score_list[0], score_list[1], score_list[2], tc, t1, t2, larger_than))
            all_scores[s_id] = scores
        print('largest: {}'.format(lt))
        return all_scores

    def _get_preds(self, sent,words):
        #for sent, wc, w1, w2 in self.read_stimuli(self.s.path):
        pre, target, post = sent.split('***')
        tokens = ['[CLS]'] + self.tokenizer.tokenize(pre)
        target_idx = len(tokens)
        target= [self.mask_tok]
        tokens += target + self.tokenizer.tokenize(post) + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        oov = self.tokenizer.convert_tokens_to_ids('[oov]')
        try:
            word_ids = self.tokenizer.convert_tokens_to_ids(words)
            if oov in word_ids:
                return None
        except:
            print('skipping',words)
            return None 

        print(word_ids)
        tens = torch.LongTensor(input_ids).unsqueeze(0)
        print(target_idx)
        res = self.model(tens).logits[0, target_idx]
        print(res)
        sm = torch.nn.Softmax(dim=0)
        res = sm(res)
        scores = res[word_ids]
        return [float(x) for x in scores]    

   

        
    def read_stimuli_eu(self, stimuli_path):
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
                yield sent, wc, w1, w2, tc, t1, t2, s_id
#        with open(stimuli_path, 'r') as rf:

if __name__ == '__main__':
    path = ''
    outpath = ''
    modelname = 'monsoon-nlp/hindi-tpu-electra'
    #modelname = 'monsoon-nlp/hindi-bert'
    #modelname = 'ixa-ehu/berteus-base-cased'
    #modelname = 'flax-community/roberta-hindi'
    #modelname = 'bert-base-multilingual-cased'
    #modelname = 'google/electra-small-discriminator'
    #transi = TransformerSurprisal('hindi', path, outpath)
    #stim_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/BASQUE/Basque_INTRs_transformers.csv'
    #out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/BASQUE/Basque_INTRs_transformers_res.csv'
    stim_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/Plos_stimuli_test_transformers.csv'
    out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/Plos_stimuli_test_transformers_electra_MLM_results.csv'
    
    #out_path = ''
    t_surp = TransformerSurprisal(modelname, stim_path, out_path, transf='electra', task='MLM', lang='HI')
    t_surp.get_all_scores_hi()
    
    
    
    pre = 'Hello, my dog'
    post = 'super duper mega cute'
    
    #print(len(out.logits[0]), out.logits, out.logits[0][target_idx], probs[0][target_idx])
    #print(out.discriminator_predictions)
    #input = tok('Hello, my dog is super [MASK].', return_tensors='pt') 
    #label = tok('Hello, my dog is super cute.', return_tensors='pt')["input_ids"]
    #m_i = '[CLS] गोपाल किताब [MASK] है । [SEP]'
    #l_i =  '[CLS] गोपाल किताब बेचता है ।[SEP]' 
    #m_i = 'Hello, my ***[MASK]*** is Eva.'
    #l_i = 'Hello, my name is Eva.'
    #m_i = 'The teachers who went to the the same universities ***[MASK]*** at the school.'



    """
    m_i = 'गोपाल किताब ***[MASK]*** है ।'
    pre, target, post = m_i.split('***')
    tokens = ['[CLS]'] + tok.tokenize(pre)
    target_idx = len(tokens)
    target= ['[MASK]']
    tokens += target + tok.tokenize(post) + ['[SEP]']
    input_ids = tok.convert_tokens_to_ids(tokens)
    w1 = 'बेचता'
    w2 = 'बेचा'
    try:
        word_ids = tok.convert_tokens_to_ids([w1,w2])
    except:
        print('skipping', w1, w2)
    print(word_ids)
    tens = torch.LongTensor(input_ids).unsqueeze(0)
    print(target_idx)
    res = mod(tens).logits[0, target_idx]
    print(res)
    scorse = res[word_ids]
    print(scorse)

    
    input = tok(m_i, return_tensors='pt')
    label = tok(l_i, return_tensors='pt')['input_ids']
    input_tokenised = tok.tokenize(m_i)
    label_tokenised = tok.tokenize(l_i)
    try_ids = input['input_ids']
    print(len(input_tokenised), len(try_ids[0]))
    #print(tok.decode(try_ids[0]))
    idx_mask = float('inf')
    for i,t in enumerate(input_tokenised):
        print(t, i)
        if t == '[MASK]':
            idx_mask = i
    with torch.no_grad():
        outputs = mod(**input, labels=label)
    _, max_idx = torch.max(outputs.logits[0][idx_mask], 0, False)
    print(max_idx.item())
    #print(torch.max(outputs.logits[idx_mask], out=tuple))
    print(tok.decode(max_idx.item()))
    #print(outputs.logits.shape)

    """