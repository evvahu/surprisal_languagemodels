from transformers import AutoTokenizer, AutoModel, ElectraTokenizer, ElectraForMaskedLM, BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM, ElectraModel, ElectraForPreTraining, AutoModelWithLMHead
import torch
import numpy as np
from transformers import pipeline
device = torch.device('cpu')
import numpy as np

#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class TransformerSurprisal():

    def __init__(self, model_name, stimuli_path, out_path, transf = 'bert',task='MLM', test = True, lang='Hi'):
        self.s_path = stimuli_path
        self.task = task
        self.lang = lang
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
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelWithLMHead.from_pretrained(model_name)
            #self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            #self.model = RobertaForMaskedLM.from_pretrained(model_name)
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
        self.test = test 
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


    def get_all_scores_each_word(self, all_scores=False):
        wf = open(self.out_path, 'w')
        if self.test:
            wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('sent_id', 'cond', 'idx', 'word', 'foi', 'score_f','score_m', 'correct', 'cond_verb', 'len_score'))
        else:
            wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('sent_id', 'cond', 'idx', 'word', 'foi', 'score_f', 'score_m', 'len_score'))
        for stim_dict in self.read_stimuli(self.s_path):
            print('sent id : {}'.format(stim_dict['sent_id']))
            sentence = stim_dict['sent'].replace('***[MASK]***', stim_dict['form']) #+ ' ')
            scores = self._get_preds_for_each_word(sentence)
            verb_de = ''
            if self.test == True and self.lang == 'DE':
                verb_de = sentence.split(' ')[-2]
                print(verb_de)
            i = 0
            for word, sc in scores.items():
                #if word == stim_dict['form']:
                word = word.strip('\"')
                #score_f = np.mean(sc)
                if all_scores:
                    score_f = sc
                    score_m = np.mean(sc)
                else:
                    score_f = sc[0]
                    score_m = np.mean(sc)
                    #score_f = sc[-1]
                this_form = False
                if word == stim_dict['form']:
                    this_form = True
                if self.test:
                    if all_scores:
                        for sc_c in score_f:
                            wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(stim_dict['sent_id'], stim_dict['cond'],i, word, this_form, sc_c, stim_dict['correct'], stim_dict['cond_verb'], len(sc)))
                    else:
                        if self.test and self.lang == 'DE':
                            cond = stim_dict['cond'] + "_" + verb_de.split('_')[0]
                            wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(stim_dict['sent_id'], cond,i, word, this_form, score_f, stim_dict['correct'], stim_dict['cond_verb'], len(sc)))
                        else:
                            wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(stim_dict['sent_id'], stim_dict['cond'],i, word, this_form, score_f, stim_dict['correct'], stim_dict['cond_verb'], len(sc)))
                else:
                    #if self.lang == 'EU':
                    wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(stim_dict['sent_id'], stim_dict['cond'],i, word, this_form, score_f, score_m,  len(sc)))
                    #if self.lang.upper() == 'HI':
                    #wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(stim_dict['sent_id'], stim_dict['cond'],i, word, this_form, score_f, '', len(sc)))
                    #else:
                    #    wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(stim_dict['sent_id'], stim_dict['cond'],i, word, this_form, score_f, stim_dict['cond_verb'], len(sc)))
                i += 1
        wf.close()
    def get_all_scores(self):
        c = 0 
        wf = open(self.out_path, 'w')
        if self.test:
            wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('sent_id','cond', 'score_f','score_m', 'correct_form', 'predicted', 'nr_word_pieces'))
        else:
            wf.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format('sent_id','cond', 'score_f', 'score_m', 'form','nr_word_pieces'))
        for stim_dict in self.read_stimuli(self.s_path):  #sent, wc, w1, w2, w3, s_id, c1, c2
            if self.task == 'MLM':
                score = self._get_preds(stim_dict['sent'], stim_dict['form'])
            else:
                curr_sent = stim_dict['sent'].replace(self.mask_tok, stim_dict['form'])
                score = self._get_discriminator_preds(curr_sent)
            verb_de = stim_dict['sent'].split(' ')[-2]
            print(score)
            """ 
            if len(score) == 4:
                score_f = score[2] #np.mean(score[1:3]) #np.mean(score[1:3])
            elif len(score) == 3:
                score_f = score[1]
            elif len(score) >4:
                #print(score, score[2:5])
                score_f = score[-2] #np.mean(score[2:5])
            else:
                print('wrong score?')
            """
            #if len(score)< 2:
            #    continue
            score_f = score[0]
            score_m = np.mean(score)
            #if score_:
                #score_f = np.mean(score)
            if self.test:
                if self.lang == 'DE':
                    cond = stim_dict['cond'] + "_" + verb_de
                    wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(stim_dict['sent_id'], cond, score_f, score_m,  stim_dict['correct'], stim_dict['form'], len(score)))

                else:
                    wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(stim_dict['sent_id'], stim_dict['cond'], score_f, score_m, stim_dict['correct'], stim_dict['cond_verb'], len(score)))
            else:
                wf.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(stim_dict['sent_id'], stim_dict['cond'], score_f, score_m, stim_dict['form'], len(score))) 
        wf.close()


    def _get_preds_for_each_word(self, sent):
        oov = self.tokenizer.convert_tokens_to_ids('[oov]')
        if self.lang == 'DE':
            sent = sent.replace('.', ' .')
        sent_split = sent.split(' ')
        #print(sent_split)
        scores = dict()
        for i, word in enumerate(sent_split):
            #print(word, sent_split[:i])
            tokens = ['[CLS]'] + self.tokenizer.tokenize(' '.join(sent_split[:i])) #pre target
            target_idx = len(tokens)
            target = [self.mask_tok]
            post = ' '.join(sent_split[i+1:]).strip()
            if not self.test:
                tokens += target
            else:
                tokens += target + self.tokenizer.tokenize(post) + ['[SEP]']
            #print('whole: {}'.format(tokens))
            print(word, self.tokenizer.tokenize(word))
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            target_word = word
            try:
                word_id = self.tokenizer.encode(word.strip(), add_special_tokens=False)#['input_ids']
                if oov in word_id:
                    print('oov in word_id')
                    continue
            except:
                print('skipping',word)
                continue
            if oov in input_ids:
                print('oov in tokens: ', tokens)
            target_ids = self.tokenizer.encode(target_word, add_special_tokens=False)
            tens = torch.LongTensor(input_ids).unsqueeze(0)
            res = self.model(tens).logits[0, target_idx]
            sm = torch.nn.Softmax(dim=0)
            res = sm(res)
            score = res[target_ids]
            #if word not in scores:
            #    scores[word] = [s.item() for s in score]
            #else:
            #    word = word + '_2'
            word_id = word + "_" + str(i)
            scores[word_id] = [s.item() for s in score] 
        return scores
    def _get_preds(self, sent,word):
        #for sent, wc, w1, w2 in self.read_stimuli(self.s.path):
        pre, target, post = sent.split('***')
        tokens = ['[CLS]'] + self.tokenizer.tokenize(pre)
        target_idx = len(tokens)
        target= [self.mask_tok]
        if not self.test:
            tokens += target
        else:
            tokens += target + self.tokenizer.tokenize(post) + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        oov = self.tokenizer.convert_tokens_to_ids('[oov]')
        print(word, self.tokenizer.tokenize(word))
        try:
            word_id = self.tokenizer.encode(word.strip(), add_special_tokens=False)#['input_ids']
            if oov in word_id:
                print('oov in word_id')
                return None
        except:
            print('skipping',word)
            return None 
        #print(word_id, self.tokenizer.tokenize(word))
        tens = torch.LongTensor(input_ids).unsqueeze(0)
        #print(target_idx)
    
        res = self.model(tens).logits[0, target_idx]
        #print(res)
        sm = torch.nn.Softmax(dim=0)
        res = sm(res)
        score = res[word_id]
        #print(self.tokenizer.tokenize(word), word_id, word, score)
         
        return [float(x) for x in score]

   
    def read_stimuli(self, stimuli_path):
        with open(stimuli_path, 'r') as rf:
            next(rf)
            for l in rf:
                l = l.strip()
                line = l.split('\t') 
                if self.test:
                    if self.lang == 'Hi':
                        yield {'sent_id':line[0], 'cond':line[1], 'sent': line[2], 'form': line[3], 'correct': line[4], 'cond_verb': line[5]}
                    else:
                        yield {'sent_id':line[0], 'cond':line[1], 'sent': line[2], 'form': line[3], 'correct': line[4], 'cond_verb': line[5]}
                else:
                    #if self.lang.upper() == 'HI':
                    yield {'sent_id':line[0], 'cond':line[1], 'sent': line[2], 'form': line[3]}
                    #else:
                    #yield {'sent_id':line[0], 'cond':line[1], 'sent': line[2], 'form': line[3], 'cond_verb': line[4]}

    def get_subwords(self):
        with open(self.out_path, 'w') as wf:
            for stim_dict in self.read_stimuli(self.s_path):
                word = stim_dict['form']
                sent = stim_dict['sent']('***<MASK>***','<mask>')
                toksd = self.tokenizer.tokenize(word)
                #wf.write('{}\t{}\n'.format(word, toksd))


if __name__ == '__main__':
    path = ''
    outpath = ''
    #modelname = 'bert-base-german-cased'
    #modelname = 'monsoon-nlp/hindi-tpu-electra'
    #modelname = 'neuralspace-reverie/indic-transformers-te-bert'
    #modelname = 'monsoon-nlp/hindi-bert'
    #modelname = 'ixa-ehu/berteus-base-cased'
    #modelname = 'surajp/RoBERTa-hindi-guj-san'
    #modelname = 'bert-base-multilingual-cased'
    #modelname = 'google/electra-small-discriminator'
    #modelname= 'flax-community/roberta-hindi'
    #transi = TransformerSurprisal('hindi', path, outpath)
    #stim_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/agent_lms/stimuli/Basque_test_transformer.csv'
    #out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/agent_lms/results/Basque_test_transformer.csv'
    #out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/results/dummy.csv'
    #stim_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/BASQUE/Basque_INTRs_transformers.csv'
    #out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/BASQUE/Basque_INTRs_transformers_res.csv'
    #stim_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/writing/stimuli/Hindi_psych_transformer.csv'
    #out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/writing/results/Hindi_psych_results_v1.csv'
    #out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/results/dummy.csv'
    #stim_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/Plos_stimuli_transf.csv'
    #stim_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/BASQUE/input/Basque_INTRs_transformer_aux.csv'
    #out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/BASQUE/results/Basque_aux_results_v2.csv'
    #stim_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/input/Plos_stimuli_test_transformers.csv'
    #out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/results/Hindi_transformer_electra_v2.csv'
    #out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/Plos_stimuli_transf_results.csv'
    #stim_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/GERMAN/stimuli_test.tsv' 
    #out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/GERMAN/results_test_whole_sent.csv'
    #stim_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/GERMAN/German_stimuli.csv'
    #out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/GERMAN/stimuli_results_transf_bert_whole_sent.csv'
    #stim_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/GERMAN/dummy_test.csv'
    #out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/GERMAN/dummy_test_out.csv'
    
    
    #t_surp = TransformerSurprisal(modelname, stim_path, out_path, transf='electra', task='MLM',test=True, lang='Hi')
    #t_surp.get_all_scores()
 
    modelname = 'ixa-ehu/berteus-base-cased'
    #modelname = 'monsoon-nlp/hindi-tpu-electra'
    #modelname = 'bert-base-german-cased'
    #modelname = 'neuralspace-reverie/indic-transformers-hi-bert'
    stim_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/agent_lms/stimuli/Basque_psych_transformer.csv'
    out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/agent_lms/results/dummy_basque.txt'
    t_surp = TransformerSurprisal(modelname, stim_path, out_path, transf='bert', task='MLM',test=False, lang='EU')
    t_surp.get_all_scores()
    """
    for i in range(46,100):
        print(i)
        stim_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/agent_lms/stimuli/German2_psych_transf_nospill.csv'
        out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/agent_lms/results/German_100runs_CR_newdata/German_psych_BERT_{}.csv'.format(str(i))
        t_surp = TransformerSurprisal(modelname, stim_path, out_path, transf='bert', task='MLM',test=False, lang='DE')
        

    
    tokenizer = RobertaTokenizer.from_pretrained(modelname)
    fill_mask = pipeline("fill-mask", modelname, tokenizer=tokenizer)
    for s_dict in t_surp.read_stimuli(stim_path):
        sent = s_dict['sent'].replace('***[MASK]***','<mask>')
        toksd = tokenizer.tokenize(s_dict['form'])
        print(toksd) 
        re = fill_mask(sent, targets=s_dict['form'])
        print(len(re))
    """