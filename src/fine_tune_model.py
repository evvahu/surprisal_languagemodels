import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, ElectraTokenizer, ElectraForMaskedLM, BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM, ElectraModel, ElectraForPreTraining
import random 
import numpy as np


def read_hindi_test(path):

    with open(path, 'r', encoding='utf8') as rf:
        next(rf)
        for l in rf:
            l = l.strip()
            s_id, sent, wc, tc, w1, w2, t1, t2 = l.split('\t') 
            yield sent, wc, w1, w2, tc, t1, t2, s_id

def read_dict(path_conds):
    d = dict()
    with open(path_conds, 'r') as rf:
        next(rf)
        for l in rf:
            l = l.strip()
            line = l.split('\t')
            d[line[0]] = {'Pm': line[1], 'Am': line[2], 'mt': line[3], 'cf': line[4]}
    return d 


def read_data(path, label_dict, lang='HI'):
    all_sents = dict()
    with open(path, 'r') as rf:
        next(rf)
        for l in rf:
            l = l.strip()
            line = l.split('\t')
            sent = ''
            id = line[0]
            for word in line[1:]:
                if word != 'NA':
                    sent += word.strip('\"') + ' '
            all_sents[id] = (sent.strip(), label_dict[id]['mt'], label_dict[id]['cf'])
    return all_sents
            
def prepare_data(sentence_dict, masktoken):
    sents = []
    labels = []
    for k,v in sentence_dict.items():
        #sent = v[0].replace(self.mask_token, self.tokenizer.mask_token)
        label = v[0]
        label = label.replace(masktoken, v[2])
        sents.append(v[0])
        labels.append(label)
    return sents, labels






class VerbClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, label_nr, dropout_rate=0, non_lin=True, function='sigmoid', layers=1):
        super(VerbClassifier, self).__init__()
        dropout_layer = nn.Dropout(dropout_rate)
        hidden_layer = nn.Linear(input_dim, hidden_dim)
        out_layer = nn.Linear(hidden_dim, input_dim)
        self._model = nn.Sequential(dropout_layer, hidden_layer, out_layer)
    

    def forward(self, batch):
        return self._model(batch)


class BertFineTuning():
    def __init__(self, model_name, lr=0.2):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
        self.model = ElectraForMaskedLM.from_pretrained(model_name).to(self.device)
        self.mask_token = '[MASK]'
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr) 

    def get_loss(self, sents, labels):
        #sents = [s.replace(self.mask_token, self.tokenizer.mask_token) for s in sents]
        #full = sents + labels
        #assert len(sents) == len(labels), "sents and labels have to be of the same length"
        #length = len(sents)
        #tokenized_input = self.tokenizer(full, padding='longest')
        #input_ids_sent = tokenized_input['input_ids'][:length]
        #attention_sent = tokenized_input['attention_mask'][:length]
        #input_ids_labs = tokenized_input['input_ids'][length:]
        #attention_labs = tokenized_input['attention_mask'][length:]
        input_ids_sent, attention_sent, input_ids_labs, attention_labs = self._prepare_for_model(sents, labels)
        mask_ids = []
        for item in input_ids_sent:
            for nr, i in enumerate(item):
                tok = "".join(self.tokenizer.decode(i).split())
                if tok == self.tokenizer.mask_token:
                    mask_ids.append(nr)

        input_ids = torch.tensor(input_ids_sent).to(self.device)
        labels = torch.tensor(input_ids_labs).to(self.device)
        attention_mask = torch.tensor(
            attention_sent).to(self.device)
        loss = self.model(
            input_ids,
            attention_mask=attention_mask, labels=labels)[0]
        return loss


    def _prepare_for_model(self, sents, labels):
        sents = [s.replace(self.mask_token, self.tokenizer.mask_token) for s in sents]
        full = sents + labels
        assert len(sents) == len(labels), "sents and labels have to be of the same length"
        length = len(sents)
        tokenized_input = self.tokenizer(full, padding='longest')
        input_ids_sent = tokenized_input['input_ids'][:length]
        attention_sent = tokenized_input['attention_mask'][:length]
        input_ids_labs = tokenized_input['input_ids'][length:]
        attention_labs = tokenized_input['attention_mask'][length:]
        return input_ids_sent, attention_sent, input_ids_labs, attention_labs

def batchify(data, labs, batch_size=30):
    print(type(len(data)), type(batch_size), batch_size)
    size = int(len(data)/int(batch_size))
    batches = []
    for i in range(0, len(data), size):
        sents = data[i:i+size]
        labels = labs[i:i+size]
        batches.append((sents, labels))
    return batches

def train(data_tr, labs_tr, data_val, labs_val, model,  out_path_model, bsize=30):

    epoch_num = 10
    #train_data = read_train(path_data, path_conds)
    batches = batchify(data_tr, labs_tr, bsize)
    val_loss = float('inf')
    patience = 0
    for epoch in range(epoch_num):
        loss_all = []
        for b in batches:
            model.model.train()
            model.optimizer.zero_grad()
            loss = model.get_loss(b[0], b[1])
            loss.backward()
            model.optimizer.step()
            loss_all.append(loss.item())
        val_loss_cur = validate(model,data_val, labs_val, bsize=bsize)
        if val_loss_cur < val_loss:
            val_loss = val_loss_cur
        else:
            patience += 1
        if patience > 1:
            print('training stopped at epoch: {}'.format(epoch))
            break
        print('average loss: {}'.format(np.mean(loss_all)))
    return model

def validate(model, data_val, labs_val, bsize):
    batches = batchify(data_val, labs_val, bsize)
    model.model.eval()
    loss_all = []
    for b in batches:
        loss = model.get_loss(b[0], b[1])
        loss_all.append(loss.item())
    return np.mean(loss_all)

def split_data(sents, labs, nr = 2, ratio= 10):
    shuffled = list(zip(sents, labs))
    random.shuffle(shuffled)
    sents, labs = zip(*shuffled)
    size = int(len(sents) / ratio)
    print(size)
    sents_tr = list(sents[size:])
    labs_tr = list(labs[size:])
    sents_te = list(sents[:size])
    labs_te = list(sents[:size])
    print('in split:', len(labs_tr), type(labs_tr))
    return sents_tr, labs_tr, sents_te, labs_te


def evaluate(data_te, label_te, model):
    model.model.eval()
    batches = batchify(data_te, label_te)
    with torch.no_grad():
        for b in batches:
            input_ids_sent, attention_sent, input_ids_labs, attention_labs = model._prepare_for_model(b[0], b[1])
            input = torch.tensor(input_ids_sent).to(model.device) 
            labs = torch.tensor(input_ids_labs).to(model.device)
            outs = model.model(input, labels = labs)
            print(outs)



def evaluate_external(data, model, out_path):
    model.model.eval()
    mask_token = '***[MASK]***'
    with open(out_path, 'w') as wf:
        for sent, wc, w1, w2, tc, t1, t2, s_id in read_hindi_test(data):
            scores = _get_preds(sent, [wc, w1, w2], model)
            print('score', scores)
            if scores:
                larger = False
                if (scores[0] > scores[1]) and (scores[0] > scores[2]):
                    larger = True
                wf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(s_id, scores[0], scores[1], scores[2], tc, t1, t2, larger))


                    

def _get_preds(sent,words, model):
        #for sent, wc, w1, w2 in self.read_stimuli(self.s.path):
        pre, target, post = sent.split('***')
        tokens = ['[CLS]'] + model.tokenizer.tokenize(pre)
        target_idx = len(tokens)
        target= ['***[MASK]***']
        tokens += target + model.tokenizer.tokenize(post) + ['[SEP]']
        input_ids = model.tokenizer.convert_tokens_to_ids(tokens)
        oov = model.tokenizer.convert_tokens_to_ids('[oov]')
        model.model.eval()
        try:
            word_ids = model.tokenizer.convert_tokens_to_ids(words)
            if oov in word_ids:
                return None
        except:
            print('skipping',words)
            return None 

        print(word_ids)
        tens = torch.LongTensor(input_ids).unsqueeze(0)
        print(target_idx)
        res = model.model(tens).logits[0, target_idx]
        print(res)
        sm = torch.nn.Softmax(dim=0)
        res = sm(res)
        scores = res[word_ids]
        return [float(x) for x in scores]    

if __name__ == '__main__':
    path_data = '/Users/eva/Documents/Work/experiments/Agent_first_project/Hindi_training_data_sents_large.txt'
    path_conds = '/Users/eva/Documents/Work/experiments/Agent_first_project/Hindi_training_data_conds_large.txt'
    model_name ='monsoon-nlp/hindi-bert' 
    #train(path_data)
    model_initial = BertFineTuning(model_name)
    dict_conds = read_dict(path_conds)
    data = read_data(path_data, dict_conds)
    sents, labels = prepare_data(data, '[MASK]')
    data_tr, lab_tr, data_te, lab_te = split_data(sents, labels)

    model_trained = train(data_tr, lab_tr, data_te, lab_te, model_initial,'', 30)
    out_p = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/results/test_trained.csv'
    out_p_2 = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/results/test_untrained.csv'
    data_extr = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/Plos_stimuli_test_transformers.csv'
    evaluate_external(data_extr, model_trained, out_p)
    model_not_trained = BertFineTuning(model_name)
    evaluate_external(data_extr, model_not_trained, out_p_2)


