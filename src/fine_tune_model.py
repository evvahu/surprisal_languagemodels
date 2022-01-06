import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, ElectraTokenizer, ElectraForMaskedLM, BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM, ElectraModel, ElectraForPreTraining

def read_dict(path_conds):
    d = dict()
    with open(path_conds, 'r') as rf:
        next(rf)
        for l in rf:
            l = l.strip()
            line = l.split('\t')
            d[line[0]] = {'Pm': line[1], 'Am': line[2], 'mt': line[3], 'cf': line[4]}
    return d 


def read_train(path, path_conds, lang='HI'):
    all_sents = dict()
    label_dict = read_dict(path_conds)
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
        self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
        self.model = ElectraForMaskedLM.from_pretrained(model_name).to(self.device)
        self.mask_token = '[MASK]'
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr) 

    def get_loss(self, sentence_dict):
        mtoken = '[MASK]'
        sents = []
        labels = []
        for k,v in sentence_dict.items():
            sent = v[0].replace(self.mask_token, self.tokenizer.mask_token)
            label = sent
            label.replace(self.tokenizer.mask_token, v[2])
            sents.append(sent)
            labels.append(label)
        sp_id = len(labels)
        full = sents + labels

        #sentences = [v[0] for k,v in sentence_dict.items()]
        #idx_mask = [int(v[1]) for k,v in sentence_dict.items()]
        #corr_forms = [v[2] for k,v in sentence_dict.items()]
        #sentences = [utterance.replace(
         #   self.mask_token, self.tokenizer.mask_token).replace('\n','')
         #   for utterance in sentences]
        tokenized_input = self.tokenizer(full, padding='longest')
        input_ids_sent = tokenized_input['input_ids'][:100]
        attention_sent = tokenized_input['attention_mask'][:100]
        input_ids_labs = tokenized_input['input_ids'][100:]
        attention_labs = tokenized_input['attention_mask'][100:]
        mask_ids = []
        for item in input_ids_sent:
            for nr, i in enumerate(item):
                tok = "".join(self.tokenizer.decode(i).split())
                if tok == self.tokenizer.mask_token:
                    mask_ids.append(nr)
        print(mask_ids)

        #sents_tok = tokenized_input[:sp_id]
        #label_tok = tokenized_input[sp_id]
        #print(sents_tok)
        #print(label_tok)
        #input_ids = []
        #labels = []
        #max_l = -100
        #for i, item in enumerate(tokenized_input['input_ids']):
         #   corr_form = corr_forms[i]
          #  corr_form_encoded = self.tokenizer.encode(corr_form)
           # lab = item[:idx_mask[i]] + corr_form_encoded + item[idx_mask[i]:]
            #item = item[:idx_mask[i]] + [self.tokenizer.encode(self.tokenizer.mask_token)[1]] * len(corr_form_encoded) + item[idx_mask[i]:] 
            #input_ids.append(item)
            #labels.append(lab)
            #if len(lab) > max_l:
            #    max_l = len(lab)
            #print(labels, item)
        """
        pad_input_ids = []
        pad_labels = []
        print(tokenized_input['attention_mask'])
        input_ids = torch.tensor(pad_input_ids).to(self.device)
        labels = torch.tensor(pad_labels).to(self.device)
        attention_mask = torch.tensor(
            tokenized_input['attention_mask']).to(self.device)
        loss = self.model(
            input_ids,
            attention_mask=attention_mask, labels=labels)[0]
        return loss
        """




def train(path_data, model_name, path_conds, out_path_model):
    model = BertFineTuning(model_name)
    epoch_num = 20
    train_data = read_train(path_data, path_conds)
    for epoch in range(epoch_num):
        model.model.train()
        model.optimizer.zero_grad()
        loss = model.get_loss(train_data)
        loss.backward()
        model.optimizer.step()
        print('loss:', loss.item())



if __name__ == '__main__':
    path_data = '/Users/eva/Documents/Work/experiments/Agent_first_project/Hindi_training_data_sents.txt'
    path_conds = '/Users/eva/Documents/Work/experiments/Agent_first_project/Hindi_training_data_conds.txt'
    model_name ='monsoon-nlp/hindi-bert' 
    #train(path_data)
    train(path_data, model_name,path_conds, '')


