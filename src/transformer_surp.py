from transformers import AutoTokenizer, AutoModel
import torch


lang2model = {'eu': ''}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class TransformerSurprisal():

    def __init__(self, model_name, bert, stimuli_path, out_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.sents = self.read_stimuli(stimuli_path)
        self.bert = bert #true or false
        self.fullstop = ''
        
    def _generate_sents(self):
        for s in self.sents:
            m_index = self.sents['m_idx']
            whole_sent = []
            mask_tok = '[MASK]'
            if not self.bert:
                s[m_index] = mask_tok
            if self.bert:
                s.append('[CLS]')
            whole_sent += s.strip().split()
            if whole_sent[-1] != self.fullstop:
                whole_sent.append('.')
            if self.bert:
                whole_sent.append('[SEP]')
            text = ' '.join(whole_sent)
            tokenised_text = self.tokenizer.tokenize(text)
            for i,tok in enumerate(tokenised_text):
                if tok == mask_tok: 
                    masked_index = i
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenised_text)
            tokens_tensor = torch.tensor([indexed_tokens]) 
            yield tokens_tensor, masked_index,tokenised_text


    def _get_preds(self):



    def read_stimuli(self, stimuli_path):
        with open(stimuli_path, 'r') as rf:

if __name__ == '__main__':
    path = ''
    outpath = ''
    modelname = 'monsoon-nlp/hindi-bert'
    transi = TransformerSurprisal('hindi', path, outpath)

