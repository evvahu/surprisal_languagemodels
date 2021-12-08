from xml.etree import ElementTree
import nltk
import random
import spacy
from tqdm import tqdm 
#nltk.download('punkt')

symbols = {'%', '+', '<', '>', '\\', '/', '::', '*', '{', '}'}
punct = {'.', ',', ':'}
def check_symbols(t, punctuation=False):
    syms = symbols
    if punctuation:
        syms = punct
    for s in syms:
        if s in t:
            return True
    return False

def count_numbers(s):
    n = 0
    for t in s:
        if t.isnumeric():
            n +=1 
    return n

def calculate_non_alpha(s):
    n_a = 0
    for t in s:
        if not t.isalpha():
            if check_symbols(s):
                continue
            n_a +=1
    return n_a/len(s)

def check_tokens(s):
    boo = False
    for t in s:
        if check_symbols(t):
            boo = True
        if ('jpg' in t) or ('<ref>' in t) or (t in symbols) or (t.startswith('{{')) or (t.startswith('[')) or (t.startswith('#') or t.startswith('\'\'') or t.startswith('*')):
            boo = True
    return boo

def extract_text(path, out_path):
    tree = ElementTree.parse(path)
    root = tree.getroot()
    with open(out_path, 'w') as wf:
        for page in root:
            for t in page:
                for l in t:
                    if l.tag.endswith('text'):
                        if l.text:
                            text = l.text.strip()
                            if text:
                                f = filter(text)
                                if len(f) > 1:
                                    for s in f:
                                        if len(s) >= 5:
                                            s_string = ' '.join(s).strip()
                                            if not s_string:
                                                continue
                                            
                                            wf.write(s_string + '\n')
def filter(text):
    fin = []
    s_split = text.split('\n')
    for s in s_split:
        if len(s) <=3:
            continue
        if ((s.startswith('[[')) or (s.startswith('\t')) or (s.startswith('==')) or (s.startswith('<')) or (s.startswith('|')) or (s.startswith('--')) or (len(s) <5)):
            continue
        else:
            tokens = s.split(' ')
            numb = count_numbers(s)
            if numb <= 2:
                n_a = calculate_non_alpha(tokens)
                
                if not check_tokens(tokens):
                    if n_a < 0.5:
                        fin.append(tokens)
    return fin


def preprocess_OSCAR(path, out_path):
    #nlp = spacy.load("es_core_news_sm")
    with open(out_path, 'w') as wf:
        with open(path, 'r') as rf:
            for l in rf:
                l = l.strip()
                if not l:
                    continue
                
                wf.write(l + '\n')
                




if __name__ == '__main__':
    #path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/BASQUE/eu.txt'
    #out_path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/BASQUE/eu_filtered.txt' 
    #preprocess_OSCAR(path, out_path)
    #with open('/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/BASQUE/eu_filtered_all_final.txt', 'w') as wf:
    print('hello')
