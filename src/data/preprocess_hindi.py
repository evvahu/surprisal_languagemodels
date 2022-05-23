import os
import re
import gzip
import json
#from inltk.inltk import tokenize, identify_language, reset_language_identifying_models, remove_foreign_languages
#from inltk.inltk import setup
from tqdm import tqdm
import mmap
#setup('hi')
import sys
import os
import codecs
import unicodedata as ud
import random
# from http://stackoverflow.com/questions/3094498/how-can-i-check-if-a-python-unicode-string-contains-non-western-letters

latin_letters = {}

CHARS = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}
def delete_empy_lines(file_path, out_path):
    with open(out_path, 'w') as wf:
        with open(file_path, 'r') as rf:
            for l in rf:
                l = l.strip()
                if not l: continue
                else:
                    wf.write(l + '\n')

def clean_hindi_sentence(sentence):
    decimal_no_for_first_devnagari_alphabate = int("900", 16)
    decimal_no_for_last_devnagari_alphabate = int("963", 16)
    devnagari_ord_values =  list(range(decimal_no_for_first_devnagari_alphabate, decimal_no_for_last_devnagari_alphabate+1)) 
    tokenised_sents = sentence.split(' ')
    counter = 0
    new_sent = []
    for word in tokenised_sents:
        w = ''
        counter +=1 
        ind_alph = list(word)
        for i in ind_alph:
            if ord(i) in devnagari_ord_values:
                w = w + i
            else:
                w = '<unk>'
        #if counter != sent_length:
        new_sent.append(w)
    return new_sent

def manual_preprocess_zip(f_path, out_path):
    nr_files = 0 
    yes = 0
    no = 0
    #with open(out_path, 'w') as wf:
    print(os.listdir(f_path)) 
    for f in os.listdir(f_path):
        print(f)
        if f.startswith('c4-hi'):

            fp = os.path.join(f_path, f)
            op = os.path.join(out_path, f.replace('json.gz', 'txt'))
            try:
                with gzip.open(fp, 'rt', encoding='utf8') as rf:
                    with open(op, 'w') as wf:
                        for l in rf:
                            jstr = json.loads(l)
                            txt = jstr['text']
                            #print(l)
                            line = clean_hindi_sentence(txt)
                            c = line.count('<unk>')
                            if (c/len(line))*100 < 20:
                                yes +=1
                                line = "".join(txt).replace("<unk>", " ")
                                if len(line) > 1: 
                                    wf.write(line + '\n')
                            else:
                                no +=1  
            except:
                print('false file: {}'.format(fp))
    print(nr_files, yes, no)

def manual_preprocess_dir(f_path, out_path):
    nr_files = 0 
    yes = 0
    no = 0
    with open(out_path, 'w') as wf: 
        for f in os.listdir(f_path):
            dir_path = os.path.join(f_path, f)
            for f in os.listdir(dir_path):
                nr_files +=1 
                new_path = os.path.join(dir_path, f)
                with open(new_path, 'r') as rf:
                        for l in rf:
                        #print(l)
                            line = clean_hindi_sentence(l)
                            c = line.count('<unk>')
                            if (c/len(line))*100 < 50:
                                yes +=1
                                line = "".join(l).replace("<unk>", " ")
                                if len(line) > 1: 
                                    wf.write(line + '\n')
                            else:
                                no +=1  
    print(nr_files, yes, no)
def manual_preprocess(f_path, out_path):
    with open(out_path, 'w') as wf: 
        with open(f_path, 'r') as rf:
                for l in tqdm(rf, total=get_num_lines(f_path)):
                #print(l)
                    line = clean_hindi_sentence(l)
                    c = line.count('<unk>')
                    for w in line:
                        if '<unk>' in w: 
                            c+=1
                    if (c/len(line))*100 < 50:
                        line = "".join(l).replace("<unk>", " ")
                        if len(line) > 1: 
                            wf.write(line + '\n') 

            #print("remove method", line)
def preprocess(dir_path, filtered_path):
    for f in os.listdir(dir_path):
        path = os.path.join(dir_path, f)
        #new_path = os.path.join(filtered_path, f)
        with open(filtered_path, 'w') as wf:
            with gzip.open(path, 'rt') as rf:
                for l in rf:
                    l = l.strip()
                    if not l: continue
                    l = remove_foreign_languages(l, 'hi')
                    c = l.count('<unk>')
                    if c/len(l)*100 < 5:
                        #"".join(l).replace("<unk>", " ").replace('▁', ' ') .split('।')
                        line = "".join(l).replace("<unk>", " ").replace('▁', ' ')
                        #l = [w.strip('_') for w in l]
                        #print(lang)
                        #l = l.split()
                        #lang = identify_language(l)
                        #line = list(filter(lambda w: not re.match(r'[()"]?[a-zA-Z]+[()"]?|[0-9]', w), l))
                        if len(line) > 1: 
                            #fmt = "{} " * (len(line)- 1) + "\n"
                            #wf.write(fmt.format(*line))
                            wf.write(line + '\n')
def get_num_lines(path):
    fp = open(path, "r+")
    buf = mmap.mmap(fp.fileno(),0)
    lines = 0
    while buf.readline():
        lines +=1
    return lines

def preprocess_directory(dir_path, filtered_path):
    with open(filtered_path, 'w') as wf:
        for f in os.listdir(dir_path):
            path = os.path.join(dir_path, f)
            print('currently processing: {}'.format(path))
            with gzip.open(path, 'rt') as rf:
                for l in tqdm(rf, total=get_num_lines(path)):
                    l = l.strip()
                    if not l: continue
                    l = remove_foreign_languages(l, 'hi')
                    c = l.count('<unk>')
                    try:
                        if c/len(l)*100 < 5:
                            line = "".join(l).replace("<unk>", " ").replace('▁', ' ')
                            if len(line) > 1: 
                                wf.write(line + '\n') 
                    except:
                        continue

                
def check_length(dir_path):
    nr_toks = 0
    for f in os.listdir(dir_path):
        path = os.path.join(dir_path, f)
        print(path)
        with open(path, 'r') as rf:
            for l in rf:
                l = l.strip()
                if not l: continue
                line = l.split(" ")
                nr_toks += len(line)
    print(nr_toks)

def check_words(listy):
    words = {'article', 'http', 'tags', 'post', '#','DIRECTV',  }
    not_pr = False
    for l in listy:
        for w in words:
            if w.lower() in l.lower():
                not_pr = True
    return not_pr


def is_latin(uchr):
    try:
        return latin_letters[uchr]
    except KeyError:
        return latin_letters.setdefault(uchr, 'LATIN' in ud.name(uchr))


def only_roman_chars(unistr):
    return all(is_latin(uchr)
               for uchr in unistr
               if uchr.isalpha())  # isalpha suggested by John Machin


def check_chars(listy):
    devn = True
    for w in listy:
        rom = only_roman_chars(w)
        if rom:
            devn=False
    return devn


def second_preprocess(inp, outp):
    for f in os.listdir(inp):
        fp = os.path.join(inp, f)
        op = os.path.join(outp, f)
        print(fp)
        with open(op, 'w') as wf:
            with open(fp, 'r') as rf:
                for l in rf:
                    l = l.strip()
                    line = l.split()
                    if len(line) < 8:
                        continue
                    else:
                        word_check = check_chars(line)
                       
                        if not word_check:
                            continue
                        else:
                            wf.write('{}\n'.format(l))



def choose_random_sents(inp, outp):
    tot_count = 0
    files = os.listdir(inp)
    random.shuffle(files)

    with open(outp, 'w') as wf:
        for f in files:
            fp = os.path.join(inp, f)
            print(fp)
            sents = []
            with open(fp, 'r') as rf:
                for l in rf:
                    l = l.strip()
                    sents.append(l)
            random.shuffle(sents)
            c = 0
            for s in sents:
                wf.write('{}\n'.format(s))
                c += len(s.split())
                tot_count += len(s.split()) 
                if c > 210000:
                    break
            print(tot_count)
            if tot_count > 220000000:
                break
        
            
            

if __name__ == "__main__":
    #file_path = '/Users/eva/Documents/Work/Resources/Hindi_OSCAR/test_recognise_Dev.txt'
    #test(file_path)
    
    #file_path = '/Users/eva/Documents/Work/Resources/Hindi_OSCAR/whole_OSCAR_filtered.txt'
    
    #out_path = '/Users/eva/Documents/Work/Resources/Hindi_OSCAR/manual_preprocessed_hi_part_1_from_filtered.txt'
    #out_out_path = '/Users/eva/Documents/Work/Resources/Hindi_OSCAR/manual_preprocessed_no_empty.txt'
    #delete_empy_lines(out_path, out_out_path)
    """
    dir_path = '/Users/eva/Documents/Work/Resources/Hindi_OSCAR/data_files'
    filtered_path = '/Users/eva/Documents/Work/Resources/Hindi_OSCAR/testtesttest.txt'
    #check_length(filtered_path)
    preprocess_directory(dir_path, filtered_path)
    """    

    #wf.write(fmt.format(*line)rew install zlibbrew install pyenvexport LDFLAGS="-L/usr/local/opt/zlib/lib" )
    #inp = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/c4hindi/input'
      


    #inp = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/c4hindi/output' 

    inp = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/c4hindi/output_filtered'  
    fin = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/HINDI/c4hindi/final.txt'  
    choose_random_sents(inp, fin)

