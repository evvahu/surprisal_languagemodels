import os
import re
import gzip
from inltk.inltk import tokenize, identify_language, reset_language_identifying_models, remove_foreign_languages
from inltk.inltk import setup
from tqdm import tqdm
import mmap
setup('hi')
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



if __name__ == "__main__":
    #file_path = '/Users/eva/Documents/Work/Resources/Hindi_OSCAR/test_recognise_Dev.txt'
    #test(file_path)
    
    #file_path = '/Users/eva/Documents/Work/Resources/Hindi_OSCAR/whole_OSCAR_filtered.txt'
    
    out_path = '/Users/eva/Documents/Work/Resources/Hindi_OSCAR/manual_preprocessed_hi_part_1_from_filtered.txt'
    out_out_path = '/Users/eva/Documents/Work/Resources/Hindi_OSCAR/manual_preprocessed_no_empty.txt'
    delete_empy_lines(out_path, out_out_path)
    """
    dir_path = '/Users/eva/Documents/Work/Resources/Hindi_OSCAR/data_files'
    filtered_path = '/Users/eva/Documents/Work/Resources/Hindi_OSCAR/testtesttest.txt'
    #check_length(filtered_path)
    preprocess_directory(dir_path, filtered_path)
    """    

    #wf.write(fmt.format(*line)rew install zlibbrew install pyenvexport LDFLAGS="-L/usr/local/opt/zlib/lib" )