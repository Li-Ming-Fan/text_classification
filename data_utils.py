# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 04:11:16 2018

@author: limingfan
"""

import pickle

import jieba

from vocab import Vocab



# task-related
def load_from_file_raw(file_raw):
        
    data_raw = []
    with open(file_raw, 'r', encoding = 'utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0: continue
            #
            str_arr = line.split('<label_text_delimiter>')
            # print(str_arr)
            # print(line)
            label = int(str_arr[0].strip())
            data_raw.append( (str_arr[1], label) )
    #
    return data_raw
    
    """
    import xlrd    
    #
    work_book = xlrd.open_workbook(file_raw)
    data_sheet = work_book.sheets()[0]
    queries = data_sheet.col_values(0)
    labels = data_sheet.col_values(2)
    return queries, labels
    """

#      
def clean_and_seg_single_text(text):        
    text = text.strip()
    #
    seg_list = jieba.cut(text, cut_all = False)
    tokens = list(seg_list)
    #
    #print(text)
    #tokens = list(text)   # cut chars
    #
    return tokens

#
def clean_and_seg_list_raw(data_raw):        
    data_seg = []
    for text, label in data_raw:
        text = replace_special_symbols(text)
        text_tokens = clean_and_seg_single_text(text)
        data_seg.append( (text_tokens, label) )
    return data_seg
#
def convert_data_seg_to_ids(vocab, data_seg):
    data_converted = []
    for item, label in data_seg:
        ids = vocab.convert_tokens_to_ids(item)
        data_converted.append( (ids, label) )
    return data_converted

#
# vocab    
def build_vocab_tokens(data_seg, filter_cnt = 5):
    
    vocab = Vocab()
    corp = []
    
    for tokens, label in data_seg:
        corp.append(tokens)
    #
    vocab.load_tokens_from_corpus(corp)
    #
    vocab.filter_tokens_by_cnt(filter_cnt)
    #    
    return vocab

#
# task-independent
def replace_special_symbols(text):
    
    text = text.replace('\u3000', ' ').replace('\u2002', ' ').replace('\u2003', ' ')
    text = text.replace('\xa0', ' ')
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('\v', ' ')
    
    return text

# task-independent
def save_data_to_pkl(data, file_path):
    with open(file_path, 'wb') as fp:
        pickle.dump(data, fp)
        
def load_data_from_pkl(file_path):
    with open(file_path, 'rb') as fp:
        return pickle.load(fp)
    
    
#
if __name__ == '__main__':
    
    pass

