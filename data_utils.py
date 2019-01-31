# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 04:11:16 2018

@author: limingfan
"""

import pickle

import jieba as segmentor


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
    seg_list = segmentor.cut(text)
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
    vocab.add_tokens_from_corpus(corp)
    #
    vocab.filter_tokens_by_cnt(filter_cnt)
    #    
    return vocab

#
# statistics
def do_data_statistics(data_examples, label_posi, num_classes):
    
    count = [0] * num_classes
    for example in data_examples:        
        label = example[label_posi]      
        count[label] += 1
        
    return count

#
def segment_sentences(text, delimiters = None):
    """ 
    """
    if delimiters is None:
        delimiters = ['?', '!', ';', '？', '！', '。', '；', '…', '\n']
    #
    text = text.replace('...', '。。。').replace('..', '。。')
    #
    # 引用(“ ”)中有句子的情况
    text = text.replace('"', '').replace('“', '').replace('”', '')
    #
    len_text = len(text)
    
    sep_posi = []
    for item in delimiters:
        posi_start = 0
        while posi_start < len_text:
            try:
                posi = posi_start + text[posi_start:].index(item)  #
                sep_posi.append(posi)
                posi_start = posi + 1               
            except BaseException:
                break # while
        #
    #
    sep_posi.sort()
    num_sep = len(sep_posi)
    #
    
    #
    list_sent = []
    #
    if num_sep == 0: return [ text ]
    #
    posi_last = 0
    for idx in range(0, num_sep - 1):
        posi_curr = sep_posi[idx] + 1
        posi_next = sep_posi[idx + 1]
        if posi_next > posi_curr:
            list_sent.append( text[posi_last:posi_curr] )
            posi_last = posi_curr
    #
    posi_curr = sep_posi[-1] + 1
    if posi_curr == len_text:
        list_sent.append( text[posi_last:] )
    else:
        list_sent.extend( [text[posi_last:posi_curr], text[posi_curr:]] )
    #
    return list_sent

#
def replace_special_symbols(text):
    
    text = text.replace('\u3000', ' ').replace('\u2002', ' ').replace('\u2003', ' ')
    text = text.replace('\xa0', ' ')
    text = text.replace('\n', '').replace('\r', '').replace('\t', '').replace('\v', '')
    
    return text

#
def save_data_to_pkl(data, file_path):
    with open(file_path, 'wb') as fp:
        pickle.dump(data, fp)
        
def load_data_from_pkl(file_path):
    with open(file_path, 'rb') as fp:
        data = pickle.load(fp)
    return data

    
    
#
if __name__ == '__main__':
    
    pass

