# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 04:11:16 2018

@author: limingfan
"""

import pickle
import random

import jieba as segmenter

#
def write_to_file_raw(file_path, data_raw):
    """
    """
    with open(file_path, 'w', encoding = 'utf-8') as fp:     
        for item in data_raw:
            fp.write(str(item[1]) + '<label_text_delimiter>' + item[0].strip() + '\n')            

def load_from_file_raw(file_raw):
    """
    """        
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
    seg_list = segmenter.cut(text)
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
def convert_data_seg_to_ids(data_seg, vocab):
    data_converted = []
    for item, label in data_seg:
        ids = vocab.convert_tokens_to_ids(item)
        data_converted.append( (ids, label) )
    return data_converted
#  
def transfer_to_data_examples(data_converted):
    return data_converted

#
# vocab    
def build_vocab_tokens(data_seg, vocab):
    corp = []    
    for tokens, label in data_seg:
        corp.append(tokens)
    #
    vocab.add_tokens_from_corpus(corp)
    #    
    return vocab

# batch standardized
def get_batch_std(data_raw, settings):
    """ data_raw: list of (text, label),
        returning: data_dict for deep-model input
    """
    vocab = settings.vocab
    min_seq_len = 1
    max_seq_len = 1000
    if settings is not None:
        min_seq_len = settings.min_seq_len
        max_seq_len = settings.max_seq_len
    #
    data_seg = clean_and_seg_list_raw(data_raw)
    data_c = convert_data_seg_to_ids(data_seg, vocab)
    data_e = transfer_to_data_examples(data_c)
    #
    if len(data_e) == 0: return []
    #
    x, y = list(zip(*data_e))
    x_std, x_len = standardize_list_seqs(x, min_seq_len, max_seq_len)
    #
    data_dict = {"input_x": x_std, "input_y": y}
    #
    return data_dict

#
# task-agnostic
def standardize_list_seqs(x, min_seq_len=5, max_seq_len=100):
    """ x: list of seqs
    """
    x_padded = []
    x_len = []
    padded_len = max(min_seq_len, max([len(item) for item in x]))
    padded_len = min(max_seq_len, padded_len)
    for item in x:
        l = len(item)
        d = padded_len - l
        item_n = item.copy()
        if d > 0:
            item_n.extend([0] * d)  # pad_id, 0
        elif d < 0:
            item_n = item_n[0:padded_len]
            l = padded_len
        #
        x_padded.append(item_n)
        x_len.append(l)
        
    return x_padded, x_len
    
#
def generate_shuffle_seed(num_max = 10000000):
    return random.randint(1, num_max)

def split_train_and_test(data_examples, ratio_split = 0.9, shuffle_seed = None):
    """
    """           
    num_examples = len(data_examples)
    #
    print('split train and test ...')
    print('num_examples: %d' % num_examples)
    #
    if shuffle_seed is not None:
        random.shuffle(data_examples, random.seed(shuffle_seed))
    #
    num_train = int(num_examples * ratio_split)
    
    train_data = data_examples[0:num_train]
    test_data = data_examples[num_train:]
    
    return (train_data, test_data)

def do_data_statistics(data_examples, label_posi, num_classes):
    """
    """    
    count = [0] * num_classes
    for example in data_examples:        
        label = example[label_posi]      
        count[label] += 1
        
    return count

def do_balancing_classes(data_examples, label_posi, num_classes, num_oversamples = None):
    """
    """            
    data_all = [[] for idx in range(num_classes)]
    for example in data_examples:
        label = example[label_posi]
        # print(label)
        if label >= num_classes:
            print('label >= num_classes when do_balancing_classes()')
        #
        data_all[label].append(example)
        #
    #
    num_list = [len(item) for item in data_all]
    num_max = max(num_list)
    #
    if num_oversamples is None:
        num_oversamples = [num_max] * num_classes
    #
    for idx in range(num_classes):
        while True:
            len_data = len(data_all[idx])
            d = num_oversamples[idx] - len_data
            if d >= len_data:
                data_all[idx].extend(data_all[idx])
            elif d > 0:
                data_all[idx].extend(data_all[idx][0:d])
            else:
                break
            #
        #
        print('oversampled class %d' % idx)
        #
    #
    data_examples = []
    for idx in range(num_classes):
        data_examples.extend(data_all[idx])
    #
    return data_examples
    #
    
def do_batching_data(data_examples, batch_size, shuffle_seed = None):
    """
    """        
    num_all = len(data_examples)
    
    if shuffle_seed is not None:
        random.shuffle(data_examples, random.seed(shuffle_seed))
    
    #
    num_batches = num_all // batch_size
    
    batches = []
    start_id = 0
    end_id = batch_size
    for i in range(num_batches):        
        batches.append( data_examples[start_id:end_id] )
        start_id = end_id
        end_id += batch_size
    
    if num_batches * batch_size < num_all:
        num_batches += 1
        data = data_examples[start_id:]
        #
        # d = batch_size - len(data)
        # data.extend( data_examples[0:d] )
        #
        batches.append( data )
    
    return batches 

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

