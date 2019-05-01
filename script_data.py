# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 21:12:06 2018

@author: limingfan
"""

import os
# import numpy as np
# import copy

import random
# random.shuffle(list_ori, random.seed(10))

import data_utils
from Zeras.vocab import Vocab

"""
build vocabulary

split data_raw

"""
    
if __name__ == '__main__':
    
    #
    list_files = ['./data_raw/data_raw.txt']
    
    #
    dir_examples = "./data_examples" 
    if not os.path.exists(dir_examples): os.mkdir(dir_examples)
    #
    
    #
    dir_vocab = "./vocab" 
    if not os.path.exists(dir_vocab): os.mkdir(dir_vocab)
    #
    vocab_filter_cnt = 2
    vocab_tokens_file = os.path.join(dir_vocab, "vocab_tokens.txt")
    #
    
    #
    num_classes = 2
    label_posi = 1
    #
    
    #
    # data_raw
    data_raw = []
    for file_path in list_files:
        data_curr = data_utils.load_from_file_raw(file_path)
        data_raw.extend(data_curr)
    #
    # data_seg
    data_seg = data_utils.clean_and_seg_list_raw(data_raw)
    #
    # vocab
    vocab = Vocab()
    vocab = data_utils.build_vocab_tokens(data_seg, vocab)
    vocab.filter_tokens_by_cnt(vocab_filter_cnt)
    vocab.save_tokens_to_file(vocab_tokens_file)
    #
    # split
    data_train, data_test = data_utils.split_train_and_test(data_raw,
                                                            ratio_split = 0.9)
    data_train, data_valid = data_utils.split_train_and_test(data_train,
                                                             ratio_split = 0.9)
    #
    print()
    print('num_train: %d' % len(data_train))
    print('num_valid: %d' % len(data_valid))
    print('num_test: %d' % len(data_test))
    print('num_all: %d' % len(data_raw))
    print()
    #
    c_train = data_utils.do_data_statistics(data_train, label_posi, num_classes)
    print('num train: ')
    print(c_train)
    c_valid = data_utils.do_data_statistics(data_valid, label_posi, num_classes)
    print('num valid: ')
    print(c_valid)
    c_test = data_utils.do_data_statistics(data_test, label_posi, num_classes)
    print('num test: ')
    print(c_test)
    c_all = data_utils.do_data_statistics(data_raw, label_posi, num_classes)
    print('num all: ')
    print(c_all)
    print()
    
    #
    print('balancing train data ...')
    # data_train = Dataset.do_balancing_classes(data_train, label_posi, num_classes)
    random.shuffle(data_train)
    #
    c_train = data_utils.do_data_statistics(data_train, label_posi, num_classes)
    print('num train: ')
    print(c_train)
    print()
    #
    # dataset.save_data_examples()          # save or NOT
    #
    file_path = os.path.join(dir_examples, 'data_examples_train.txt')
    data_utils.write_to_file_raw(file_path, data_train)
    #
    file_path = os.path.join(dir_examples, 'data_examples_valid.txt')
    data_utils.write_to_file_raw(file_path, data_valid)
    #
    file_path = os.path.join(dir_examples, 'data_examples_test.txt')
    data_utils.write_to_file_raw(file_path, data_test)
    #
    print('prepared')
    print()
    
    #
    # test batching
    from collections import namedtuple
    Settings = namedtuple('Settings', ['vocab',
                                       'min_seq_len',
                                       'max_seq_len'])
    settings = Settings(vocab, 5, 1000)
    #
    batch_std = data_utils.get_batch_std(data_train[0:3], settings)
    print(batch_std)
    #
    
    
