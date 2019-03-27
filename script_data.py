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

from data_set import Dataset
import data_utils

    
if __name__ == '__main__':
    
    list_files = ['./data_raw/data_raw.txt']

    #
    dir_base = './'
    dir_examples = './data_examples'
    
    if not os.path.exists(dir_base): os.mkdir(dir_base)
    if not os.path.exists(dir_examples): os.mkdir(dir_examples)
    
    dir_vocab = './vocab'
    vocab_filter_cnt = 2
    # emb_dim = 64
    
    #
    num_classes = 2
    label_posi = 1
    #
    
    #
    dataset = Dataset(dir_examples, dir_vocab)
    dataset.vocab_filter_cnt = vocab_filter_cnt
    # dataset.emb_dim = emb_dim    
    #
    dataset.list_files = list_files
    #
    
    #
    # prepare
    dataset.prepare_data_examples(load_vocab = False)
    #
    dataset.save_vocab_tokens()             # save or NOT
    dataset.save_data_examples()            # save or NOT
    #
    print('prepared')
    #
    # test load
    dataset.load_vocab_tokens()  
    dataset.load_data_examples()
    print('loaded')
    #
    
    #
    # split
    data_examples = dataset.data_examples
    random.shuffle(data_examples)
    #
    data_train, data_test = Dataset.split_train_and_test(data_examples,
                                                         ratio_split = 0.9)
    data_train, data_valid = Dataset.split_train_and_test(data_train,
                                                          ratio_split = 0.9)
    #
    print()
    print('num_train: %d' % len(data_train))
    print('num_valid: %d' % len(data_valid))
    print('num_test: %d' % len(data_test))
    print('num_all: %d' % len(data_examples))
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
    c_all = data_utils.do_data_statistics(data_examples, label_posi, num_classes)
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
    file_path = os.path.join(dir_examples, 'data_examples_train.pkl')
    dataset.data_examples = data_train
    dataset.save_data_examples(file_path)          # save or NOT
    #
    file_path = os.path.join(dir_examples, 'data_examples_valid.pkl')
    dataset.data_examples = data_valid
    dataset.save_data_examples(file_path)          # save or NOT
    #
    file_path = os.path.join(dir_examples, 'data_examples_test.pkl')
    dataset.data_examples = data_test
    dataset.save_data_examples(file_path)          # save or NOT
    #
    print('prepared')
    print()
    

    #
    from collections import namedtuple
    Settings = namedtuple('Settings', ['vocab',
                                       'min_seq_len',
                                       'max_seq_len'])
    settings = Settings(dataset.vocab, 5, 1000)
    #
    # test batching
    #
    # train_batches = Dataset.do_batching_data(data_train, 32)
    test_batches = Dataset.do_batching_data(data_valid, 32)
    #
    # train_batches_padded = Dataset.do_standardizing_batches(train_batches, settings)
    test_batches_padded = Dataset.do_standardizing_batches(test_batches, settings)
    print('batched')
    #
    # test for prediction
    dataset.load_data_raw()
    data_raw = dataset.data_raw
    #
    num_examples = len(data_raw)
    data_raw = data_raw[0:min(10, num_examples)]
    #
    data_pred = Dataset.preprocess_for_prediction(data_raw, settings)
    print('test for pred')
    #

