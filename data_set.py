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
from vocab import Vocab


class Dataset():
    
    def __init__(self, list_files = [], vocab = None):
        
        # vocab
        self.vocab = vocab
        
        # directories for saving results, auto-mk,
        self.dir_vocab = './vocab'
        self.dir_data_examples = './data_examples'
        
        self.vocab_filter_cnt = 5
        self.emb_dim = 64
        self.pretrained_emb_file = None
             
        # train and valid
        #
        # data_raw files
        self.list_files = list_files
        self.data_raw = []                  # data_raw
        self.data_seg = []                  # data_seg
        self.data_examples = []             # data_examples
        #
        
    #
    # for prediction, task-related
    @staticmethod
    def preprocess_for_prediction(data_raw, settings):
        """ data_raw: list of (text, label),
            returning: data for deep-model input
        """
        vocab = settings.vocab
        
        data_seg = data_utils.clean_and_seg_list_raw(data_raw)
        data_c = data_utils.convert_data_seg_to_ids(vocab, data_seg)
        #
        if len(data_c) == 0: return []
        #
        data_batches = Dataset.do_batching_data(data_c, len(data_c), False)        
        data_standardized = Dataset.do_standardizing_batches(data_batches, settings)
        return data_standardized[0]

    #
    # load data_raw
    def load_data_raw(self):
        """ just load to self.data_raw, from self.list_files
        """
        print('load data_raw ...')
        for item in self.list_files:
            data_raw = data_utils.load_from_file_raw(item)
            self.data_raw.extend(data_raw)
        # self.data_raw = data_utils.load_from_file_raw(self.list_files[0])
    
    #
    # load, seg and convert
    def prepare_data_examples(self, load_vocab):
        """ prepare data to train and test
        """
        # load, seg
        self.load_data_raw()

        print('cleanse and seg ...')        
        self.data_seg = data_utils.clean_and_seg_list_raw(self.data_raw)

        # vocab
        if load_vocab:
            self.load_vocab_tokens_and_emb()
        else:
            self.build_vocab_tokens_and_emb()
        print('num_tokens in vocab: %d' % self.vocab.size() )
        #
        # convert
        print('convert to ids ...')
        self.data_examples = data_utils.convert_data_seg_to_ids(self.vocab, self.data_seg)
        #        
        print('preparation done.')
        
    #
    # data_examples, task-independent
    def save_data_examples(self, file_basename = 'data_examples.pkl'):
        """
        """
        print('save data_examples ...')
        if not os.path.exists(self.dir_data_examples): os.makedirs(self.dir_data_examples)
        #
        file_path = os.path.join(self.dir_data_examples, file_basename)
        data_utils.save_data_to_pkl(self.data_examples, file_path)
        
    def load_data_examples(self, file_basename = 'data_examples.pkl'):
        """
        """
        print('load data_examples ...')
        file_path = os.path.join(self.dir_data_examples, file_basename)
        self.data_examples = data_utils.load_data_from_pkl(file_path)
        
    #
    # vocab, task-independent
    def load_vocab_tokens(self, file_tokens = None, emb_dim = None):
        """
        """
        print('load vocab tokens and randomly initialize emb ...')
        
        if file_tokens is None:
            file_tokens = os.path.join(self.dir_vocab, 'vocab_tokens.txt')
        if emb_dim is None:
            emb_dim = self.emb_dim

        self.vocab = Vocab()
        self.vocab.add_tokens_from_file(file_tokens)
        self.vocab.randomly_init_embeddings(emb_dim)
        
    def build_vocab_tokens_and_emb(self):
        
        print('build vocab tokens and emb ...')
        # token
        self.vocab = data_utils.build_vocab_tokens(self.data_seg, self.vocab_filter_cnt)
        
        # emb
        if self.pretrained_emb_file:
            self.vocab.load_pretrained_embeddings(self.pretrained_emb_file)             
        else:
            self.vocab.randomly_init_embeddings(self.emb_dim)
  
    def load_vocab_tokens_and_emb(self, file_tokens = None, file_emb = None):
        """
        """
        print('load vocab tokens and emb ...')
        
        if file_tokens is None:
            file_tokens = os.path.join(self.dir_vocab, 'vocab_tokens.txt')
        if file_emb is None:
            file_emb = os.path.join(self.dir_vocab, 'vocab_emb.bin')

        self.vocab = Vocab()
        self.vocab.add_tokens_from_file(file_tokens)
        self.vocab.load_pretrained_embeddings(file_emb)
    
    def save_vocab_tokens_and_emb(self, file_tokens = None, file_emb = None):
        """
        """
        print('save vocab tokens and emb ...')
        if not os.path.exists(self.dir_vocab): os.mkdir(self.dir_vocab) 
        
        if file_tokens is None:
            file_tokens = os.path.join(self.dir_vocab, 'vocab_tokens.txt')
        if file_emb is None:
            file_emb = os.path.join(self.dir_vocab, 'vocab_emb.bin')
            
        self.vocab.save_tokens_to_file(file_tokens)
        self.vocab.save_embeddings_to_file(file_emb)
        
    #
    # task-independent
    @staticmethod
    def generate_shuffle_seed(num_max = 10000000):
        return random.randint(1, num_max)
    
    @staticmethod
    def split_train_and_test(data_examples, ratio_split = 0.9, shuffle_seed = None):
               
        num_examples = len(data_examples)
        
        print('split train and test ...')
        print('num_examples: %d' % num_examples)
        
        if shuffle_seed is not None:
            random.shuffle(data_examples, random.seed(shuffle_seed))
        
        num_train = int(num_examples * ratio_split)
        
        train_data = data_examples[0:num_train]
        test_data = data_examples[num_train:]
        
        return (train_data, test_data)
    
    @staticmethod
    def do_balancing_classes(data_examples, label_posi, num_classes, num_oversamples = None):
        
        return data_examples
    
    #
    @staticmethod
    def do_batching_data(data_examples, batch_size, shuffle_seed = None):
        
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
    
    # task-related
    @staticmethod
    def do_standardizing_batches(data_batches, settings = None):
        """ padding batches
        """
        min_seq_len = 1
        max_seq_len = 1000
        if settings is not None:
            min_seq_len = settings.min_seq_len
            max_seq_len = settings.max_seq_len
        #
        batches_normed = []
        for batch in data_batches:
            texts, y = zip(*batch)
            #
            t_std, _ = Dataset.standardize_list_seqs(texts, min_seq_len, max_seq_len)
            #
            batches_normed.append( (t_std, y) )
            
        return batches_normed
    
    @staticmethod
    def standardize_list_seqs(x, min_seq_len=5, max_seq_len=100):

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
                item_n = item_n[0:max_seq_len]
                l = max_seq_len
            #
            x_padded.append(item_n)
            x_len.append(l)
            
        return x_padded, x_len

    
    
if __name__ == '__main__':
    
    list_files = ['./data_raw/data_raw.txt']
    
    pretrained_emb_file = None
    # pretrained_emb_file = '../z_data/wv_64.txt'
    
    #
    dataset = Dataset(list_files)
    #
    dataset.pretrained_emb_file = pretrained_emb_file
    dataset.vocab_filter_cnt = 2
    dataset.emb_dim = 64
    
    #
    # prepare
    dataset.prepare_data_examples(load_vocab = False)
    #
    dataset.save_vocab_tokens_and_emb()     # save or NOT
    dataset.save_data_examples()            # save or NOT
    #
    print('prepared')
    #
    # test load
    dataset.load_vocab_tokens_and_emb()
    dataset.load_vocab_tokens()
    dataset.load_data_examples()
    print('test load')
    #
    
    #
    # split
    data_examples = dataset.data_examples
    #
    data_train, data_test = Dataset.split_train_and_test(data_examples,
                                                         ratio_split = 0.9)
    data_train, data_valid = Dataset.split_train_and_test(data_train,
                                                          ratio_split = 0.9)
    #
    data_train = Dataset.do_balancing_classes(data_train, 1, 2)
    #
    print('split')
    #
    file_path = os.path.join(dataset.dir_data_examples, 'examples_train.pkl')
    data_utils.save_data_to_pkl(data_train, file_path)
    
    file_path = os.path.join(dataset.dir_data_examples, 'examples_valid.pkl')
    data_utils.save_data_to_pkl(data_valid, file_path)
    
    file_path = os.path.join(dataset.dir_data_examples, 'examples_test.pkl')
    data_utils.save_data_to_pkl(data_test, file_path)

    #
    # from collections import namedtuple
    # Settings = namedtuple('Settings', ['vocab','min_seq_len','max_seq_len'])
    # settings = Settings(dataset.vocab, 5, 1000)
    #
    from model_settings import ModelSettings
    settings = ModelSettings(vocab = dataset.vocab)
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
    


