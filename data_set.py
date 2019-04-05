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

from vocab import Vocab
from data_utils import load_from_file_raw, clean_and_seg_list_raw
from data_utils import convert_data_seg_to_ids, transfer_to_data_examples
from data_utils import build_vocab_tokens
from data_utils import save_data_to_pkl, load_data_from_pkl


class Dataset():
    
    def __init__(self, dir_examples = None, dir_vocab = None):
        
        # directories for saving results
        self.dir_vocab = dir_vocab
        self.dir_examples = dir_examples
        
        # vocab and examples
        self.vocab = None
        self.data_examples = []             # data_examples
        
        # vocab-related settings
        self.vocab_filter_cnt = 2
        self.emb_dim = 64
             
        # data_raw process
        self.list_files = []
        self.data_raw = []                  # data_raw
        self.data_seg = []                  # data_seg        
        #
        
    #
    # for prediction, task-related
    @staticmethod
    def preprocess_for_prediction(data_raw, settings):
        """ data_raw: list of (text, label),
            returning: data for deep-model input
        """
        vocab = settings.vocab
        
        data_seg = clean_and_seg_list_raw(data_raw)
        data_c = convert_data_seg_to_ids(data_seg, vocab)
        data_e = transfer_to_data_examples(data_c)
        #
        if len(data_e) == 0: return []
        #
        data_batches = Dataset.do_batching_data(data_e, len(data_e), None)        
        data_standardized = Dataset.do_standardizing_batches(data_batches, settings)
        return data_standardized[0]

    #
    # load data_raw
    def load_data_raw(self):
        """ just load to self.data_raw, from self.list_files
        """
        print('load data_raw ...')
        for item in self.list_files:
            data_raw = load_from_file_raw(item)
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
        self.data_seg = clean_and_seg_list_raw(self.data_raw)

        # vocab
        if load_vocab:
            self.load_vocab_tokens()
        else:
            self.build_vocab_tokens()
        print('num_tokens in vocab: %d' % self.vocab.size() )
        #
        # convert
        print('convert to ids ...')
        self.data_converted = convert_data_seg_to_ids(self.data_seg, self.vocab)
        #
        print('transfer to examples ...')
        self.data_examples = transfer_to_data_examples(self.data_converted)
        #        
        print('data_examples prepared')
        
    #
    # data_examples, task-independent
    def save_data_examples(self, file_path = None):
        """
        """
        print('save data_examples ...')
        if not os.path.exists(self.dir_examples): os.makedirs(self.dir_examples)
        
        if file_path is None:
            file_path = os.path.join(self.dir_examples, "data_examples.pkl")            
        #
        save_data_to_pkl(self.data_examples, file_path)
        
    def load_data_examples(self, file_path = None):
        """
        """
        print('load data_examples ...')
        if file_path is None:
            file_path = os.path.join(self.dir_examples, "data_examples.pkl")
        #
        self.data_examples = load_data_from_pkl(file_path)
        
    #
    # vocab, task-independent
    def build_vocab_tokens(self):
        """
        """        
        print('build vocab tokens and randomly initialize emb ...')
        self.vocab = Vocab()
        self.vocab = build_vocab_tokens(self.data_seg, self.vocab)
        self.vocab.filter_tokens_by_cnt(self.vocab_filter_cnt)
        self.vocab.randomly_init_embeddings(self.emb_dim)
        
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
        self.vocab.filter_tokens_by_cnt(self.vocab_filter_cnt)
        self.vocab.randomly_init_embeddings(emb_dim)
    
    def save_vocab_tokens(self, file_tokens = None, file_emb = None):
        """
        """
        print('save vocab tokens and emb ...')
        if not os.path.exists(self.dir_vocab): os.mkdir(self.dir_vocab)
        
        if file_tokens is None:
            file_tokens = os.path.join(self.dir_vocab, 'vocab_tokens.txt')
            
        self.vocab.save_tokens_to_file(file_tokens)
        
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
        data_examples = []
        for idx in range(num_classes):
            data_examples.extend(data_all[idx])
        #
        return data_examples
        #
    
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

#    
if __name__ == '__main__':
    
    pass

