# -*- coding: utf-8 -*-
"""
@author: limingfan

"""

import os
import numpy as np

# import random
# random.shuffle(list_ori, random.seed(10))

import data_utils

from vocab import Vocab


"""
for prediction,
for vocab,
for data_preprocessed,

for split,
for batching, standardizing,

"""


class Dataset():
    
    def __init__(self, vocab = None):
        
        # vocab
        self.vocab = vocab
        
        # directories for saving results, auto-mk,
        self.dir_vocab = './vocab'
        self.dir_data_converted = './data_converted'
        
        self.vocab_filter_cnt = 5
        self.emb_dim = 300
        self.pretrained_emb_file = None
             
        # train and valid
        self.file_raw_1 = "./data_raw/neg.xls"
        self.file_raw_2 = "./data_raw/pos.xls"
        #
        self.max_seq_len = 200

        self.data_raw_1 = []
        self.data_raw_2 = []
        
        self.data_seg_1 = []
        self.data_seg_2 = []
        
        self.data_converted_1 = []
        self.data_converted_2 = []
        #

    # preprocess 
    @staticmethod
    def preprocess_for_prediction(data_raw, settings):
        """ data_raw: list of texts
            returning: data for deep-model input
        """
        vocab = settings.vocab
        min_seq_len = settings.min_seq_len
        
        data_seg = data_utils.clean_and_seg_list_raw(data_raw)
        data_converted = data_utils.convert_data_seg_to_ids(vocab, data_seg)
        data_check, _ = Dataset.do_padding_data_converted(data_converted, min_seq_len)              
                
        return [ data_check ]

    # vocab
    def build_vocab_tokens_and_emb(self):
        """ from dataset data_seg
        """
        print('build vocab tokens and emb ...')   
        self.vocab = Vocab()
        
        # data_seg, task-related
        self.vocab.load_tokens_from_corpus(self.data_seg_1)  # data_seg
        self.vocab.load_tokens_from_corpus(self.data_seg_2)
        
        #
        # build, task-independent
        self.vocab.filter_tokens_by_cnt(self.vocab_filter_cnt)
        #
        if self.pretrained_emb_file:
            self.vocab.load_pretrained_embeddings(self.pretrained_emb_file)             
        else:
            self.vocab.randomly_init_embeddings(self.emb_dim)
        
        # save, task-independent
        self.save_vocab_tokens_and_emb()
        #
    
    # task-independent
    def load_vocab_tokens_and_emb(self, file_tokens = None, file_emb = None):
        """
        """
        print('load vocab tokens and emb ...')
        #        
        if file_tokens is None:
            file_tokens = os.path.join(self.dir_vocab, 'vocab_tokens.txt')
        if file_emb is None:
            file_emb = os.path.join(self.dir_vocab, 'vocab_emb.txt')

        self.vocab = Vocab()
        self.vocab.load_tokens_from_file(file_tokens)
        self.vocab.load_pretrained_embeddings(file_emb)
        
    def save_vocab_tokens_and_emb(self, file_tokens = None, file_emb = None):
        """
        """
        print('save vocab tokens and emb ...')
        #        
        if file_tokens is None:
            file_tokens = os.path.join(self.dir_vocab, 'vocab_tokens.txt')
        if file_emb is None:
            file_emb = os.path.join(self.dir_vocab, 'vocab_emb.txt')
        # 
        if not os.path.exists(self.dir_vocab): os.mkdir(self.dir_vocab)
        #
        self.vocab.save_tokens_to_file(os.path.join(self.dir_vocab, 'vocab_tokens.txt'))
        self.vocab.save_embeddings_to_file(os.path.join(self.dir_vocab, 'vocab_emb.txt'))
    
    #
    # data, task-related,
    #
    def load_data_raw(self):
        
        print('load data_raw ...')
        self.data_raw_1 = data_utils.load_from_file_raw(self.file_raw_1)
        self.data_raw_2 = data_utils.load_from_file_raw(self.file_raw_2)        
        
    def prepare_preprocessed_data(self, load_vocab):
        """ prepare data to train and test
        
            NOT batched, NOT standardized
        """
        max_seq_len = self.max_seq_len
        
        # load
        self.load_data_raw()
        
        # clean and seg
        print('cleanse and seg ...')        
        self.data_seg_1 = data_utils.clean_and_seg_list_raw(self.data_raw_1)
        self.data_seg_2 = data_utils.clean_and_seg_list_raw(self.data_raw_2)
        
        # 
        print('truncate data ...')
        self.data_seg_1 = [item[0:min(len(item), max_seq_len)] for item in self.data_seg_1]
        self.data_seg_2 = [item[0:min(len(item), max_seq_len)] for item in self.data_seg_2]

        # vocab
        print('load or build vocab ...')
        if load_vocab:
            self.load_vocab_tokens_and_emb()
        else:
            self.build_vocab_tokens_and_emb()
        print('num_tokens in vocab: %d' % self.vocab.size() )
        
        # convert
        print('convert to ids ...')
        self.data_converted_1 = data_utils.convert_data_seg_to_ids(self.vocab, self.data_seg_1)
        self.data_converted_2 = data_utils.convert_data_seg_to_ids(self.vocab, self.data_seg_2)
        #
        print('save data converted ...')
        if not os.path.exists(self.dir_data_converted): os.mkdir(self.dir_data_converted)
        
        file_path = os.path.join(self.dir_data_converted, 'data_1.pkl')
        data_utils.save_data_to_pkl(self.data_converted_1, file_path)
            
        file_path = os.path.join(self.dir_data_converted, 'data_2.pkl')
        data_utils.save_data_to_pkl(self.data_converted_2, file_path)
        
        print('preparation done.')
        
    def load_preprocessed_data(self):
        """
        """
        self.load_vocab_tokens_and_emb()
        #
        print('load preprocessed data ...')
        #
        file_path = os.path.join(self.dir_data_converted, 'data_1.pkl')
        self.data_converted_1 = data_utils.load_data_from_pkl(file_path)
            
        file_path = os.path.join(self.dir_data_converted, 'data_2.pkl')
        self.data_converted_2 = data_utils.load_data_from_pkl(file_path)
      
    #
    # split, task-related
    def split_train_and_test(self, ratio_train = 0.8, shuffle = False):
        
        num_1 = len(self.data_converted_1)
        num_2 = len(self.data_converted_2)
        
        print('split train and test ...')
        print('num_1: %d' % num_1)
        print('num_2: %d' % num_2)
        
        if shuffle:
            indices = np.random.permutation(np.arange(num_1))
            data_1 = np.array(self.data_converted_1)[indices].tolist()
            
            indices = np.random.permutation(np.arange(num_2))
            data_2 = np.array(self.data_converted_2)[indices].tolist()
        else:
            data_1 = self.data_converted_1
            data_2 = self.data_converted_2
        
        num_train_1 = int(num_1 * ratio_train)
        num_train_2 = int(num_2 * ratio_train)
        
        train_data = (data_1[0:num_train_1], data_2[0:num_train_2])
        test_data = (data_1[num_train_1:], data_2[num_train_2:])
        
        return train_data, test_data
    
    # batching
    @staticmethod
    def do_batching_data(data, batch_size, shuffle = True):
        
        num_1 = len(data[0])
        num_2 = len(data[1])
        num_all = num_1 + num_2
        
        labels = [0] * num_1 + [1] * num_2
        texts = np.concatenate(data, axis = 0)
        
        indices = np.random.permutation(np.arange(num_all))
        texts = np.array(texts)[indices].tolist()
        labels = np.array(labels)[indices].tolist()
        
        num_batches = num_all // batch_size
        
        batches = []
        start_id = 0
        end_id = batch_size
        for i in range(num_batches):            
            batches.append( (texts[start_id:end_id], labels[start_id:end_id]) )
            start_id = end_id
            end_id += batch_size
        
        if num_batches * batch_size < num_all:
            num_batches += 1
            texts_rem = texts[end_id:]
            labels_rem = labels[end_id:]
            #
            d = batch_size - len(labels_rem)
            texts_rem.extend( texts[0:d] )
            labels_rem.extend( labels[0:d] )
            #
            batches.append( (texts_rem, labels_rem) )
        
        return batches 
    
    @staticmethod
    def do_standardizing_batches(data_batches, settings = None):
        """ padding batches
        """
        min_seq_len = 5
        if settings is not None:
            min_seq_len = settings.min_seq_len
        #
        batches_normed = []
        for batch in data_batches:
            x, y = batch
            x_padded, x_len = Dataset.do_padding_data_converted(x, min_seq_len)
            batches_normed.append( (x_padded, y) )
            
        return batches_normed
    
    @staticmethod
    def do_padding_data_converted(list_converted, min_seq_len = 5):
        """ padding list_converted
        """        
        x_padded = []
        x_len = []
        padded_len = max(min_seq_len, max([len(item) for item in list_converted]))
        for item in list_converted:
            l = len(item)
            d = padded_len - l
            item_n = item.copy()
            if d > 0: item_n.extend([0] * d)  # pad_id, 0
            x_padded.append(item_n)
            x_len.append(l)
            
        return x_padded, x_len
    
    
if __name__ == '__main__':   
    
    pretrained_emb_file = None
    
    #
    dataset = Dataset()
    
    dataset.pretrained_emb_file = pretrained_emb_file
    dataset.vocab_filter_cnt = 5
    dataset.emb_dim = 64
    
    dataset.max_seq_len = 200    
    dataset.prepare_preprocessed_data(load_vocab = False)

    #
    data_train, data_valid = dataset.split_train_and_test()

    #
    train_batches = dataset.do_batching_data(data_train, 32)
    test_batches = dataset.do_batching_data(data_valid, 32)
    
    from collections import namedtuple
    Settings = namedtuple('Settings', ['min_seq_len'])
    settings = Settings(5)
    
    train_batches_padded = dataset.do_standardizing_batches(train_batches, settings)
    test_batches_padded = dataset.do_standardizing_batches(test_batches, settings)
    
    #
    dataset = Dataset()
    dataset.load_preprocessed_data()
    
