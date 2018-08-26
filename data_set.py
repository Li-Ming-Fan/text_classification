# -*- coding: utf-8 -*-
"""
@author: limingfan

"""

import xlrd

import os
import pickle
import numpy as np

# import random
# random.shuffle(list_ori, random.seed(10))

import jieba

from vocab import Vocab


class Dataset():
    
    def __init__(self, vocab = None):
        
        # vocab
        self.vocab = vocab
        
        # directories for saving results, auto-mk,
        self.dir_vocab = './vocab'
        self.dir_data_converted = './data_converted'
        
        self.vocab_filter_cnt = 5
        self.emb_dim = 64
        self.pretrained_emb_file = None
             
        # train and valid
        self.file_raw_1 = "./data_raw/neg.xls"
        self.file_raw_2 = "./data_raw/pos.xls"        

        self.data_raw_1 = []
        self.data_raw_2 = []
        
        self.data_seg_1 = []
        self.data_seg_2 = []
        
        self.data_converted_1 = []
        self.data_converted_2 = []
        #

    # preprocess, all are staticmethod
    @staticmethod
    def preprocess_wholesuitely(vocab, data_raw):
        """ data_raw: list of texts
            returning: data for deep-model input
        """
        data_seg = Dataset.clean_and_seg_list_raw(data_raw)
        data_converted = Dataset.convert_data_seg_to_ids(vocab, data_seg)
        return data_converted
    
    @staticmethod
    def clean_and_seg_list_raw(data_raw):        
        data_seg = []
        for item in data_raw:
            text_tokens = Dataset.clean_and_seg_single_text(item)
            data_seg.append(text_tokens)
        return data_seg
    
    @staticmethod
    def clean_and_seg_single_text(text):        
        text = text.strip()
        #
        seg_list = jieba.cut(text, cut_all = False)
        text = '<jieba_cut>'.join(seg_list)
        tokens = text.split('<jieba_cut>')
        #print(text)
        #tokens = list(text)   # cut chars
        #
        return tokens
    
    @staticmethod
    def convert_data_seg_to_ids(vocab, data_seg):
        data_converted = []
        for item in data_seg:
            ids = vocab.convert_tokens_to_ids(item)
            data_converted.append(ids)
        return data_converted
    
    # vocab   
    def load_vocab_tokens_and_emb(self, file_tokens = None, file_emb = None):
        """
        """        
        if file_tokens is None:
            file_tokens = os.path.join(self.dir_vocab, 'vocab_tokens.txt')
        if file_emb is None:
            file_emb = os.path.join(self.dir_vocab, 'vocab_emb.txt')

        self.vocab = Vocab()
        self.vocab.load_tokens_from_file(file_tokens)
        self.vocab.load_pretrained_embeddings(file_emb)
    
    #
    def build_vocab_tokens_and_emb(self):
        """ dataset data_seg
        """
        if not os.path.exists(self.dir_vocab): os.mkdir(self.dir_vocab)        
        self.vocab = Vocab()
        
        # data_seg
        self.vocab.load_tokens_from_corpus(self.data_seg_1)  # data_seg
        self.vocab.load_tokens_from_corpus(self.data_seg_2)
        
        # save
        self.vocab.filter_tokens_by_cnt(self.vocab_filter_cnt)
        self.vocab.save_tokens_to_file(os.path.join(self.dir_vocab, 'vocab_tokens.txt'))
        #
        if self.pretrained_emb_file:
            self.vocab.load_pretrained_embeddings(self.pretrained_emb_file)             
        else:
            self.vocab.randomly_init_embeddings(self.emb_dim)
        self.vocab.save_embeddings_to_file(os.path.join(self.dir_vocab, 'vocab_emb.txt'))
    
    # train and valid
    def prepare_processed_data(self, load_vocab = False):
        """ prepare data to train and test
        """
        # load and seg
        print('load data_raw ...')
        self.data_raw_1 = self._load_from_file_raw(self.file_raw_1)
        self.data_raw_2 = self._load_from_file_raw(self.file_raw_2)

        print('cleanse and seg ...')        
        self.data_seg_1 = Dataset.clean_and_seg_list_raw(self.data_raw_1)
        self.data_seg_2 = Dataset.clean_and_seg_list_raw(self.data_raw_2)

        #
        print('load or build vocab ...')
        if load_vocab:
            self.load_vocab_tokens_and_emb()
        else:
            self.build_vocab_tokens_and_emb()
        print('num_tokens in vocab: %d' % self.vocab.size() )
        
        # convert
        print('convert to ids ...')
        self.data_converted_1 = Dataset.convert_data_seg_to_ids(self.vocab, self.data_seg_1)
        self.data_converted_2 = Dataset.convert_data_seg_to_ids(self.vocab, self.data_seg_2)
        #
        print('save data converted ...')
        self._save_data_converted()
        
        print('preparation done.')
        
    def load_processed_data(self):
        """
        """
        print('load vocab tokens and emb ...')
        self.load_vocab_tokens_and_emb()
        self._load_data_converted()
        
    #
    def _save_data_converted(self):
    
        if not os.path.exists(self.dir_data_converted): os.makedirs(self.dir_data_converted)
        
        file_path = os.path.join(self.dir_data_converted, 'data_1.pkl')
        with open(file_path, 'wb') as fp:
            pickle.dump(self.data_converted_1, fp)
            
        file_path = os.path.join(self.dir_data_converted, 'data_2.pkl')
        with open(file_path, 'wb') as fp:
            pickle.dump(self.data_converted_2, fp)
        
    def _load_data_converted(self):
        
        file_path = os.path.join(self.dir_data_converted, 'data_1.pkl')
        with open(file_path, 'rb') as fp:
            self.data_converted_1 = pickle.load(fp)
            
        file_path = os.path.join(self.dir_data_converted, 'data_2.pkl')
        with open(file_path, 'rb') as fp:
            self.data_converted_2 = pickle.load(fp)
    
    #
    def _load_from_file_raw(self, file_raw):
        
        work_book = xlrd.open_workbook(file_raw)
        data_sheet = work_book.sheets()[0]
        text_raw = data_sheet.col_values(0)
        return text_raw
        
        """
        text_raw = []
        with open(file_raw, 'r', encoding = 'utf-8') as fp:
            lines = fp.readlines()
            for line in lines:
                if line.strip() != '':
                    text_raw.append(line)
        #
        return text_raw
        
        #
        work_book = xlrd.open_workbook(file_raw)
        data_sheet = work_book.sheets()[0]
        queries = data_sheet.col_values(0)
        labels = data_sheet.col_values(2)
        return queries, labels
        """
        
    #
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

    @staticmethod
    def do_batching_data(data, batch_size):
        
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
    def do_normalizing_batches(data_batches, min_seq_len = 5):
        """ padding batches
        """
        batches_normed = []
        for batch in data_batches:
            x, y = batch
            x_padded, x_len = Dataset.do_padding_data_converted(x)
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
    
    dataset.prepare_processed_data(load_vocab = False)
     
    data_train, data_valid = dataset.split_train_and_test()
    
    train_batches = dataset.do_batching_data(data_train, 32)
    test_batches = dataset.do_batching_data(data_valid, 32)
    
    train_batches_padded = dataset.do_normalizing_batches(train_batches, min_seq_len = 5)
    test_batches_padded = dataset.do_normalizing_batches(test_batches, min_seq_len = 5)
    
    #
    dataset = Dataset()
    dataset.load_processed_data()
    