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

import tensorflow as tf

import data_utils
from vocab import Vocab

"""
interface functions

for prediction,
for vocabulary,

for data_raw,
for data_processed,

for split, batching,
for standardizing,

for tfrecord, batch_iter

#
# data_utils
#


data_raw = data_utils.load_from_file_raw(item)

data_seg = data_utils.clean_and_seg_list_raw(data_raw)
data_c = data_utils.convert_data_seg_to_ids(vocab, data_seg)

self.vocab = data_utils.build_vocab_tokens(self.data_seg, self.vocab_filter_cnt)


data_utils.save_data_to_pkl(self.data_examples, file_path)

self.data_examples = data_utils.load_data_from_pkl(file_path)


"""


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
        data_batches = Dataset.do_batching_data(data_c, len(data_c), False)        
        data_standardized = Dataset.do_standardizing_batches(data_batches, settings)
        return data_standardized[0]
    
    #
    # vocab, task-independent
    def build_vocab_tokens_and_emb(self):
        
        print('build vocab tokens and emb ...')
        # token
        self.vocab = data_utils.build_vocab_tokens(self.data_seg, self.vocab_filter_cnt)
        
        # emb
        if self.pretrained_emb_file:
            self.vocab.load_pretrained_embeddings(self.pretrained_emb_file)             
        else:
            self.vocab.randomly_init_embeddings(self.emb_dim)
           
        # save
        self.save_vocab_tokens_and_emb()

    def load_vocab_tokens_and_emb(self, file_tokens = None, file_emb = None):
        """
        """
        print('load vocab tokens and emb ...')
        
        if file_tokens is None:
            file_tokens = os.path.join(self.dir_vocab, 'vocab_tokens.txt')
        if file_emb is None:
            file_emb = os.path.join(self.dir_vocab, 'vocab_emb.bin')

        self.vocab = Vocab()
        self.vocab.load_tokens_from_file(file_tokens)
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
    # load data_raw
    def load_data_raw(self):
        """ just load to self.data_raw,
            from self.list_files
        """
        print('load data_raw ...')
        for item in self.list_files:
            data_raw = data_utils.load_from_file_raw(item)
            self.data_raw.extend(data_raw)
        # self.data_raw = data_utils.load_from_file_raw(self.list_files[0])
    
    #
    # taks-related
    # load, seg and convert
    def prepare_processed_data(self, load_vocab):
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
    # task-independent
    def save_processed_data(self):
        """
        """
        print('save processed data ...')
        if not os.path.exists(self.dir_data_examples): os.makedirs(self.dir_data_examples)
        
        file_path = os.path.join(self.dir_data_examples, 'data_examples.pkl')
        data_utils.save_data_to_pkl(self.data_examples, file_path)

        
    def load_processed_data(self):
        """
        """
        print('load processed data ...')
        self.load_vocab_tokens_and_emb()
        #
        file_path = os.path.join(self.dir_data_examples, 'data_examples.pkl')
        self.data_examples = data_utils.load_data_from_pkl(file_path)
        
    #
    # task-independent
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
    def do_balancing_classes(data_examples, counts_oversample = None):
        
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
            t_std, _ = Dataset.standardize_list_texts(texts, min_seq_len, max_seq_len)
            #
            batches_normed.append( (t_std, y) )
            
        return batches_normed
    
    @staticmethod
    def standardize_list_texts(x, min_seq_len=5, max_seq_len=100):
        """ 
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
                item_n = item_n[0:max_seq_len]
                l = max_seq_len
            #
            x_padded.append(item_n)
            x_len.append(l)
            
        return x_padded, x_len
    
    #
    # tfrecord
    #
    # task-related
    @staticmethod
    def generate_tfrecords(data_examples, tfrecod_filepath):
    
        with tf.python_io.TFRecordWriter(tfrecod_filepath) as writer:  
            for feature, label in data_examples:
                
                mapped = map(lambda idx: tf.train.Feature(int64_list = tf.train.Int64List(value = [idx])),
                             feature)
                seq_feature = list(mapped)
                
                var_len_dict = {
                        'sequence': tf.train.FeatureList(feature = seq_feature) }                
                fixed_len_dict = {
                        'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label] )) }
                
                #
                example = tf.train.SequenceExample(
                        feature_lists = tf.train.FeatureLists(feature_list = var_len_dict),
                        context = tf.train.Features(feature = fixed_len_dict) )
                writer.write(example.SerializeToString())
    
    @staticmethod
    def single_example_parser(serialized_example):
        
        sequence_features = {"sequence": tf.FixedLenSequenceFeature([], dtype = tf.int64) }
        context_features = {"label": tf.FixedLenFeature([], dtype = tf.int64) }

        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized = serialized_example,
            context_features = context_features,
            sequence_features = sequence_features )
    
        labels = context_parsed['label']
        sequences = sequence_parsed['sequence']
        
        return sequences, labels
    
    #
    # task-independent
    @staticmethod
    def get_batched_data(tfrecord_filenames, single_example_parser,
                         batch_size, padded_shapes,
                         num_epochs = 1, buffer_size = 100000):
        
        dataset = tf.data.TFRecordDataset(tfrecord_filenames) \
            .map(single_example_parser) \
            .repeat(num_epochs) \
            .shuffle(buffer_size) \
            .padded_batch(batch_size, padded_shapes = padded_shapes) \
        
        return dataset.make_one_shot_iterator().get_next()
    
    
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
    dataset.prepare_processed_data(load_vocab = False)
    #
    dataset.save_processed_data()          # save or NOT
    #
    print('prepared')
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
    data_train = Dataset.do_balancing_classes(data_train)
    #
    print('split')
    #
    # train_batches = Dataset.do_batching_data(data_train, 32)
    # test_batches = Dataset.do_batching_data(data_valid, 32)
    #
    
    #
    # from collections import namedtuple
    # Settings = namedtuple('Settings', ['vocab','min_seq_len','max_seq_len'])
    # settings = Settings(dataset.vocab, 5, 1000)
    #
    from model_settings import ModelSettings
    settings = ModelSettings(vocab = dataset.vocab)
    # train_batches_padded = Dataset.do_standardizing_batches(train_batches, settings)
    # test_batches_padded = Dataset.do_standardizing_batches(test_batches, settings)
    # print('batched')
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
    # test
    dataset.load_processed_data()
    print('test for load')
    #

    #
    print('write to tfrecord ...')
    #
    tfrecord_filename = './data_examples/data_train.tfrecord'
    Dataset.generate_tfrecords(data_train, tfrecord_filename)
    #
    tfrecord_filename = './data_examples/data_valid.tfrecord'
    Dataset.generate_tfrecords(data_valid, tfrecord_filename)
    #
    tfrecord_filename = './data_examples/data_test.tfrecord'
    Dataset.generate_tfrecords(data_test, tfrecord_filename)
    #    
    print('written')
    #

    batch = Dataset.get_batched_data([ tfrecord_filename ],
                                     Dataset.single_example_parser,
                                     batch_size = 2,
                                     padded_shapes = ([None], []),
                                     num_epochs = 2,
                                     buffer_size = 100000)
    """
    #
    def model(features, labels):
        return features, labels
    
    out = model(*batch)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)
        try:
            while not coord.should_stop():
                print(sess.run(out))

        except tf.errors.OutOfRangeError:
            print("done training")
        finally:
            coord.request_stop()
        coord.join(threads)
        
    """
    
    

