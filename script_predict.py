# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 04:55:31 2019

@author: limingfan
"""

import os

from model_settings import ModelSettings
from model_wrapper import ModelWrapper

from data_set import Dataset


import argparse


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('predict-sentence-cls')
    parser.add_argument('--gpu', type=str, default = '0',
                        help = 'specify gpu device')
    parser.add_argument('--note', type=str, default = 'note_something',
                        help = 'make some useful notes')

    data_related = parser.add_argument_group('data related settings')
    data_related.add_argument('--data', choices=['train', 'valid', 'test', 'all'],
                              default = 'test', help = 'run mode')
    
    model_related = parser.add_argument_group('model related settings')
    model_related.add_argument('--model', type=str,
                               default = 'cnn', help='model tag')
    
    return parser.parse_args()

#  
def do_predict(vocab, settings, data_pkl_basename):    
    #
    # model
    model = ModelWrapper(settings)
    model.prepare_for_prediction()
    # model.prepare_for_train_and_valid()
    # model.assign_dropout_keep_prob(1.0)    
    #
    # data
    dataset = Dataset()
    dataset.load_data_examples(data_pkl_basename)
    data_examples = dataset.data_examples
    #
    batch_size_eval = model.batch_size_eval
    #
    data_batches = Dataset.do_batching_data(data_examples, batch_size_eval)
    data_batches = Dataset.do_standardizing_batches(data_batches, model.settings)
    #
    count_max = len(data_batches)
    for idx in range(count_max):
        batch = data_batches[idx]
        #
        result = model.predict_from_batch(batch)
        #
        print(result)
        #
    #
    
#
if __name__ == '__main__':
    
    args = parse_args()
    #
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #
    # model
    model_tag = args.model    
    #
    if model_tag.startswith('cnn'):
        from model_graph_cnn import build_graph
    elif model_tag.startswith('rnn'):
        from model_graph_rnn import build_graph
    elif model_tag.startswith('rnf'):
        from model_graph_rnf import build_graph
    elif model_tag.startswith('msa'):
        from model_graph_msa import build_graph
    elif model_tag.startswith('cap'):
        from model_graph_cap import build_graph
    #
    
    # data
    data_tag = args.data
    #
    pkl_basename = "data_examples_test.pkl"
    #
    if data_tag == "train":
        pkl_basename = "data_examples_train.pkl"
    elif data_tag == "valid":
        pkl_basename = "data_examples_valid.pkl"
    elif data_tag == "test":
        pkl_basename = "data_examples_test.pkl"
    elif data_tag == "all":
        pkl_basename = "data_examples.pkl"
    else:
        print("NOT supported data_tag: " % data_tag)
        assert False, "must be train|valid|test|all"
    #
        
    #
    # vocab and settings
    dataset = Dataset()
    dataset.load_vocab_tokens()
    vocab = dataset.vocab
    #
    settings = ModelSettings(vocab)
    settings.model_tag = model_tag
    settings.model_graph = build_graph
    #
    settings.is_train = False
    #
    settings.check_settings()
    #
    settings.create_or_reset_log_file()
    settings.logger.info('running with args : {}'.format(args))
    settings.logger.info(settings.trans_info_to_dict())
    
    #
    # run
    do_predict(vocab, settings, pkl_basename)
    #
    
