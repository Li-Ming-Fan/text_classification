# -*- coding: utf-8 -*-
"""
@author: limingfan

"""

import os

from data_set import Dataset

from model_settings import ModelSettings
from model_wrapper import ModelWrapper


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#
model_tag = 'cnn'
#

if model_tag == 'cnn':
    from model_graph_cnn import build_graph, debug_the_model
elif model_tag == 'csm':
    from model_graph_csm import build_graph, debug_the_model
elif model_tag == 'rnn':
    from model_graph_rnn import build_graph, debug_the_model
elif model_tag == 'rnf':
    from model_graph_rnf import build_graph, debug_the_model
#


    
# 
# data
dataset = Dataset()
dataset.load_processed_data()
#
#
data_examples = dataset.data_converted   
#
data_train, data_test = Dataset.split_train_and_test(data_examples, ratio_split = 0.9)
data_train, data_valid = Dataset.split_train_and_test(data_train, ratio_split = 0.9)
#

print('num_train: %d' % len(data_train[0]))
print('num_valid: %d' % len(data_valid[0]))
print('num_test: %d' % len(data_test[0]))

#
# test
config = ModelSettings()
config.vocab = dataset.vocab
config.model_tag = model_tag
config.model_graph = build_graph
config.is_train = True
#
config.keep_prob = 1.0
#
config.check_settings()
#
#
model = ModelWrapper(config)
model.prepare_for_train_and_valid()
#

#
debug = 0  # 0, 1
#

#
batch_size_eval = config.batch_size_eval
if debug: batch_size_eval = 3
#
test_batches = Dataset.do_batching_data(data_test, batch_size_eval)
test_batches = Dataset.do_standardizing_batches(test_batches, config)
#
if debug:    
    tensor_v = debug_the_model(model, test_batches)
    #
else:
    print('begin evaluation ...')
    loss, acc = model.evaluate(test_batches)
    print('evaluation end.')
    #
    
    

