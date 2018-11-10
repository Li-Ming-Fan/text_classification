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
# 
# data
dataset = Dataset()
dataset.load_processed_data()
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
config.is_train = False
#
config.check_settings()
#
#
model = ModelWrapper(config)
model.prepare_for_prediction()
#

#
batch_size_eval = 10  # config.batch_size_eval
#
test_batches = Dataset.do_batching_data(data_test, batch_size_eval)
test_batches = Dataset.do_standardizing_batches(test_batches, config)
#
posi_end = len(test_batches)  # 10
#

#
for idx in range(0, len(test_batches[0:posi_end]), 1):
    
    print('curr: %d' % idx)
    
    batch = test_batches[idx]
    
    print(batch)
    
    #
    result = model.predict_from_batch(batch)[0]
    #    
    target = batch[-1]
    #
    #print('new news:')
    print(result)
    print(target)
    #
    