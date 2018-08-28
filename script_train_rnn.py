# -*- coding: utf-8 -*-

import os

from data_set import Dataset

from model_settings import ModelSettings
from model_wrapper import ModelWrapper


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#
flag_load_data = True
model_tag = 'rnn'
#

if model_tag == 'rnn':
    from model_graph_rnn import build_graph

    
# 
# data
pretrained_emb_file = None
emb_dim = 64
#
dataset = Dataset()
dataset.pretrained_emb_file = pretrained_emb_file
dataset.emb_dim = emb_dim
#

# data
if flag_load_data:
    dataset.load_processed_data()
else:
    dataset.prepare_processed_data()
#
data_train, data_test = dataset.split_train_and_test()
#

#
config = ModelSettings(dataset.vocab)
config.model_tag = model_tag
config.model_graph = build_graph
config.is_train = True

#
model = ModelWrapper(config)
model.check_and_make()
model.prepare_for_train_and_valid()
#
model.train_and_valid(data_train, data_test)
#
