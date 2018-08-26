# -*- coding: utf-8 -*-

import os

from data_set import Dataset

from model_wrap import ModelSettings
from model_wrap import ModelClassification


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#
flag_load_data = True
model_tag = 'cnn'
#

#
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
#
model = ModelClassification(config)
model.prepare_graph_and_sess()
#
model.train_and_valid(data_train, data_test)
#
