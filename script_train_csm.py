# -*- coding: utf-8 -*-

import os

from data_set import Dataset

from model_settings import ModelSettings
from model_wrapper import ModelWrapper


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#
model_tag = 'csm'
#

if model_tag == 'cnn':
    from model_graph_cnn import build_graph
elif model_tag == 'csm':
    from model_graph_csm import build_graph
elif model_tag == 'rnn':
    from model_graph_rnn import build_graph
elif model_tag == 'rnf':
    from model_graph_rnf import build_graph
    
# 
# data
dataset = Dataset()
dataset.load_processed_data()
#
data_train, data_test = Dataset.split_train_and_test(dataset.data_converted)
#

#
settings = ModelSettings()
settings.vocab = dataset.vocab
settings.model_tag = model_tag
settings.model_graph = build_graph
settings.is_train = True
settings.check_settings()

#
model = ModelWrapper(settings)
model.prepare_for_train_and_valid()
#
model.train_and_valid(data_train, data_test)
#
