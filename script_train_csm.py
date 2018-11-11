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
    from model_graph_cnn import build_model_graph
elif model_tag == 'csm':
    from model_graph_csm import build_model_graph
elif model_tag == 'rnn':
    from model_graph_rnn import build_model_graph
elif model_tag == 'rnf':
    from model_graph_rnf import build_model_graph
    
# 
# data
dataset = Dataset()
dataset.load_vocab_tokens_and_emb()
#
train_tfrecords = ["./data_examples/data_train.tfrecord"]
valid_tfrecords = ["./data_examples/data_valid.tfrecord"]
#

# train
#
settings = ModelSettings()
settings.vocab = dataset.vocab
settings.model_tag = model_tag
settings.model_graph_builder = build_model_graph
settings.is_train = True
settings.check_settings()

#
model = ModelWrapper(settings)
model.train_and_valid(train_tfrecords, valid_tfrecords)
#

# eval best
#
settings = ModelSettings()
settings.vocab = dataset.vocab
settings.model_tag = model_tag
settings.model_graph_builder = build_model_graph
settings.is_train = True
#
settings.keep_prob = 1.0
settings.batch_size = settings.batch_size_eval
settings.num_epochs = 1
#
settings.check_settings()

#
model = ModelWrapper(settings)
dir_ckpt = model.model_dir + '_best'
loss, acc = model.evaluate(valid_tfrecords, dir_ckpt, flag_log_info = True)
#

