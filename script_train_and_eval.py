# -*- coding: utf-8 -*-

import os

from model_settings import ModelSettings
from model_wrapper import ModelWrapper

from data_set import Dataset
import model_utils


import argparse


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('sentence-cls')
    parser.add_argument('--mode', choices=['train', 'eval', 'debug'],
                        default = 'train', help = 'run mode')
    #
    parser.add_argument('--gpu', type=str, default = '0',
                        help = 'specify gpu device')
    parser.add_argument('--note', type=str, default = 'note_something',
                        help = 'make some useful notes')

    data_related = parser.add_argument_group('data related settings')
    data_related.add_argument('--data', choices=['train', 'valid', 'test', 'all'],
                              default = 'test', help = 'run mode')
    
    model_related = parser.add_argument_group('model related settings')
    model_related.add_argument('--model', type=str,
                               default = 'cap', help='model tag')
    
    return parser.parse_args()
    
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
    file_data_train = "./data_examples/data_examples_train.pkl"
    file_data_valid = "./data_examples/data_examples_valid.pkl"
    file_data_test = "./data_examples/data_examples_test.pkl"
    file_data_all = "./data_examples/data_examples_test.pkl"
    #
    data_tag = args.data
    #
    file_data_pkl = file_data_test
    #
    if data_tag == "train":
        file_data_pkl = file_data_train
    elif data_tag == "valid":
        file_data_pkl = file_data_valid
    elif data_tag == "test":
        file_data_pkl = file_data_test
    elif data_tag == "all":
        file_data_pkl = file_data_all
    else:
        print("NOT supported data_tag: " % data_tag)
        assert False, "must be one of [train|valid|test|all]"
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
    settings.is_train = True
    #
    settings.check_settings()
    #
    settings.create_or_reset_log_file()
    settings.logger.info('running with args : {}'.format(args))
    settings.logger.info(settings.trans_info_to_dict())
    #
    # model
    model = ModelWrapper(settings)
    model.prepare_for_train_and_valid()
    #
    
    #
    # run
    run_mode = args.mode
    #
    if run_mode == 'train':
        # data
        dataset = Dataset()
        dataset.load_data_examples(file_data_train)
        data_train = dataset.data_examples
        #
        dataset = Dataset()
        dataset.load_data_examples(file_data_valid)
        data_valid = dataset.data_examples
        #
        model_utils.do_train_and_valid(model, data_train, data_valid)
        #
    elif run_mode == 'eval':
        # data
        dataset = Dataset()
        dataset.load_data_examples(file_data_pkl)
        data_eval = dataset.data_examples
        #
        model.assign_dropout_keep_prob(1.0)
        eval_score, loss_aver, metric_val = model_utils.do_eval(model, data_eval)
        model.logger.info("eval finished with loss_aver, metric: %g, %g" % (loss_aver, metric_val) )
        print("eval finished with loss_aver, metric: %g, %g" % (loss_aver, metric_val) )
        #
    elif run_mode == 'debug':
        # data
        dataset = Dataset()
        dataset.load_data_examples(file_data_pkl)
        data_eval = dataset.data_examples
        #
        model_utils.do_debug(model, data_eval)
        #
    else:
        print('NOT supported mode. supported modes: [train|eval|debug]')
    #
    settings.close_logger()
    #
    
    