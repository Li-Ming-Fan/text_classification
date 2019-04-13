# -*- coding: utf-8 -*-

import os

from data_set import Dataset
from model_settings import ModelSettings
import model_utils


from Zeras.vocab import Vocab
from Zeras.model_wrapper import ModelWrapper


import argparse

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('sentence-cls')
    parser.add_argument('--mode', choices=['train', 'eval', 'debug', 'predict'],
                        default = 'train', help = 'run mode')
    #
    parser.add_argument('--gpu', type=str, default = '0',
                        help = 'specify gpu device')
    parser.add_argument('--note', type=str, default = 'note_something',
                        help = 'make some useful notes')
    #
    model_related = parser.add_argument_group('model related settings')    
    model_related.add_argument('--base_dir', type=str, default = './',
                               help='base directory for saving results')
    model_related.add_argument('--model_tag', type=str,
                               default = 'cnn', help='model_tag')
    #
    vocab_related = parser.add_argument_group('vocab related settings')
    vocab_related.add_argument('--emb_file', type=str, default = None,
                               help='pretrained embeddings file')
    vocab_related.add_argument('--emb_dim', type=int,
                               default = 64, help='embeddings dim')
    vocab_related.add_argument('--filter_cnt', type=int,
                               default = 2, help='filter tokens')
    vocab_related.add_argument('--tokens_file', type=str,
                               default = './vocab/vocab_tokens.txt',
                               help='tokens file')
    #
    data_related = parser.add_argument_group('data related settings')
    data_related.add_argument('--dir_examples', type=str,
                              default = './data_examples',
                              help = 'dir_examples')
    data_related.add_argument('--data', choices=['train', 'valid', 'test', 'all'],
                              default = 'test', help = 'run mode')
    
    
    
    return parser.parse_args()
    
#
if __name__ == '__main__':
    
    args = parse_args()
    run_mode = args.mode
    #
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #
    # data
    file_data_train = os.path.join(args.dir_examples, "data_examples_train.pkl")
    file_data_valid = os.path.join(args.dir_examples, "data_examples_valid.pkl")
    file_data_test = os.path.join(args.dir_examples, "data_examples_test.pkl")
    file_data_all = os.path.join(args.dir_examples, "data_examples.pkl")
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
    # model
    model_tag = args.model_tag
    #
    if model_tag.startswith('cnn'):
        from model_graph_cnn import ModelGraph
    elif model_tag.startswith('rnn'):
        from model_graph_rnn import ModelGraph
    elif model_tag.startswith('rnf'):
        from model_graph_rnf import ModelGraph
    elif model_tag.startswith('msa'):
        from model_graph_msa import ModelGraph
    elif model_tag.startswith('cap'):
        from model_graph_cap import ModelGraph
    else:
        assert False, "NOT supported model_tag"
    #
    # vocab and settings
    vocab = Vocab()
    vocab.add_tokens_from_file(args.tokens_file)
    vocab.filter_tokens_by_cnt(args.filter_cnt)
    vocab.emb_dim = args.emb_dim
    vocab.load_pretrained_embeddings(args.emb_file)
    #
    settings = ModelSettings(vocab)
    settings.model_tag = model_tag
    settings.model_graph = ModelGraph
    settings.gpu_available = args.gpu
    #    
    if run_mode == 'predict':
        settings.is_train = False
    else:
        settings.is_train = True
    #
    settings.base_dir = args.base_dir
    settings.check_settings()
    settings.create_or_reset_log_file()
    settings.logger.info('running with args : {}'.format(args))
    settings.logger.info(settings.trans_info_to_dict())
    #
    # model
    model = ModelWrapper(settings)
    #
    # run
    #
    if run_mode == 'predict':
        model.prepare_for_prediction()
    else:
        model.prepare_for_train_and_valid()
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
    elif run_mode == 'predict':
        # data
        dataset = Dataset()
        dataset.load_data_examples(file_data_pkl)
        data_predict = dataset.data_examples
        #
        report = model_utils.do_predict(model, data_predict)
        model.logger.info('prediction results: {}'.format(report))
        print('prediction results: {}'.format(report))
        #
    else:
        print('NOT supported mode. supported modes: [train|eval|debug|predict]')
    #
    settings.logger.info("task finished")
    settings.close_logger()
    print("task finished")  
    #
    
    