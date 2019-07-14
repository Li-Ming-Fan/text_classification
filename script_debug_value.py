# -*- coding: utf-8 -*-


import os
import pickle

from Zeras.vocab import Vocab

from Zeras.model_wrapper import ModelWrapper

from model_settings import ModelSettings


def do_check_intermediate_value(settings, args):
    #
    if args.ckpt_loading == "latest":
        dir_ckpt = settings.model_dir
    else:
        dir_ckpt = settings.model_dir_best
    #
    #
    settings.batch_size_eval = 1    
    #
    # model
    model = ModelWrapper(settings, settings.model_graph)
    model.prepare_for_train_and_valid(dir_ckpt)
    model.assign_dropout_keep_prob(1.0)
    #
    # data
    file_path = "data_check_result/data_diff.pkl"
    #
    with open(file_path, 'rb') as fp:
        data_loaded = pickle.load(fp)
    #
    # debug
    count = 0
    for key in data_loaded.keys():
        #
        count += 1
        #
        data_0, data_1 = data_loaded[key]
        #
        batch_0, p_0 = data_0
        batch_1, p_1 = data_1
        #
        result_0 = model.run_debug_one_batch(batch_0)
        result_1 = model.run_debug_one_batch(batch_1)
        #
        print("result_0:")
        print(result_0[0][0])
        #
        print("result_1:")
        print(result_1[0][0])
        #
        print()
        print("result_0:")
        print(result_0[1][0])
        #
        print("result_1:")
        print(result_1[1][0])
        #
        break
        #
    #



import argparse

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('sentence-cls')
    parser.add_argument('--mode', choices=['train', 'eval', 'convert', 'predict', 'debug'],
                        default = 'debug', help = 'run mode')
    #
    parser.add_argument('--gpu', type=str, default = '0',
                        help = 'specify gpu device')
    parser.add_argument('--note', type=str, default = 'note_something',
                        help = 'make some useful notes')
    #
    parser.add_argument('--ckpt_loading', choices=['best', 'latest'],
                        default = 'best', help='lastest ckpt or best')
    #
    model_related = parser.add_argument_group('model related settings')    
    model_related.add_argument('--model_tag', type=str,
                               default = 'msa', help='model_tag')
    #
    data_related = parser.add_argument_group('data related settings')
    data_related.add_argument('--dir_examples', type=str,
                              default = './data_examples',
                              help = 'dir_examples')
    data_related.add_argument('--data', choices=['train', 'valid', 'test'],
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
    # settings
    settings = ModelSettings()
    settings.gpu_available = args.gpu
    settings.model_tag = model_tag
    #    
    if run_mode == 'predict':
        settings.is_train = False
    else:
        settings.is_train = True
    #
    settings.check_settings()
    settings.create_or_reset_log_file()
    settings.logger.info('running with args : {}'.format(args))
    settings.logger.info(settings.trans_info_to_dict())
    settings.save_to_json_file("./temp_settings.json")
    #
    # vocab
    vocab = Vocab()    
    vocab.add_tokens_from_file(settings.tokens_file)
    vocab.load_pretrained_embeddings(settings.emb_file)
    vocab.emb_dim = settings.emb_dim
    #
    # model & vocab
    settings.model_graph = ModelGraph
    settings.vocab = vocab
    #
    # run
    # do_debug(settings, args)
    #
    do_check_intermediate_value(settings, args)
    #
    #
    settings.logger.info("task finished")
    settings.close_logger()
    print("task finished")
    #
    