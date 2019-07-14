# -*- coding: utf-8 -*-


import os
import numpy as np
import pickle

from Zeras.vocab import Vocab

from Zeras.data_batcher import DataBatcher
from Zeras.model_wrapper import ModelWrapper

from model_settings import ModelSettings
import data_utils


def do_debug(settings, args):
    #
    if args.ckpt_loading == "latest":
        dir_ckpt = settings.model_dir
    else:
        dir_ckpt = settings.model_dir_best
    #
    # model
    model = ModelWrapper(settings, settings.model_graph)
    model.prepare_for_train_and_valid(dir_ckpt)
    model.assign_dropout_keep_prob(1.0)
    #
    # data
    file_raw = os.path.join(args.dir_examples, "data_examples_test.txt")
    data_raw = data_utils.load_from_file_raw(file_raw)
    #
    batch_stder = lambda x: data_utils.get_batch_std(x, settings)
    data_batcher = DataBatcher(data_raw, batch_stder, settings.batch_size_eval,
                               single_pass=True)
    #
    # eval
    list_batches_result = []
    #
    loss_aver, metric_aver = 0.0, 0.0
    count = 0
    while True:
        batch = data_batcher.get_next_batch()  
        #
        if batch is None: break
        if count == 1000000: continue  #
        #
        count += 1
        # print(count)
        #
        results, loss, metric = model.run_eval_one_batch(batch)
        loss_aver += loss
        metric_aver += metric
        # print(loss)
        # print(metric)
        #
        print(count)
        print("batch data:")
        print(batch[-1])
        #
        print("results:")
        print(np.argmax(results[0], -1) )
        print()
        #
        item = batch[0], batch[1], np.argmax(results[0], -1)
        list_batches_result.append(item)
        #
    #
    dir_result = "data_check_result"
    if not os.path.exists(dir_result): os.mkdir(dir_result)
    #
    file_path = os.path.join(dir_result, "list_batches_result_%d.pkl" % settings.batch_size_eval)
    #
    with open(file_path, 'wb') as fp:
        pickle.dump(list_batches_result, fp)
    #
    loss_aver /= count
    metric_aver /= count
    #
    print('loss_aver, metric_aver: %g, %g' % (loss_aver, metric_aver))
    model.logger.info('loss_aver, metric_aver: %g, %g' % (loss_aver, metric_aver))
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
    settings.batch_size_eval = 32
    print("processing with batch_size_eval: %d ..." % settings.batch_size_eval)
    do_debug(settings, args)
    #
    settings.batch_size_eval = 1
    print("processing with batch_size_eval: %d ..." % settings.batch_size_eval)
    do_debug(settings, args)
    #
    #
    settings.logger.info("task finished")
    settings.close_logger()
    print("task finished")
    #
    