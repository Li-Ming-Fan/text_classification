# -*- coding: utf-8 -*-

import os

from model_settings import ModelSettings
from model_wrapper import ModelWrapper

from data_set import Dataset


import argparse


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('sentence-cls')
    parser.add_argument('--mode', choices=['train', 'eval'],
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
                               default = 'cnn', help='model tag')
    
    return parser.parse_args()

#
def eval_process(model, data_batches):
    #
    loss_aver = 0.0
    metric_aver = 0.0
    num_examples = 0
    #
    count_max = len(data_batches)
    for idx in range(count_max):
        batch = data_batches[idx]
        #        
        results, loss, metric = model.run_eval_one_batch(batch)
        #
        batch_size = len(batch[0])
        num_examples += batch_size
        #
        loss_aver += (loss * batch_size)
        metric_aver += (metric * batch_size)
        #
    #
    loss_aver /= num_examples
    metric_aver /= num_examples
    #
    eval_score = metric_aver
    #
    return eval_score, loss_aver, metric_aver
    #
    
    
def do_eval(vocab, settings, data_pkl_basename):
    #
    # model
    model = ModelWrapper(settings)
    model.prepare_for_train_and_valid()
    model.assign_dropout_keep_prob(1.0)
    #
    # data
    dataset = Dataset()
    dataset.load_data_examples(data_pkl_basename)
    data_examples = dataset.data_examples
    #
    batch_size_eval = model.batch_size_eval
    #
    data_batches = Dataset.do_batching_data(data_examples, batch_size_eval)
    data_batches = Dataset.do_standardizing_batches(data_batches, model.settings)
    #
    # eval
    eval_score, loss_aver, metric_aver = eval_process(model, data_batches)
    #
    str_info = "eval_score, loss_aver, metric_aver: %g, %g, %g" % (eval_score, loss_aver, metric_aver)
    model.logger.info(str_info)
    #
    

def do_train_and_valid(vocab, settings, args):   
    #
    file_train_basename = "data_examples_train.pkl"
    file_valid_basename = "data_examples_valid.pkl"
    #
    # model
    model = ModelWrapper(settings)
    model.prepare_for_train_and_valid()
    #    
    # data
    dataset = Dataset()
    dataset.load_data_examples(file_train_basename)
    data_train = dataset.data_examples
    #
    dataset = Dataset()
    dataset.load_data_examples(file_valid_basename)
    data_valid = dataset.data_examples
    #

    # train adn valid
    eval_period = model.valid_period_batch
    #
    loss = 10000.0
    best_metric_val = 0
    last_improved = 0
    lr = model.learning_rate_base
    #
    flag_stop = False
    count = 0
    for epoch in range(model.num_epochs):
        #
        train_batches = Dataset.do_batching_data(data_train, model.batch_size)
        train_batches = Dataset.do_standardizing_batches(train_batches, model.settings)
        #
        batch_idx_max = len(train_batches)
        for batch_idx in range(batch_idx_max):
            #
            # eval
            if count % eval_period == 0:
                model.logger.info("training curr batch, loss, lr: %d, %g, %g" % (count, loss, lr) )
                #
                model.save_ckpt(model.model_dir, model.model_name, count)
                model.assign_dropout_keep_prob(1.0)
                #
                model.logger.info('evaluating after num_batches: %d' % count)
                valid_batches = Dataset.do_batching_data(data_valid, model.batch_size_eval)
                valid_batches = Dataset.do_standardizing_batches(valid_batches, model.settings)
                eval_score, loss_aver, metric_val = eval_process(model, valid_batches)
                model.logger.info("eval loss_aver, metric, metric_best: %g, %g, %g" % (
                        loss_aver, metric_val, best_metric_val) )
                #
                # save best
                if metric_val >= best_metric_val:  # >=
                    best_metric_val = metric_val
                    last_improved = count
                    # ckpt
                    model.logger.info('a new best model, saving ...')
                    model.save_ckpt_best(model.model_dir + '_best', model.model_name, count)
                    # pb
                    model.save_graph_pb_file(model.pb_file)
                    #

                # decay
                if count - last_improved >= model.patience_decay:
                    lr *= model.ratio_decay
                    model.assign_learning_rate(lr)
                    last_improved = count
                    model.logger.info('learning_rate decayed after num_batches: %d' % count)
                    model.logger.info('current learning_rate %g' % lr)
                    #
                    
                    # stop
                    if lr < model.learning_rate_minimum:
                        str_info = "learning_rate < learning_rate_minimum, stop optimization"
                        model.logger.info(str_info)
                        #
                        flag_stop = True
                        break # for batch
                        
                #
                model.assign_dropout_keep_prob(settings.keep_prob)
                model.logger.info("")
                #
            #
            # end if eval
            #
            
            # train
            batch = train_batches[batch_idx]
            count += 1      
            #
            loss = model.run_train_one_batch(batch)
            # model.logger.info("training curr batch, loss, lr: %d, %g, %g" % (count, loss, lr) )
            #
        #
        if flag_stop:
            break  # for epoch
        #
    #
    model.logger.info("training finshed with total num_batches: %d" % count)
    #


    
#
if __name__ == '__main__':
    
    args = parse_args()
    #
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #
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
    data_tag = args.data
    #
    pkl_basename = "data_examples_test.pkl"
    #
    if data_tag == "train":
        pkl_basename = "data_examples_train.pkl"
    elif data_tag == "valid":
        pkl_basename = "data_examples_valid.pkl"
    elif data_tag == "test":
        pkl_basename = "data_examples_test.pkl"
    elif data_tag == "all":
        pkl_basename = "data_examples.pkl"
    else:
        print("NOT supported data_tag: " % data_tag)
        assert False, "must be train|valid|test|all"
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
    # run
    run_mode = args.mode
    #
    if run_mode == 'train':
        do_train_and_valid(vocab, settings, args)
    elif run_mode == 'eval':
        do_eval(vocab, settings, pkl_basename)
    else:
        print('NOT supported mode. supported modes: train, and eval')
    # 
    
    