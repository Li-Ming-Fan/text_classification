# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 05:41:27 2019

@author: limingfan
"""

import os
import numpy as np

from Zeras.data_batcher import DataBatcher

import data_utils

#
def eval_process(model, eval_batcher, max_batches_eval, mode_eval):
    """
    """
    loss_aver, metric_aver = 0.0, 0.0
    count = 0
    while True:
        batch = eval_batcher.get_next_batch()  
        #
        if batch is None: break
        if count == max_batches_eval: continue  #
        #
        count += 1
        # print(count)
        # print(len(batch["input_x"]))
        # if len(batch) == 0: continue
        if len(batch["input_x"]) != model.settings.batch_size and model.num_gpu > 1:
            break
        #
        result_dict = model.run_eval_one_batch(batch)
        loss = result_dict["loss_optim"]
        metric = result_dict["metric"]
        loss_aver += loss
        metric_aver += metric
        # print(loss)
        # print(metric)
        #
        if mode_eval:
            print(count)
            print("batch data:")
            print(batch["input_y"])
            #
            print("results:")
            print(np.argmax(result_dict["logits"], -1) )
            print()
        #
    #
    loss_aver /= count
    metric_aver /= count
    # metric_aver = 100 - loss_aver
    #
    model.settings.logger.info('eval finished, with total num_batches: %d' % count)
    # model.settings.logger.info('loss_aver, metric_aver: %g, %g' % (loss_aver, metric_aver))
    #
    eval_score = {}
    return eval_score, loss_aver, metric_aver
    #
    
def do_eval(settings, args):
    #
    if args.ckpt_loading == "latest":
        dir_ckpt = settings.model_dir
    else:
        dir_ckpt = settings.model_dir_best
    #
    # model
    model = settings.ModelClass(settings)
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
    eval_score, loss_aver, metric_aver = eval_process(model, data_batcher,
                                                      settings.max_batches_eval,
                                                      mode_eval = True)
    #
    print('loss_aver, metric_aver: %g, %g' % (loss_aver, metric_aver))
    settings.logger.info('loss_aver, metric_aver: %g, %g' % (loss_aver, metric_aver))
    settings.logger.info('{}'.format(eval_score))
    #
    
#
def do_train_and_valid(settings, args):
    #
    if args.ckpt_loading == "latest":
        dir_ckpt = settings.model_dir
    else:
        dir_ckpt = settings.model_dir_best
    #
    # model
    model = settings.ModelClass(settings)
    model.prepare_for_train_and_valid(dir_ckpt)
    #    
    # data
    file_raw = os.path.join(args.dir_examples, "data_examples_train.txt")
    data_raw = data_utils.load_from_file_raw(file_raw)
    #
    batch_stder = lambda x: data_utils.get_batch_std(x, settings)
    data_batcher = DataBatcher(data_raw, batch_stder, settings.batch_size,
                               single_pass = False)
    #
    eval_period = settings.valid_period_batch
    file_raw_eval = os.path.join(args.dir_examples, "data_examples_valid.txt")
    data_raw_eval = data_utils.load_from_file_raw(file_raw_eval)
    #
    # train
    loss = 10000.0
    best_metric_val = 0
    # last_improved = 0
    lr = 0.0
    #
    count = 0
    model.settings.logger.info("")
    while True:
        #
        # eval
        if count % eval_period == 0:            
            settings.logger.info("training curr batch, loss, lr: %d, %g, %g" % (count, loss, lr) )
            #
            model.save_ckpt(settings.model_dir, settings.model_name, count)
            model.assign_dropout_keep_prob(1.0)
            #
            settings.logger.info('evaluating after num_batches: %d' % count)
            eval_batcher = DataBatcher(data_raw_eval, batch_stder, settings.batch_size,
                                       single_pass = True)
            #
            eval_score, loss_aver, metric_val = eval_process(model, eval_batcher,
                                                             settings.max_batches_eval,
                                                             mode_eval = False)
            settings.logger.info("eval loss_aver, metric, metric_best: %g, %g, %g" % (
                    loss_aver, metric_val, best_metric_val) )
            #
            # save best
            if metric_val >= best_metric_val:  # >=
                best_metric_val = metric_val
                # last_improved = count
                # ckpt
                settings.logger.info('a new best model, saving ...')
                model.save_ckpt_best(settings.model_dir_best, settings.model_name, count)
                #
            #
            if lr < settings.learning_rate_minimum and count > settings.warmup_steps:
                settings.logger.info('current learning_rate < learning_rate_minimum, stop training')
                break
            #
            model.assign_dropout_keep_prob(settings.keep_prob)
            settings.logger.info("")
            #
        #
        # train
        batch = data_batcher.get_next_batch()  
        # if batch is None: break
        count += 1
        # print(count)        
        #
        result_dict = model.run_train_one_batch(batch)   # just for train
        loss = result_dict["loss_optim"]
        lr = result_dict["lr"]
        #
        # print(loss)
        # model.logger.info("training curr batch, loss, lr: %d, %g, %g" % (count, loss, lr)
        #
    #
    settings.logger.info("training finshed with total num_batches: %d" % count)
    #
    
#  
def do_predict(settings, args):
    #
    if args.ckpt_loading == "latest":
        dir_ckpt = settings.model_dir
    else:
        dir_ckpt = settings.model_dir_best
    #
    pb_file = os.path.join(dir_ckpt, "model_frozen.pb")
    #
    # model
    model = settings.ModelClass(settings)
    model.prepare_for_prediction_with_pb(pb_file)
    #
    # data
    if args.data == "test":
        file_raw = os.path.join(args.dir_examples, "data_examples_test.txt")
    elif args.data == "train":
        file_raw = os.path.join(args.dir_examples, "data_examples_train.txt")
    elif args.data == "valid":
        file_raw = os.path.join(args.dir_examples, "data_examples_valid.txt")
    #
    data_raw = data_utils.load_from_file_raw(file_raw)
    #
    batch_stder = lambda x: data_utils.get_batch_std(x, settings)
    data_batcher = DataBatcher(data_raw, batch_stder, settings.batch_size_eval,
                               single_pass=True)
    #
    # predict
    count = 0
    while True:
        batch = data_batcher.get_next_batch()  
        #
        if batch is None: break
        if count == settings.max_batches_eval: continue  #
        #
        count += 1
        print(count)
        #
        print("batch data:")
        print(batch["input_y"])
        print("batch data end")
        #
        results = model.predict_with_pb_from_batch(batch)["logits"]
        #
        print("results:")
        print(np.argmax(results[0], -1) )
        print("results end")
        print()
        #
    #
    settings.logger.info('prediction finished, with total num_batches: %d' % count)
    #
    
def do_convert(settings, args):
    #
    if args.ckpt_loading == "latest":
        dir_ckpt = settings.model_dir
    else:
        dir_ckpt = settings.model_dir_best
    #
    # pb_file = os.path.join(dir_ckpt, "model_saved.pb")
    #
    # model
    model = settings.ModelClass(settings)    
    settings.ModelClass.load_ckpt_and_save_pb_file(model, dir_ckpt)
    #
    