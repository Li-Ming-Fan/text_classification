# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 05:41:27 2019

@author: limingfan
"""

from data_set import Dataset

#
def do_eval(model, eval_data):
    #
    batch_size_eval = model.batch_size_eval
    #
    shuffle_seed = Dataset.generate_shuffle_seed()
    data_batches = Dataset.do_batching_data(eval_data, batch_size_eval, shuffle_seed)
    data_batches = Dataset.do_standardizing_batches(data_batches, model.settings)
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
    

def do_train_and_valid(model, data_train, data_valid):   
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
        shuffle_seed = Dataset.generate_shuffle_seed()
        train_batches = Dataset.do_batching_data(data_train, model.batch_size, shuffle_seed)
        train_batches = Dataset.do_standardizing_batches(train_batches, model.settings)
        #
        batch_idx_max = len(train_batches)
        for batch_idx in range(batch_idx_max):
            #
            # eval
            if count % eval_period == 0:                
                model.logger.info("epoch: %d" % epoch)
                model.logger.info("training curr batch, loss, lr: %d, %g, %g" % (count, loss, lr) )
                #
                model.save_ckpt(model.model_dir, model.model_name, count)
                model.assign_dropout_keep_prob(1.0)
                #
                model.logger.info('evaluating after num_batches: %d' % count)
                eval_score, loss_aver, metric_val = do_eval(model, data_valid)
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
                model.assign_dropout_keep_prob(model.settings.keep_prob)
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

def do_debug(model, data_debug):
    #
    batch_size_eval = model.batch_size_eval
    #
    shuffle_seed = Dataset.generate_shuffle_seed()
    data_batches = Dataset.do_batching_data(data_debug, batch_size_eval, shuffle_seed)
    data_batches = Dataset.do_standardizing_batches(data_batches, model.settings)
    #
    count_max = len(data_batches)
    for idx in range(count_max):
        batch = data_batches[idx]
        #
        result = model.run_debug_one_batch(batch)
        #
        print(result)
        #
    #

#  
def do_predict(model, data_examples):    
    #
    batch_size_eval = model.batch_size_eval
    #
    shuffle_seed = Dataset.generate_shuffle_seed()
    data_batches = Dataset.do_batching_data(data_examples, batch_size_eval, shuffle_seed)
    data_batches = Dataset.do_standardizing_batches(data_batches, model.settings)
    #
    report = []
    #
    count_max = len(data_batches)
    for idx in range(count_max):
        batch = data_batches[idx]
        #
        result = model.predict_from_batch(batch)
        #
        print(result)
        #
    #
    return report
    #
    