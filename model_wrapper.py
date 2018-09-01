# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 22:31:20 2018

@author: limingfan

"""

import os
import time
import numpy as np
import json

import tensorflow as tf
from tensorflow.python.framework import graph_util

from model_settings import ModelSettings 
from data_set import Dataset


"""
This class is meant to be task-independent.

"""
      
        
class ModelWrapper(ModelSettings):
    
    def __init__(self, settings):
        
        self.settings = settings
        #
        super().__init__(settings.is_train)
        for key in settings.__dict__.keys():
            # print(key + ': ' + str(self.__dict__[key]) )                   
            self.__dict__[key] = settings.__dict__[key]
            # print(key + ': ' + str(self.__dict__[key]) ) 
            
        # session info
        self.sess_config = tf.ConfigProto()
        self.sess_config.gpu_options.allow_growth = True
        
        
    def _log_info(self, str_info):     
        if not os.path.exists(self.log_dir): os.mkdir(self.log_dir)
        with open(self.log_path, 'a', encoding='utf-8') as fp:
            if len(str_info.strip()) == 0:
                fp.write("\n")
            else:
                fp.write(time.strftime("%Y-%m-%d, %H:%M:%S: ") + str_info + "\n")  
    
    # predict
    def prepare_for_prediction(self, pb_file_path = None):
        #
        if pb_file_path is None: pb_file_path = self.pb_file 
        if not os.path.exists(pb_file_path):
            assert False, 'ERROR: %s NOT exists, when prepare_for_predict()' % pb_file_path
        #
        self._graph = tf.Graph()
        with self._graph.as_default():
            with open(pb_file_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
                #
            #
            self._inputs_predict = []
            for item in self.inputs_predict_name:
                tensor = self._graph.get_tensor_by_name(item)
                self._inputs_predict.append(tensor)
            #
            self._outputs_predict = []
            for item in self.outputs_predict_name:
                tensor = self._graph.get_tensor_by_name(item)
                self._outputs_predict.append(tensor)
            #
            self._inputs_predict_num = len(self._inputs_predict)
            print('Graph loaded for prediction')
            #
        #
        self._sess = tf.Session(graph = self._graph, config = self.sess_config)
        #
    
    def predict(self, inputs_data):
        """
        """
        x_batch = Dataset.preprocess_for_prediction(inputs_data, self.settings) # a single batch
        
        print(x_batch)
        
        feed_dict = self._feed_data_predict(x_batch)
        outputs = self._sess.run(self._outputs_predict, feed_dict = feed_dict)
        
        return outputs

    def _feed_data_predict(self, x_batch):        
        feed_dict = {}
        for idx in range(self._inputs_predict_num):
            feed_dict[self._inputs_predict[idx] ] = x_batch[idx]
        return feed_dict
        
    def _feed_data_train(self, data_batch):        
        feed_dict = {}
        for idx in range(self._inputs_train_num):
            feed_dict[self._inputs_train[idx] ] = data_batch[idx]
        return feed_dict

    def prepare_for_train_and_valid(self):

        # graph
        self._graph = tf.Graph()
        with self._graph.as_default():
            #
            self.model_graph(self.settings)
            #
            # all trainable vars
            self.trainable_vars = tf.trainable_variables()
            # print(self.trainable_vars)
            #            
            self._global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
            self._lr = tf.get_variable("lr", shape=[], dtype=tf.float32, trainable=False)
            #
            if self.use_metric:
                self._metric_tensor = self._graph.get_tensor_by_name(self.metric_name)
            self._loss_tensor = self._graph.get_tensor_by_name(self.loss_name)
            #
            if self.reg_lambda > 0.0:
                loss_reg = tf.add_n([ tf.nn.l2_loss(v) for v in self.trainable_vars \
                                     if 'bias' not in v.name ])
                loss_reg = tf.multiply(loss_reg, self.reg_lambda)
                self._loss_tensor = tf.add(self._loss_tensor, loss_reg)
            #
            # Optimizer
            self._opt = tf.train.AdamOptimizer(learning_rate = self._lr)
            #
            self._train_op = None
            if self.grad_clip > 0.0:
                grads = self._opt.compute_gradients(self._loss_tensor)
                gradients, variables = zip(*grads)
                grads, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
                self._train_op = self._opt.apply_gradients(zip(grads, variables),
                                                           global_step = self._global_step)
            else:
                self._train_op = self._opt.minimize(self._loss_tensor)
                    
            #         
            # save info
            self._saver = tf.train.Saver()
            self._saver_best = tf.train.Saver()
            
            # sess
            self._sess = tf.Session(graph=self._graph, config = self.sess_config)

            # initialize the model
            self._sess.run(tf.global_variables_initializer())
            
            # params count
            self.param_num = sum([np.prod(self._sess.run(tf.shape(v))) \
                                  for v in self.trainable_vars])
            #
            str_info = 'Graph built, there are %d parameters in the model' % self.param_num
            self._log_info(str_info)
            print(str_info)
            #
            info_dict = self.settings.trans_info_to_dict()
            str_info = json.dumps(info_dict, ensure_ascii = False)
            self._log_info(str_info)
            print(str_info)
            #
            self._inputs_train = []
            for item in self.inputs_train_name:
                tensor = self._graph.get_tensor_by_name(item)
                self._inputs_train.append(tensor)
            #
            self._outputs_train = []
            for item in self.outputs_train_name:
                tensor = self._graph.get_tensor_by_name(item)
                self._outputs_train.append(tensor)
            #            
            self._inputs_train_num = len(self._inputs_train)
            #
    #
    def evaluate(self, eval_batches):
        
        data_len = 0        
        total_loss = 0.0
        total_acc = 0.0  # accuracy, metric
        curr = 0
        for data_batch in eval_batches:
            batch_len = len(data_batch)
            data_len += batch_len
            feed_dict = self._feed_data_train(data_batch)
            loss = self._sess.run(self._loss_tensor, feed_dict = feed_dict)
            if self.use_metric:
                metric = self._sess.run(self._metric_tensor, feed_dict = feed_dict)
                total_acc += metric * batch_len
            #
            total_loss += loss * batch_len
            curr += 1
            
            #cls_pred = self.sess.run(self.y_pred_cls, feed_dict = feed_dict)
            #print(cls_pred)
            
        return total_loss / data_len, total_acc / data_len
    
    def train_and_valid(self, train_data, valid_data):
        """ 
        """        
        if not os.path.exists(self.model_dir): os.mkdir(self.model_dir)
        if not os.path.exists(self.model_dir + '_best'): os.mkdir(self.model_dir + '_best')
        
        print('Training and evaluating...')
        #start_time = time.time()
        total_batch = 0
        best_acc_val = 0.0 
        last_improved = 0
        
        lr = self.learning_rate_base
        with self._graph.as_default():
            self._sess.run(tf.assign(self._lr, tf.constant(lr, dtype=tf.float32)))
        
        valid_batches = Dataset.do_batching_data(valid_data, self.batch_size_eval)
        valid_batches = Dataset.do_standardizing_batches(valid_batches, self.settings)
        
        print('Creating model for evaluation ...')
        config_e = self.settings
        config_e.keep_prob = 1.0
        model_e = ModelWrapper(config_e)
        model_e.prepare_for_train_and_valid()
        
        flag_stop = False
        for epoch in range(self.num_epochs):
            print('Epoch: %d, training ...' % (epoch + 1) )
            
            train_batches = Dataset.do_batching_data(train_data, self.batch_size)
            train_batches = Dataset.do_standardizing_batches(train_batches, self.settings)
            
            for data_batch in train_batches:
                feed_dict = self._feed_data_train(data_batch)
                
                # valid
                if total_batch % self.valid_per_batch == 0:
                    # load
                    ckpt = tf.train.get_checkpoint_state(self.model_dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        model_e._saver.restore(model_e._sess, ckpt.model_checkpoint_path)
                    #print('evaluation-model created')                                        
                    loss_val, acc_val = model_e.evaluate(valid_batches)                    
                    #print('evaluated')
                    
                    # save best
                    if acc_val >= best_acc_val:  # >=
                        best_acc_val = acc_val
                        last_improved = total_batch
                        model_e._saver_best.save(model_e._sess,
                                                 os.path.join(model_e.model_dir + '_best', model_e.model_name),
                                                 global_step = total_batch)
                        
                        # pb
                        constant_graph = graph_util.convert_variables_to_constants(
                                model_e._sess, model_e._sess.graph_def,
                                output_node_names = self.pb_outputs_name)
                        with tf.gfile.FastGFile(model_e.pb_file, mode='wb') as f:
                            f.write(constant_graph.SerializeToString())
                        #
                    
                    # stop
                    if total_batch - last_improved >= self.patience_stop:
                        str_info = "no improvement for a long time, stop optimization at curr_batch: %d" \
                                    % total_batch
                        self._log_info(str_info)
                        print(str_info)
                        #
                        flag_stop = True
                        break # for batch
                    
                    # decay
                    if total_batch - last_improved >= self.patience_decay or \
                    (not self.use_metric and \
                     total_batch > 0 and \
                     total_batch % self.patience_decay == 0):
                        lr *= self.ratio_decay
                        with self._graph.as_default():
                            self._sess.run(tf.assign(self._lr, tf.constant(lr, dtype=tf.float32)))
                        last_improved = total_batch
                        #
                        str_info = 'learning_rate DECAYED at total_batch: %d' % total_batch
                        self._log_info(str_info)
                        print(str_info)
                    
                    # time
                    # time_cost = time.time() - start_time
                    #
                    str_info = 'loss, metric, best_metric: %.6f, %.4f, %.4f' % (loss_val,
                                                                                acc_val, best_acc_val)
                    self._log_info(str_info)
                    # print(str_info)
                    #
                    str_info = 'curr_batch: %d, lr: %f' % (total_batch, lr)
                    self._log_info(str_info)
                    # print(str_info)

                # optim
                self._sess.run(self._train_op, feed_dict = feed_dict)
                total_batch += 1
                    
                # save
                if total_batch % self.save_per_batch == 0:
                    #s = session.run(merged_summary, feed_dict=feed_dict)
                    #writer.add_summary(s, total_batch)
                    loss = self._sess.run(self._loss_tensor, feed_dict = feed_dict)
                    metric = 0.0
                    if self.use_metric:
                        metric = self._sess.run(self._metric_tensor, feed_dict = feed_dict)
                    #
                    str_info = "epoch: %d" % (epoch + 1)
                    self._log_info(str_info)
                    # print(str_info)
                    #
                    str_info = "loss, metric of train: %f, %f" % (loss, metric)
                    self._log_info(str_info)
                    self._log_info("")
                    # print(str_info)
                    # print()
                    
                    self._saver.save(self._sess,
                                    os.path.join(self.model_dir, self.model_name),
                                    global_step = total_batch)
                #
            #
            if flag_stop: break # for epoch
            #
        #
        str_info = "training ended after total epoches: %d" % (epoch + 1)
        self._log_info(str_info)
        self._log_info("")
        # print(str_info)
        # print()
        #
            
if __name__ == '__main__':
    
    sett = ModelSettings('vocab_placeholder', False)
    
    sett.model_tag = 'cnn'
    
    sett.check_settings()
    
    #print(dir(sett))    
    #l = [i for i in dir(sett) if inspect.isbuiltin(getattr(sett, i))]
    #l = [i for i in dir(sett) if inspect.isfunction(getattr(sett, i))]
    #l = [i for i in dir(sett) if not callable(getattr(sett, i))]
    
    print(sett.__dict__.keys())
    print()
    
    model = ModelWrapper(sett)
    
    print(model.__dict__.keys())
    print()
    for key in model.__dict__.keys():
        print(key + ': ' + str(model.__dict__[key]) )
  
    
            
            