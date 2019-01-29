# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 22:31:20 2018

@author: limingfan
"""

import os
import numpy as np
# import json
# import copy
# import time

import tensorflow as tf
from tensorflow.python.framework import graph_util

from model_settings import ModelSettings


"""
This class is meant to be task-agnostic.

"""
      
        
class ModelWrapper():
    
    def __init__(self, settings):
        """
        """        
        self.set_model_settings(settings)
        
    def set_model_settings(self, settings):
        """
        """        
        self.settings = settings
        #
        # super().__init__(settings.is_train)
        for key in settings.__dict__.keys():
            # print(key + ': ' + str(self.__dict__[key]) )                   
            self.__dict__[key] = settings.__dict__[key]
            # print(key + ': ' + str(self.__dict__[key]) ) 
        # 
        # session info
        self.sess_config = tf.ConfigProto(log_device_placement = settings.log_device)
        self.sess_config.gpu_options.allow_growth = settings.gpu_mem_growth
        
    
    # predict
    def prepare_for_prediction(self, pb_file_path = None):
        #
        if pb_file_path is None: pb_file_path = self.pb_file 
        if not os.path.exists(pb_file_path):
            assert False, 'ERROR: %s NOT exists, when prepare_for_prediction()' % pb_file_path
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
    
    def predict_from_batch(self, x_batch):
        """
        """
        # x_batch = data_batcher.convert_data_to_batch(data, vocab, settings)  
        feed_dict = self.feed_data_predict(x_batch)
        outputs = self._sess.run(self._outputs_predict, feed_dict = feed_dict)
        
        return outputs

    def feed_data_predict(self, x_batch):        
        feed_dict = {}
        for idx in range(self._inputs_predict_num):
            feed_dict[self._inputs_predict[idx] ] = x_batch[idx]
        return feed_dict
        
    def feed_data_train(self, data_batch):        
        feed_dict = {}
        for idx in range(self._inputs_train_num):
            feed_dict[self._inputs_train[idx] ] = data_batch[idx]
        return feed_dict
    
    # one_batch functions
    def run_train_one_batch(self, one_batch):
        """
        """
        feed_dict = self.feed_data_train(one_batch)
        loss, _ = self._sess.run([self._loss_tensor, self._train_op],
                                 feed_dict = feed_dict)
        return loss
        
    def run_eval_one_batch(self, one_batch):
        """
        """
        feed_dict = self.feed_data_train(one_batch)        
        metric = None
        if self.use_metric:         
            results, loss, metric = self._sess.run(self._outputs_eval,
                                                   feed_dict = feed_dict)
        else:
            results, loss = self._sess.run(self._outputs_eval,
                                           feed_dict = feed_dict)         
        return results, loss, metric

    def run_predict_one_batch(self, one_batch):
        """
        """
        feed_dict = self.feed_data_train(one_batch)
        results = self._sess.run(self._outputs_train, feed_dict = feed_dict)        
        return results
    
    def run_debug_one_batch(self, one_batch):
        """
        """
        feed_dict = self.feed_data_train(one_batch)
        results = self._sess.run(self._debug_tensors, feed_dict = feed_dict)        
        return results

    #
    def prepare_for_train_and_valid(self, dir_ckpt = None):

        # graph
        self._graph = tf.Graph()
        with self._graph.as_default():
            #
            # model
            self.model_graph(self.settings)
            #
            # debug tensors
            self._debug_tensors = []
            for name in self.debug_tensors_name:
                tensor = self._graph.get_tensor_by_name(name)
                self._debug_tensors.append(tensor)
            #
            # keep_prob
            self._keep_prob = self._graph.get_tensor_by_name("keep_prob:0")
            #
            # all trainable vars
            self.trainable_vars = tf.trainable_variables()
            # print(self.trainable_vars)
            #            
            self._global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
            self._lr = tf.get_variable("lr", shape=[], dtype=tf.float32, trainable=False)
            #
            if self.use_metric:
                self._metric_tensor = self._graph.get_tensor_by_name(self.metric_name)
            self._loss_tensor = self._graph.get_tensor_by_name(self.loss_name)
            #
            if self.reg_lambda > 0.0:
                loss_reg = tf.add_n([ tf.nn.l2_loss(v) for v in self.trainable_vars
                                     if 'bias' not in v.name and 'embedding' not in v.name])
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
            self.assign_dropout_keep_prob(self.keep_prob)
            self.assign_learning_rate(self.learning_rate_base)
            
            # params count
            self.param_num = sum([np.prod(self._sess.run(tf.shape(v)))
                                  for v in self.trainable_vars])
            #
            str_info = 'Graph built, there are %d parameters in the model' % self.param_num
            self.logger.info(str_info)
            # print(str_info)
            #
            # info_dict = self.settings.trans_info_to_dict()
            # str_info = json.dumps(info_dict, ensure_ascii = False)
            # self.logger.info(str_info)
            # print(str_info)
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
            # for eval
            self._outputs_predict = []
            for item in self.outputs_predict_name:
                tensor = self._graph.get_tensor_by_name(item)
                self._outputs_predict.append(tensor)
            #
            if self.use_metric:
                self._outputs_eval = self._outputs_predict + [self._loss_tensor,
                                                              self._metric_tensor]
            else:
                self._outputs_eval = self._outputs_predict + [self._loss_tensor]
        #
        # load
        if dir_ckpt is None: dir_ckpt = self.model_dir + '_best'
        self.load_ckpt(dir_ckpt)
        #
        
    #
    def assign_dropout_keep_prob(self, keep_prob):
        """
        """        
        with self._graph.as_default():
            self._sess.run(tf.assign(self._keep_prob, tf.constant(keep_prob, dtype=tf.float32)))
            
    def assign_learning_rate(self, lr):
        """
        """        
        with self._graph.as_default():
            self._sess.run(tf.assign(self._lr, tf.constant(lr, dtype=tf.float32)))
        
    #
    def save_graph_pb_file(self, file_path):
        """
        """
        is_train = self.settings.is_train
        self.settings.is_train = False       #
        #
        model = ModelWrapper(self.settings)
        model.prepare_for_train_and_valid()        
        # model.load_ckpt(model.model_dir + '_best')
        model.assign_dropout_keep_prob(1.0)
        #
        constant_graph = graph_util.convert_variables_to_constants(
                model._sess, model._sess.graph_def,
                output_node_names = model.pb_outputs_name)
        with tf.gfile.FastGFile(file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())
        #
        str_info = 'pb_file saved: %s' % file_path
        self.logger.info(str_info)
        #
        self.settings.is_train = is_train           #
        #
            
    def save_ckpt_best(self, model_dir, model_name, step):
        """
        """
        self._saver_best.save(self._sess, os.path.join(model_dir, model_name),
                              global_step = step)
        
    def save_ckpt(self, model_dir, model_name, step):
        """
        """
        self._saver.save(self._sess, os.path.join(model_dir, model_name),
                         global_step = step)
    
    def load_ckpt(self, dir_ckpt):
        """
        """
        ckpt = tf.train.get_checkpoint_state(dir_ckpt)        
        if ckpt and ckpt.model_checkpoint_path:
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)
            #
            str_info = 'ckpt loaded from %s' % dir_ckpt
            self.logger.info(str_info)
            print(str_info)
        else:
            str_info = 'Failed: ckpt loading from %s' % dir_ckpt
            self.logger.info(str_info)
            print(str_info)            
    
    # graph and sess
    def get_model_graph_and_sess(self):
        """ for debug
        """
        return self._graph, self._sess
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
    
    #
    model = ModelWrapper(sett)
    
    print(model.__dict__.keys())
    print()
    for key in model.__dict__.keys():
        print(key + ': ' + str(model.__dict__[key]) )
        
    
    
