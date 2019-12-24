#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 23:19:42 2019
@author: li-ming-fan
"""

import os
import numpy as np

import re
import collections

import time
import logging
import json

import tensorflow as tf
from tensorflow.python.framework import graph_util

from abc import ABCMeta, abstractmethod

from .optim import linear_warmup_and_polynomial_decayed_lr
from .optim import adam_wd_optimizer, adam_optimizer


"""
This class is meant to be task-agnostic.
""" 
    
#
class ModelBaseboard(metaclass=ABCMeta):
    """
    """    
    def __init__(self, settings,
                 learning_rate_schedule = linear_warmup_and_polynomial_decayed_lr,
                 customized_optimizer = adam_optimizer):
        #
        self.learning_rate_schedule = learning_rate_schedule
        self.customized_optimizer = customized_optimizer
        #
        self.set_model_settings(settings)
        #
        # pb/debug tensor names
        self.vs_str_multi_gpu = "vs_gpu"
        #
        self.pb_input_names = {"input_x": "input_x:0"}    
        self.pb_output_names = {"logits": "vs_gpu/score/logits:0"}
        self.pb_save_names = ["vs_gpu/score/logits"]
        #
        self.debug_tensor_names = ["vs_gpu/score/logits:0",
                                   "vs_gpu/loss/loss_model:0"]
        #
        # be aware to assign the right value
        #

    #
    # three abstract methods
    @abstractmethod
    def build_placeholder(self):
        """  input_tensors, label_tensors = self.build_placeholder()
        """
        pass
    
    @abstractmethod
    def build_inference(self, input_tensors):
        """ output_tensors = self.build_inference(input_tensors)
            keep_prob = tf.get_variable("keep_prob", shape=[], dtype=tf.float32,
                                        trainable=False)
        """
        pass
    
    @abstractmethod
    def build_loss_and_metric(self, output_tensors, label_tensors):
        """ loss_tensors = self.build_loss(output_tensors, label_tensors)
        """
        pass

    #
    # settings and logger
    def set_model_settings(self, settings):
        #
        # session info
        if "log_device" not in settings.__dict__.keys():
            settings.__dict__["log_device"] = False 
        #
        if "soft_placement" not in settings.__dict__.keys():
            settings.__dict__["soft_placement"] = True 
        #
        if "gpu_mem_growth" not in settings.__dict__.keys():
            settings.__dict__["gpu_mem_growth"] = True 
        #
        # params
        if "reg_lambda" not in settings.__dict__.keys():
            settings.__dict__["reg_lambda"] = 0.0
        #
        if "reg_exclusions" not in settings.__dict__.keys():
            settings.__dict__["reg_exclusions"] = ["embedding", "bias", "layer_norm", "LayerNorm"]
        #
        if "grad_clip" not in settings.__dict__.keys():
            settings.__dict__["grad_clip"] = 0.0
        #
        if "saver_num_keep" not in settings.__dict__.keys():
            settings.__dict__["saver_num_keep"] = 5
        #
        # logger
        if "logger" in settings.__dict__.keys():
            self.log_path = settings.log_path
            self.logger = settings.logger
        else:
            #
            str_datetime = time.strftime("%Y-%m-%d-%H-%M")
            log_path = os.path.join(settings.log_dir, settings.model_name + "_" + str_datetime +".txt")
            self.log_path = log_path
            self.logger = ModelBaseboard.create_logger(log_path)
        #
        """
        for key in settings.__dict__.keys():                 
            self.__dict__[key] = settings.__dict__[key]
        """
        #
        # settings
        self.settings = settings
        self.num_gpu = len(settings.gpu_available.split(","))
        #
        # session info
        self.sess_config = tf.ConfigProto(log_device_placement = settings.log_device,
                                          allow_soft_placement = settings.soft_placement)
        self.sess_config.gpu_options.allow_growth = settings.gpu_mem_growth
        #

    @staticmethod
    def create_logger(log_path):
        """
        """
        logger = logging.getLogger(log_path)  # use log_path as log_name
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # self.logger.info('test')
        return logger

    @staticmethod
    def create_or_reset_log_file(log_path):        
        with open(log_path, 'w', encoding='utf-8'):
            print("log file renewed")
    
    @staticmethod
    def close_logger(logger):
        for item in self.logger.handlers:
            item.close()
            print("logger handler item closed")

    #
    # feed_data_train
    def feed_data_train(self, data_batch):
        feed_dict = {}
        for tensor_tag, tensor in self._inputs_train.items():
            feed_dict[tensor] = data_batch[tensor_tag]
        return feed_dict
    
    # one_batch functions
    def run_train_one_batch(self, one_batch):
        #
        feed_dict = self.feed_data_train(one_batch)
        result_dict = self._sess.run(self._outputs_train, feed_dict = feed_dict)
        return result_dict
        
    def run_eval_one_batch(self, one_batch):
        #
        feed_dict = self.feed_data_train(one_batch)
        result_dict = self._sess.run(self._outputs_eval, feed_dict = feed_dict)      
        return result_dict
        
    def run_debug_one_batch(self, one_batch):
        #
        assert self.num_gpu == 1, "debug mode can only be run with single gpu"
        #
        feed_dict = self.feed_data_train(one_batch)
        result_list = self._sess.run(self._debug_tensors, feed_dict = feed_dict)        
        return result_list
    
    #
    def prepare_for_train_and_valid(self, dir_ckpt = None):
        """
        """        
        if self.num_gpu == 1:
            self.prepare_for_train_and_valid_single_gpu(dir_ckpt)
        else:
            self.prepare_for_train_and_valid_multi_gpu(dir_ckpt)
        #
    
    #
    def prepare_for_train_and_valid_single_gpu(self, dir_ckpt = None):
        """
        """        
        # graph
        self._graph = tf.Graph()
        with self._graph.as_default():
            #
            self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                                               initializer = tf.constant_initializer(0),
                                               trainable = False)
            #
            self.learning_rate_tensor = self.learning_rate_schedule(self.settings, self.global_step)
            self.learning_rate_tensor = tf.identity(self.learning_rate_tensor, name= "lr")
            #
            # optimizer
            # optimizer = tf.train.MomentumOptimizer(learning_rate, MOMENTUM, use_nesterov=True)
            # optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr, epsilon=1e-6)              
            # optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = MOMENTUM)
            #
            if self.settings.optimizer_type == 'sgd':
                self._opt = tf.train.GradientDescentOptimizer(self.learning_rate_tensor)
            elif self.settings.optimizer_type == 'momentum':
                self._opt = tf.train.MomentumOptimizer(self.learning_rate_tensor, self.settings.beta_1, use_nesterov=True)
            elif self.settings.optimizer_type == 'adam':
                self._opt = tf.train.AdamOptimizer(self.learning_rate_tensor, self.settings.beta_1, self.settings.beta_2)
            elif self.settings.optimizer_type == 'adam_wd':
                self._opt = adam_wd_optimizer(self.settings, self.learning_rate_tensor)
            elif self.settings.optimizer_type == 'customized':
                self._opt = self.customized_optimizer(self.settings, self.learning_rate_tensor)
            else:
                assert False, "NOT supported optimizer_type"
            #
            # model
            input_tensors, label_tensors = self.build_placeholder()
            #
            vs_str = self.vs_str_multi_gpu
            vs_prefix = vs_str + "/"
            with tf.variable_scope(vs_str):
                output_tensors = self.build_inference(input_tensors)
                loss_metric_tensors = self.build_loss_and_metric(output_tensors, label_tensors)
            #
            # all trainable vars
            self.trainable_vars = tf.trainable_variables()
            # print(self.trainable_vars)
            #
            # keep_prob
            self._keep_prob = self._graph.get_tensor_by_name(vs_prefix + "keep_prob:0")
            #
            # loss and metric
            self._loss_tensor = loss_metric_tensors["loss_model"]
            if self.settings.use_metric_in_graph:
                self._metric_tensor = loss_metric_tensors["metric"]         
            #
            # regularization
            def is_excluded(v):
                for item in self.settings.reg_exclusions:
                    if item in v.name: return True
                return False
            #
            if self.settings.reg_lambda > 0.0 and self.settings.optimizer_type != 'adam_wd':
                loss_reg = tf.add_n( [tf.nn.l2_loss(v) for v in self.trainable_vars
                                     if not is_excluded(v)] )
                loss_reg = tf.multiply(loss_reg, self.settings.reg_lambda)
                self._loss_tensor = tf.add(self._loss_tensor, loss_reg)
            #
            # grad_clip
            grad_and_vars = self._opt.compute_gradients(self._loss_tensor)            
            if self.settings.grad_clip > 0.0:
                gradients, variables = zip(*grad_and_vars)
                grads, _ = tf.clip_by_global_norm(gradients, self.settings.grad_clip)
                grad_and_vars = zip(grads, variables)
            #
            # train_op
            self._train_op = self._opt.apply_gradients(grad_and_vars,
                                                       global_step = self.global_step)
            #                 
            # save info
            self._saver = tf.train.Saver()
            self._saver_best = tf.train.Saver()
            
            # sess
            self._sess = tf.Session(graph=self._graph, config = self.sess_config)

            # initialize the model
            self._sess.run(tf.global_variables_initializer())
            self.assign_dropout_keep_prob(self.settings.keep_prob)
            # self.assign_learning_rate(self.learning_rate_base)
            
            # params count
            self.num_vars = len(self.trainable_vars)
            str_info = 'graph built, there are %d variables in the model' % self.num_vars
            self.logger.info(str_info)
            print(str_info)
            #
            tf_shapes = [tf.shape(v) for v in self.trainable_vars]
            shapes_v = self._sess.run(tf_shapes)
            params_v = [np.prod(item) for item in shapes_v]
            self.param_num = sum(params_v)
            #
            str_info = 'there are %d parameters in the model' % self.param_num
            self.logger.info(str_info)
            print(str_info)
            #
            print()
            for idx in range(self.num_vars):
                print(self.trainable_vars[idx])
                print(params_v[idx])
            print()
            #
            
            # outputs_eval
            self._outputs_eval = output_tensors
            self._outputs_eval["loss_optim"] = self._loss_tensor
            if self.settings.use_metric_in_graph:
                self._outputs_eval["metric"] = self._metric_tensor
            #
            # outputs_train
            # [self._loss_tensor, self.learning_rate_tensor, self._train_op]
            self._outputs_train = {"loss_optim": self._loss_tensor,
                                   "lr": self.learning_rate_tensor,
                                   "global_step": self.global_step,
                                   "train_op": self._train_op }
            #
            # inputs_train
            self._inputs_train = dict(input_tensors, **label_tensors)
            #
            # debug tensors
            print("self.debug_tensor_names:")
            print(self.debug_tensor_names)
            print("if the above is not right, please assign the right value")
            #
            self._debug_tensors = []
            for name in self.debug_tensor_names:
                tensor = self._graph.get_tensor_by_name(name)
                self._debug_tensors.append(tensor)
            #

        #
        # load
        # if dir_ckpt is None: dir_ckpt = self.model_dir + '_best'
        if dir_ckpt is not None: self.load_ckpt(dir_ckpt)
        #
        
    #
    @staticmethod
    def sum_up_gradients(list_grad_bundles):
        """ list_grad_bundles: [ [(g1,v1), (g2, v2), ...],
                                 [(g1,v1), (g2, v2), ...], ...,
                                 [(g1,v1), (g2, v2), ...] ]
            zip(*list_grad_bundles): [ ... ]
        """
        summed_grads = []
        for grads_per_var in zip(*list_grad_bundles):
            grads = []
            for g, _ in grads_per_var:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            #
            grads_concat = tf.concat(grads, 0)
            grads_sum = tf.reduce_sum(grads_concat, 0)
            grad_and_var = (grads_sum, grads_per_var[0][1])
            summed_grads.append(grad_and_var)
        #
        return summed_grads
    
    #
    def prepare_for_train_and_valid_multi_gpu(self, dir_ckpt = None):
        """
        """        
        gpu_batch_split = self.settings.gpu_batch_split
        #
        # graph
        self._graph = tf.Graph()
        with self._graph.as_default():
            #
            self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                                               initializer = tf.constant_initializer(0),
                                               trainable = False)
            #
            self.learning_rate_tensor = self.learning_rate_schedule(self.settings, self.global_step)
            self.learning_rate_tensor = tf.identity(self.learning_rate_tensor, name= "lr")
            #
            # optimizer
            # optimizer = tf.train.MomentumOptimizer(learning_rate, MOMENTUM, use_nesterov=True)
            # optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr, epsilon=1e-6)              
            # optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = MOMENTUM)
            #
            if self.settings.optimizer_type == 'sgd':
                self._opt = tf.train.GradientDescentOptimizer(self.learning_rate_tensor)
            elif self.settings.optimizer_type == 'momentum':
                self._opt = tf.train.MomentumOptimizer(self.learning_rate_tensor, self.settings.beta_1, use_nesterov=True)
            elif self.settings.optimizer_type == 'adam':
                self._opt = tf.train.AdamOptimizer(self.learning_rate_tensor, self.settings.beta_1, self.settings.beta_2)
            elif self.settings.optimizer_type == 'adam_wd':
                self._opt = adam_wd_optimizer(self.settings, self.learning_rate_tensor)
            elif self.settings.optimizer_type == 'customized':
                self._opt = self.customized_optimizer(self.settings, self.learning_rate_tensor)
            else:
                assert False, "NOT supported optimizer_type"
            #
            # model, placeholder
            input_tensors, label_tensors = self.build_placeholder()
            #
            # split among gpu
            inputs = []
            labels = []
            for idx in range(self.num_gpu):
                inputs.append({})
                labels.append({})
            #
            for key in input_tensors:
                in_split = tf.split(input_tensors[key], gpu_batch_split, axis = 0)
                for idx in range(self.num_gpu):
                    inputs[idx][key] = in_split[idx]
            #
            for key in label_tensors:
                lb_split = tf.split(label_tensors[key], gpu_batch_split, axis = 0)
                for idx in range(self.num_gpu):
                    labels[idx][key] = lb_split[idx]
            #
            # model, inference, loss
            outputs_list = []
            loss_list = []
            metric_list = []
            grads_bundles = []
            #
            vs_str = self.vs_str_multi_gpu
            vs_prefix = vs_str + "/"
            with tf.variable_scope(vs_str):
                for gid in range(self.num_gpu):
                    with tf.device("/gpu:%d" % gid), tf.name_scope("bundle_%d" % gid):
                        #
                        output_tensors = self.build_inference(inputs[gid])
                        loss_metric_tensors = self.build_loss_and_metric(output_tensors,
                                                                         labels[gid])
                        #
                        tf.get_variable_scope().reuse_variables()
                        #
                        outputs_list.append(output_tensors)
                        #
                        loss = loss_metric_tensors["loss_model"]
                        loss_list.append(loss)
                        #
                        if self.settings.use_metric_in_graph:
                            metric = loss_metric_tensors["metric"]
                            metric_list.append(metric)
                        #
                        grads = self._opt.compute_gradients(loss)
                        grads_bundles.append(grads)
                        #
            #
            # all trainable vars
            self.trainable_vars = tf.trainable_variables()
            # print(self.trainable_vars)
            #
            # keep_prob
            self._keep_prob = self._graph.get_tensor_by_name(vs_prefix + "keep_prob:0")
            #
            # regularization
            def is_excluded(v):
                for item in self.settings.reg_exclusions:
                    if item in v.name: return True
                return False
            #
            if self.settings.reg_lambda > 0.0 and self.settings.optimizer_type != 'adam_wd':
                loss_reg = tf.add_n( [tf.nn.l2_loss(v) for v in self.trainable_vars
                                     if not is_excluded(v)] )
                loss_reg = tf.multiply(loss_reg, self.settings.reg_lambda)
                grads_reg = self._opt.compute_gradients(loss_reg)
                #
                grads_reg_clear = []
                for g, v in grads_reg:
                    if g is None: g = tf.zeros_like(v)
                    grads_reg_clear.append( (g,v) )
                    #
                #
                grads_bundles.append(grads_reg_clear)
                #
            #           
            # grad sum
            grads_summed = ModelBaseboard.sum_up_gradients(grads_bundles)
            #            
            # grad_clip
            if self.settings.grad_clip > 0.0:
                gradients, variables = zip(*grads_summed)
                grads, _ = tf.clip_by_global_norm(gradients, self.settings.grad_clip)
                grads_summed = zip(grads, variables)
            #
            # train_op
            self._train_op = self._opt.apply_gradients(grads_summed,
                                                       global_step = self.global_step)
            #               
            # save info
            self._saver = tf.train.Saver()
            self._saver_best = tf.train.Saver()
            
            # sess
            self._sess = tf.Session(graph=self._graph, config = self.sess_config)

            # initialize the model
            self._sess.run(tf.global_variables_initializer())
            self.assign_dropout_keep_prob(self.settings.keep_prob)
            # self.assign_learning_rate(self.learning_rate_base)
            
            # params count
            self.num_vars = len(self.trainable_vars)
            str_info = 'graph built, there are %d variables in the model' % self.num_vars
            self.logger.info(str_info)
            print(str_info)
            #
            tf_shapes = [tf.shape(v) for v in self.trainable_vars]
            shapes_v = self._sess.run(tf_shapes)
            params_v = [np.prod(item) for item in shapes_v]
            self.param_num = sum(params_v)
            #
            str_info = 'there are %d parameters in the model' % self.param_num
            self.logger.info(str_info)
            print(str_info)
            #
            print()
            for idx in range(self.num_vars):
                print(self.trainable_vars[idx])
                print(params_v[idx])
            print()
            #
            
            #
            # loss
            loss_w = [loss_list[gid] * self.settings.gpu_batch_split[gid]
                                         for gid in range(self.num_gpu)]
            self._loss_tensor = tf.add_n(loss_w) / self.settings.batch_size
            #
            # metric
            if self.settings.use_metric_in_graph:
                metric_w = [metric_list[gid] * self.settings.gpu_batch_split[gid]
                                            for gid in range(self.num_gpu)]
                self._metric_tensor = tf.add_n(metric_w) / self.settings.batch_size
            #
            # outputs eval
            self._outputs_eval = {}
            for key in outputs_list[0]:
                value = []
                for idx in range(self.num_gpu):
                    value.append(outputs_list[idx][key])
                #
                self._outputs_eval[key] = tf.concat(value, axis=0)
                #
            #
            self._outputs_eval["loss_optim"] = self._loss_tensor
            if self.settings.use_metric_in_graph:
                self._outputs_eval["metric"] = self._metric_tensor
            #
            # outputs_train
            # [self._loss_tensor, self.learning_rate_tensor, self._train_op]
            self._outputs_train = {"loss_optim": self._loss_tensor,
                                   "lr": self.learning_rate_tensor,
                                   "global_step": self.global_step,
                                   "train_op": self._train_op }
            #
            # inputs_train
            self._inputs_train = dict(input_tensors, **label_tensors)
            #
            # debug tensors
            # debug only works with single-gpu
            #
            
        #
        # load
        # if dir_ckpt is None: dir_ckpt = self.model_dir + '_best'
        if dir_ckpt is not None: self.load_ckpt(dir_ckpt)
        #
    
    #
    # assign
    def assign_dropout_keep_prob(self, keep_prob):
        #
        with self._graph.as_default():
            self._sess.run(tf.assign(self._keep_prob,
                                     tf.constant(keep_prob, dtype=tf.float32)))
        #
        
    def assign_global_step(self, step):
        #
        with self._graph.as_default():
            self._sess.run(tf.assign(self.global_step, tf.constant(step, dtype=tf.int32)))
        #
        
    def assign_learning_rate(self, lr_value):
        #
        with self._graph.as_default():
            self._sess.run(tf.assign(self.learning_rate_tensor,
                                     tf.constant(lr_value, dtype=tf.float32)))
        #
    
    #
    # save and load
    def save_ckpt_best(self, model_dir, model_name, step):
        #
        self._saver_best.save(self._sess, os.path.join(model_dir, model_name),
                              global_step = step)
        
    def save_ckpt(self, model_dir, model_name, step):
        #
        self._saver.save(self._sess, os.path.join(model_dir, model_name),
                         global_step = step)
    
    def load_ckpt(self, dir_ckpt):
        #
        ckpt = tf.train.get_checkpoint_state(dir_ckpt)        
        if ckpt and ckpt.model_checkpoint_path:
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)
            #
            str_info = 'ckpt loaded from %s' % dir_ckpt
            self.logger.info(str_info)
            print(str_info)
        else:
            str_info = 'loading ckpt failed: ckpt loading from %s' % dir_ckpt
            self.logger.info(str_info)
            print(str_info)
            
    #
    # predict, pb
    @staticmethod
    def load_ckpt_and_save_pb_file(model, dir_ckpt):
        """
        """
        is_train = model.settings.is_train
        num_gpu = model.num_gpu
        #
        model.settings.is_train = False                #
        model.num_gpu = 1                              #
        #
        model.prepare_for_train_and_valid(dir_ckpt)              # loaded here 
        model.assign_dropout_keep_prob(1.0)
        #
        print("model.pb_save_names:")
        print(model.pb_save_names)
        print("if the above is not right, please assign the right value")
        #
        pb_file = os.path.join(dir_ckpt, "model_frozen.pb")
        #
        constant_graph = graph_util.convert_variables_to_constants(
                model._sess, model._sess.graph_def,
                output_node_names = model.pb_save_names)
        with tf.gfile.GFile(pb_file, mode='wb') as f:
            f.write(constant_graph.SerializeToString())
        #
        str_info = 'pb_file saved: %s' % pb_file
        model.logger.info(str_info)
        print(str_info)
        #
        model.settings.is_train = is_train
        model.num_gpu = num_gpu
        #
        
    def prepare_for_prediction_with_pb(self, pb_file_path = None):
        """ load pb for prediction
        """
        if pb_file_path is None: pb_file_path = self.settings.pb_file 
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
                print('Graph loaded for prediction')
                #
            #
            # tensor names
            print("self.pb_input_names:")
            print(self.pb_input_names)
            print("if the above is not right, please assign the right value")
            print("similar for self.pb_output_names")
            #
            self._inputs_pred = {}
            for tensor_tag, tensor_name in self.pb_input_names.items():
                tensor = self._graph.get_tensor_by_name(tensor_name)
                self._inputs_pred[tensor_tag] = tensor
            #
            self._outputs_pred = {}
            for tensor_tag, tensor_name in self.pb_output_names.items():
                tensor = self._graph.get_tensor_by_name(tensor_name)
                self._outputs_pred[tensor_tag] = tensor
            #
            print('Graph loaded for prediction')
            #
        #
        self._sess = tf.Session(graph = self._graph, config = self.sess_config)
        #
    
    def predict_with_pb_from_batch(self, x_batch):
        #
        feed_dict = self.feed_data_predict(x_batch)
        output_dict = self._sess.run(self._outputs_pred, feed_dict = feed_dict)        
        return output_dict

    def feed_data_predict(self, x_batch):        
        feed_dict = {}
        for tensor_tag, tensor in self._inputs_pred.items():
            feed_dict[tensor] = x_batch[tensor_tag]
        return feed_dict
    
    #
    # graph and sess
    def get_model_graph_and_sess(self):
        #
        return self._graph, self._sess
        #

#
def get_assignment_map_from_ckpt(init_ckpt,
                                 name_replace_dict={},
                                 trainable_vars=None):
    """ name_replace_dict = { old_name_str_chunk: new_name_str_chunk }
    """
    if trainable_vars is None:
        trainable_vars = tf.trainable_variables()
    #
    name_to_variable = collections.OrderedDict()
    for var in trainable_vars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var
        #
    
    #
    ckpt_vars = tf.train.list_variables(init_ckpt)
    # 
    assignment_map = collections.OrderedDict()
    for x in ckpt_vars:
        (name, var) = (x[0], x[1])
        #
        for k, v in name_replace_dict.items():
            if k in name:
                name_new = name.replace(k, v)
                break
        else:
            continue
        #
        if name_new not in name_to_variable:
            continue
        #
        assignment_map[name] = name_new
        print("name_old: %s" % name)
        print("name_new: %s" % name_new)
        #
    
    return assignment_map

def remove_from_trainable_variables(non_trainable_names,
                                    trainable_vars=None,
                                    graph=None):
    """
    """
    if graph is None:
        graph = tf.get_default_graph()
    #
    if trainable_vars is None:
        trainable_vars = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # tf.trainable_variables()
        
    #    
    graph.clear_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    #
    for var in trainable_vars:
        for item in non_trainable_names:
            if item in var.name:
                print("not_training: %s" % var.name)
                break
        else:
            graph.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES, var)
        #
    #
        
def initialize_with_pretrained_ckpt(init_ckpt,                                    
                                    name_replace_dict={},
                                    non_trainable_names=[],                                    
                                    assignment_map=None,
                                    trainable_vars=None,
                                    graph=None):
    """ name_replace_dict = { old_name_str_chunk: new_name_str_chunk }
        non_trainable_names = ["bert", "word_embeddings"]  # for example
    """
    if assignment_map is None:
        assignment_map = get_assignment_map_from_ckpt(init_ckpt,
                                                      name_replace_dict,
                                                      trainable_vars)
    #
    # assign
    tf.train.init_from_checkpoint(init_ckpt, assignment_map)
    #
    # tune or not
    remove_from_trainable_variables(non_trainable_names, trainable_vars, graph)
    #

#
if __name__ == '__main__':
    
    pass