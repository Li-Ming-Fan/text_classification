# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 22:31:20 2018

@author: limingfan
"""

import os
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import graph_util


"""
This class is meant to be task-agnostic.

"""

#
def get_warmup_and_exp_decayed_lr(settings, global_step):
    """ lr_base, warmup_steps, decay_steps, decay_rate, staircase
    """
    learning_rate = tf.constant(value = settings.learning_rate_base,
                                shape = [], dtype = tf.float32)    
    learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                               settings.decay_steps,
                                               settings.decay_rate,
                                               settings.staircase)    
    if settings.warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(settings.warmup_steps, dtype=tf.int32)
        
        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
        
        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = settings.learning_rate_base * warmup_percent_done
        
        learning_rate = tf.cond(global_steps_int < warmup_steps_int,
                                lambda: warmup_learning_rate,
                                lambda: learning_rate)
    #
    return learning_rate
    #
    
#
class ModelWrapper():
    
    def __init__(self, settings, model_graph,
                 learning_rate_schedule = None, customized_optimizer = None):
        #
        self.set_model_settings(settings)
        self.model_graph = model_graph
        #
        if learning_rate_schedule is None:
            self.learning_rate_schedule = get_warmup_and_exp_decayed_lr
        else:
            self.learning_rate_schedule = learning_rate_schedule
        #
        self.customized_optimizer = customized_optimizer
        # self._opt = self.customized_optimizer(self.settings, self.learning_rate_tensor)
        #
        
    def set_model_settings(self, settings):
        #
        self.settings = settings
        #
        for key in settings.__dict__.keys():                 
            self.__dict__[key] = settings.__dict__[key]
        #
        self.num_gpu = len(self.gpu_available.split(","))
        #
        # session info
        self.sess_config = tf.ConfigProto(log_device_placement = settings.log_device,
                                          allow_soft_placement = settings.soft_placement)
        self.sess_config.gpu_options.allow_growth = settings.gpu_mem_growth
        #
    
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
        #
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
        #
        feed_dict = self.feed_data_train(one_batch)
        loss, lr, _ = self._sess.run([self._loss_tensor, self.learning_rate_tensor,
                                      self._train_op], feed_dict = feed_dict)
        return loss, lr
        
    def run_eval_one_batch(self, one_batch):
        #
        feed_dict = self.feed_data_train(one_batch)        
        metric = None
        if self.use_metric:         
            *results, loss, metric = self._sess.run(self._outputs_eval,
                                                    feed_dict = feed_dict)
        else:
            *results, loss = self._sess.run(self._outputs_eval,
                                            feed_dict = feed_dict)         
        return results, loss, metric
        
    def run_debug_one_batch(self, one_batch):
        #
        assert self.num_gpu == 1, "debug mode can only be run with single gpu"
        #
        feed_dict = self.feed_data_train(one_batch)
        results = self._sess.run(self._debug_tensors, feed_dict = feed_dict)        
        return results

    #
    def prepare_for_train_and_valid(self, dir_ckpt = None):
        #
        if self.num_gpu == 1:
            self.prepare_for_train_and_valid_single_gpu(dir_ckpt)
        else:
            self.prepare_for_train_and_valid_multi_gpu(dir_ckpt)
        #
    
    #
    def prepare_for_train_and_valid_single_gpu(self, dir_ckpt = None):

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
            if self.optimizer_type == 'sgd':
                self._opt = tf.train.GradientDescentOptimizer(self.learning_rate_tensor)
            elif self.optimizer_type == 'momentum':
                self._opt = tf.train.MomentumOptimizer(self.learning_rate_tensor, self.momentum, use_nesterov=True)
            elif self.optimizer_type == 'adam':
                self._opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate_tensor, beta1 = self.momentum)
            elif self.optimizer_type == 'customized':
                self._opt = self.customized_optimizer(self.settings, self.learning_rate_tensor)
            else:
                assert False, "NOT supported optimizer_type"
            #
            # model
            input_tensors, label_tensors = self.model_graph.build_placeholder(self.settings)
            #
            vs_str = self.vs_str_multi_gpu
            vs_prefix = vs_str + "/"
            with tf.variable_scope(vs_str):
                output_tensors = self.model_graph.build_inference(self.settings, input_tensors)
                loss, metric = self.model_graph.build_loss_and_metric(self.settings, output_tensors, label_tensors)
            #
            # all trainable vars
            self.trainable_vars = tf.trainable_variables()
            # print(self.trainable_vars)
            #
            # keep_prob
            self._keep_prob = self._graph.get_tensor_by_name(vs_prefix + "keep_prob:0")
            #
            # metric and loss
            if self.use_metric:
                self._metric_tensor = metric # self._graph.get_tensor_by_name(self.metric_name)
            #
            self._loss_tensor = loss # self._graph.get_tensor_by_name(self.loss_name)
            #
            # regularization
            def is_excluded(v):
                for item in self.reg_exclusions:
                    if item in v.name: return True
                return False
            #
            if self.reg_lambda > 0.0:
                loss_reg = tf.add_n( [tf.nn.l2_loss(v) for v in self.trainable_vars
                                     if not is_excluded(v)] )
                loss_reg = tf.multiply(loss_reg, self.reg_lambda)
                self._loss_tensor = tf.add(self._loss_tensor, loss_reg)
            #
            # grad_clip
            grad_and_vars = self._opt.compute_gradients(self._loss_tensor)            
            if self.grad_clip > 0.0:
                gradients, variables = zip(*grad_and_vars)
                grads, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
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
            self.assign_dropout_keep_prob(self.keep_prob)
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
            
            # outputs train
            self._outputs_train = output_tensors
            #
            # outputs eval
            self._outputs_eval = []
            for item in self._outputs_train:
                self._outputs_eval.append(item)
            self._outputs_eval.append(self._loss_tensor)
            #
            if self.use_metric:
                self._outputs_eval.append(self._metric_tensor)
            #
            # train inputs
            self._inputs_train = []
            for item in self.inputs_train_name:
                tensor = self._graph.get_tensor_by_name(item)
                self._inputs_train.append(tensor)
            #          
            self._inputs_train_num = len(self._inputs_train)
            #
            # debug tensors
            self._debug_tensors = []
            for name in self.debug_tensors_name:
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
        
        gpu_batch_split = self.gpu_batch_split

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
            if self.optimizer_type == 'sgd':
                self._opt = tf.train.GradientDescentOptimizer(self.learning_rate_tensor)
            elif self.optimizer_type == 'momentum':
                self._opt = tf.train.MomentumOptimizer(self.learning_rate_tensor, self.momentum, use_nesterov=True)
            elif self.optimizer_type == 'adam':
                self._opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate_tensor, beta1 = self.momentum)
            elif self.optimizer_type == 'customized':
                self._opt = self.customized_optimizer(self.settings, self.learning_rate_tensor)
            else:
                assert False, "NOT supported optimizer_type"
            #
            # model, placeholder
            input_tensors, label_tensors = self.model_graph.build_placeholder(self.settings)
            #
            # split among gpu
            inputs = []
            labels = []
            #
            for idx in range(len(input_tensors)):
                in_split = tf.split(input_tensors[idx], gpu_batch_split, axis = 0)
                inputs.append(in_split)
            #
            for idx in range(len(label_tensors)):
                lb_split = tf.split(label_tensors[idx], gpu_batch_split, axis = 0)
                labels.append(lb_split)
            #
            inputs_split = list(zip(*inputs))
            labels_split = list(zip(*labels))
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
                        output_tensors = self.model_graph.build_inference(self.settings,
                                                                          inputs_split[gid])
                        loss, metric = self.model_graph.build_loss_and_metric(self.settings,
                                                                              output_tensors,
                                                                              labels_split[gid])
                        #
                        tf.get_variable_scope().reuse_variables()
                        #
                        grads = self._opt.compute_gradients(loss)
                        grads_bundles.append(grads)
                        #
                        outputs_list.append(output_tensors)
                        loss_list.append(loss)
                        metric_list.append(metric)
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
                for item in self.reg_exclusions:
                    if item in v.name: return True
                return False
            #
            if self.reg_lambda > 0.0:
                loss_reg = tf.add_n( [tf.nn.l2_loss(v) for v in self.trainable_vars
                                     if not is_excluded(v)] )
                loss_reg = tf.multiply(loss_reg, self.reg_lambda)
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
            grads_summed = ModelWrapper.sum_up_gradients(grads_bundles)
            #            
            # grad_clip
            if self.grad_clip > 0.0:
                gradients, variables = zip(*grads_summed)
                grads, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
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
            self.assign_dropout_keep_prob(self.keep_prob)
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
            loss_w = [loss_list[gid] * self.gpu_batch_split[gid]
                                         for gid in range(self.num_gpu)]
            self._loss_tensor = tf.add_n(loss_w) / self.batch_size
            #
            # metric
            if self.use_metric:
                metric_w = [metric_list[gid] * self.gpu_batch_split[gid]
                                                for gid in range(self.num_gpu)]
                self._metric_tensor = tf.add_n(metric_w) / self.batch_size
            #
            # output
            outputs_list_z = zip(*outputs_list)
            outputs_list_c = []
            for item in outputs_list_z:
                item_c = tf.concat(item, 0)
                outputs_list_c.append(item_c)
            #
            self._outputs_train = outputs_list_c
            #
            # outputs eval
            if self.use_metric:
                self._outputs_eval = self._outputs_train + [self._loss_tensor,
                                                            self._metric_tensor]
            else:
                self._outputs_eval = self._outputs_train + [self._loss_tensor]
            #
            # train inputs
            self._inputs_train = []
            for item in self.inputs_train_name:
                tensor = self._graph.get_tensor_by_name(item)
                self._inputs_train.append(tensor)
            #          
            self._inputs_train_num = len(self._inputs_train)
            #
            
        #
        # load
        # if dir_ckpt is None: dir_ckpt = self.model_dir + '_best'
        if dir_ckpt is not None: self.load_ckpt(dir_ckpt)
        #
        
    #
    def assign_dropout_keep_prob(self, keep_prob):
        #
        with self._graph.as_default():
            self._sess.run(tf.assign(self._keep_prob, tf.constant(keep_prob, dtype=tf.float32)))
        #
        
    def assign_global_step(self, step):
        #
        with self._graph.as_default():
            self._sess.run(tf.assign(self.global_step, tf.constant(step, dtype=tf.int32)))
        #
    
    #
    def load_ckpt_and_save_pb_file(self, dir_ckpt):
        #
        is_train = self.settings.is_train
        self.settings.is_train = False       #
        #
        model = ModelWrapper(self.settings, self.settings.model_graph)
        model.prepare_for_train_and_valid_single_gpu(dir_ckpt)         # loaded here 
        model.assign_dropout_keep_prob(1.0)
        #
        pb_file = os.path.join(dir_ckpt, "model_saved.pb")
        #
        constant_graph = graph_util.convert_variables_to_constants(
                model._sess, model._sess.graph_def,
                output_node_names = model.pb_outputs_name)
        with tf.gfile.GFile(pb_file, mode='wb') as f:
            f.write(constant_graph.SerializeToString())
        #
        str_info = 'pb_file saved: %s' % pb_file
        self.logger.info(str_info)
        #
        self.settings.is_train = is_train           #
        #
            
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
            str_info = 'Failed: ckpt loading from %s' % dir_ckpt
            self.logger.info(str_info)
            print(str_info)            

    # graph and sess
    def get_model_graph_and_sess(self):
        #
        return self._graph, self._sess
        #
            
if __name__ == '__main__':
    
    pass

        
    
    
