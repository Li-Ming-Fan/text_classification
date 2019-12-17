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
    """ settings.warmup_steps
        settings.decay_steps
        settings.decay_rate
        settings.staircase
        
        learning_rate_schedule = get_warmup_and_exp_decayed_lr
        self.learning_rate_tensor = self.learning_rate_schedule(self.settings, self.global_step)
    """
    learning_rate = tf.constant(value = settings.learning_rate_base,
                                shape = [], dtype = tf.float32)
        
    if settings.warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(settings.warmup_steps, dtype=tf.int32)
        
        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
        
        step_surplus = global_steps_int - warmup_steps_int
        learning_rate = tf.train.exponential_decay(learning_rate,
                                                   step_surplus,
                                                   settings.decay_steps,
                                                   settings.decay_rate,
                                                   settings.staircase)
        
        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = settings.learning_rate_base * warmup_percent_done
        
        learning_rate = tf.cond(global_steps_int < warmup_steps_int,
                                lambda: warmup_learning_rate,
                                lambda: learning_rate)
    #
    else:
        learning_rate = tf.train.exponential_decay(learning_rate,
                                                   global_step,
                                                   settings.decay_steps,
                                                   settings.decay_rate,
                                                   settings.staircase)
    #
    return learning_rate
    #
    
def get_adam_optimizer(settings, learning_rate_tensor_or_value):
    """ 
        customized_optimizer = get_adam_optimizer
        self._opt = self.customized_optimizer(self.settings, self.learning_rate_tensor)
        
        grad_and_vars = self._opt.compute_gradients(self.loss_train_tensor)
        self.train_op = self._opt.apply_gradients(grad_and_vars, global_step = self.global_step)
    """
    opt = tf.train.AdamOptimizer(learning_rate_tensor_or_value, beta1 = settings.momentum)
    return opt
    #
    
    
#
class ModelWrapper():
    
    def __init__(self, settings, model_graph,
                 learning_rate_schedule = get_warmup_and_exp_decayed_lr,
                 customized_optimizer = get_adam_optimizer):
        #
        self.learning_rate_schedule = get_warmup_and_exp_decayed_lr
        self.customized_optimizer = customized_optimizer
        #
        self.set_model_settings(settings)
        self.model_graph = model_graph
        #
        
    def set_model_settings(self, settings):
        #
        # settings
        self.settings = settings
        self.num_gpu = len(settings.gpu_available.split(","))
        self.vs_str_multi_gpu = "vs_gpu"
        #
        # session info
        self.sess_config = tf.ConfigProto(log_device_placement = settings.log_device,
                                          allow_soft_placement = settings.soft_placement)
        self.sess_config.gpu_options.allow_growth = settings.gpu_mem_growth
        #
            
    # predict
    def prepare_for_prediction_with_pb(self, pb_file_path = None):
        #
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
            #
            self._inputs_pred = {}
            for tensor_tag, tensor_name in self.model_graph.pb_input_names.items():
                tensor = self._graph.get_tensor_by_name(tensor_name)
                self._inputs_pred[tensor_tag] = tensor
            #
            self._outputs_pred = {}
            for tensor_tag, tensor_name in self.model_graph.pb_output_names.items():
                tensor = self._graph.get_tensor_by_name(tensor_name)
                self._outputs_pred[tensor_tag] = tensor
            #
            # self._inputs_pred_num = len(self._inputs_pred)
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
                self._opt = tf.train.MomentumOptimizer(self.learning_rate_tensor, self.settings.momentum, use_nesterov=True)
            elif self.settings.optimizer_type == 'adam':
                self._opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate_tensor, beta1 = self.settings.momentum)
            elif self.settings.optimizer_type == 'customized':
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
                loss_metric_tensors = self.model_graph.build_loss_and_metric(self.settings, output_tensors, label_tensors)
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
            if self.settings.reg_lambda > 0.0:
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
            self.settings.logger.info(str_info)
            print(str_info)
            #
            tf_shapes = [tf.shape(v) for v in self.trainable_vars]
            shapes_v = self._sess.run(tf_shapes)
            params_v = [np.prod(item) for item in shapes_v]
            self.param_num = sum(params_v)
            #
            str_info = 'there are %d parameters in the model' % self.param_num
            self.settings.logger.info(str_info)
            print(str_info)
            #
            print()
            for idx in range(self.num_vars):
                print(self.trainable_vars[idx])
                print(params_v[idx])
            print()
            #
            
            # outputs eval
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
            self._debug_tensors = []
            for name in self.model_graph.debug_tensor_names:
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
                self._opt = tf.train.MomentumOptimizer(self.learning_rate_tensor, self.settings.momentum, use_nesterov=True)
            elif self.settings.optimizer_type == 'adam':
                self._opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate_tensor, beta1 = self.settings.momentum)
            elif self.settings.optimizer_type == 'customized':
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
                        output_tensors = self.model_graph.build_inference(self.settings,
                                                                          inputs[gid])
                        loss_metric_tensors = self.model_graph.build_loss_and_metric(self.settings,
                                                                                     output_tensors,
                                                                                     labels[gid])
                        #
                        tf.get_variable_scope().reuse_variables()
                        #
                        loss = loss_metric_tensors["loss_model"]
                        if self.settings.use_metric_in_graph:
                            metric = loss_metric_tensors["metric"]
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
                for item in self.settings.reg_exclusions:
                    if item in v.name: return True
                return False
            #
            if self.settings.reg_lambda > 0.0:
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
            grads_summed = ModelWrapper.sum_up_gradients(grads_bundles)
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
            self.settings.logger.info(str_info)
            print(str_info)
            #
            tf_shapes = [tf.shape(v) for v in self.trainable_vars]
            shapes_v = self._sess.run(tf_shapes)
            params_v = [np.prod(item) for item in shapes_v]
            self.param_num = sum(params_v)
            #
            str_info = 'there are %d parameters in the model' % self.param_num
            self.settings.logger.info(str_info)
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
            self.settings.logger.info(str_info)
            print(str_info)
        else:
            str_info = 'loading ckpt failed: ckpt loading from %s' % dir_ckpt
            self.settings.logger.info(str_info)
            print(str_info)

    #
    # predict, pb
    def load_ckpt_and_save_pb_file(self, dir_ckpt):
        #
        is_train = self.settings.is_train
        self.settings.is_train = False       #
        #
        model = ModelWrapper(self.settings, self.model_graph)
        model.prepare_for_train_and_valid_single_gpu(dir_ckpt)         # loaded here 
        model.assign_dropout_keep_prob(1.0)
        #
        print("model.model_graph.pb_save_names:")
        print(model.model_graph.pb_save_names)
        print("if the above is not right, please assign the right value")
        #
        pb_file = os.path.join(dir_ckpt, "model_frozen.pb")
        #
        constant_graph = graph_util.convert_variables_to_constants(
                model._sess, model._sess.graph_def,
                output_node_names = model.model_graph.pb_save_names)
        with tf.gfile.GFile(pb_file, mode='wb') as f:
            f.write(constant_graph.SerializeToString())
        #
        str_info = 'pb_file saved: %s' % pb_file
        self.settings.logger.info(str_info)
        #
        self.settings.is_train = is_train           #
        #

    # graph and sess
    def get_model_graph_and_sess(self):
        #
        return self._graph, self._sess
        #
            
if __name__ == '__main__':
    
    pass

        
    
    
