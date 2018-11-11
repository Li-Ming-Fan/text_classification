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
        
        # settings
        self.set_model_settings(settings)
            
        # session info
        self.sess_config = tf.ConfigProto()
        self.sess_config.gpu_options.allow_growth = True
        
    def set_model_settings(self, settings):
        
        self.settings = settings
        #
        # super().__init__(settings.is_train)
        for key in settings.__dict__.keys():
            # print(key + ': ' + str(self.__dict__[key]) )                   
            self.__dict__[key] = settings.__dict__[key]
            # print(key + ': ' + str(self.__dict__[key]) ) 
          
    def log_info(self, str_info):     
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
        #
        feed_dict = {}
        for idx in range(self._inputs_predict_num):
            feed_dict[self._inputs_predict[idx] ] = x_batch[idx]
        #
        outputs = self._sess.run(self._outputs_predict, feed_dict = feed_dict)        
        return outputs
    
    def predict_from_batch(self, x_batch):
        """
        """
        # x_batch = Dataset.preprocess_for_prediction(inputs_data, self.settings) # a single batch        
        #
        feed_dict = {}
        for idx in range(self._inputs_predict_num):
            feed_dict[self._inputs_predict[idx] ] = x_batch[idx]
        #
        outputs = self._sess.run(self._outputs_predict, feed_dict = feed_dict)        
        return outputs
    
    #
    @staticmethod
    def build_batch_iter(settings, tfrecord_filelist, dataset):
        
        settings.tfrecord_filelist = tfrecord_filelist
        return dataset.get_batched_data(
                settings.tfrecord_filelist,
                dataset.single_example_parser,
                padded_shapes = settings.padded_shapes,
                batch_size = settings.batch_size,                
                num_epochs = settings.num_epochs,
                buffer_size = settings.buffer_size)  
        
    @staticmethod
    def build_graph_all(settings, batch_iter, model_graph_builder):
        
        global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                      initializer=tf.constant_initializer(0), trainable=False)
        lr = tf.get_variable("lr", shape=[], dtype=tf.float32, trainable = False)
        #        
        result, metric, loss_infer = model_graph_builder(settings, batch_iter)
        #
        # all trainable vars
        trainable_vars = tf.trainable_variables()  # print(trainable_vars)
        #
        
        #
        if settings.reg_lambda > 0.0:
            loss_reg = tf.add_n([ tf.nn.l2_loss(v) for v in trainable_vars \
                                 if 'bias' not in v.name ])
            loss_reg = tf.multiply(loss_reg, settings.reg_lambda)
            loss = tf.add(loss_infer, loss_reg, "loss")
        else:
            loss = tf.identity(loss_infer, "loss")
        #
        if settings.keep_prob < 1.0:
            print("total loss tensor:") 
            print(loss)
            print()
        #
        # Optimizer
        opt = tf.train.AdamOptimizer(learning_rate = lr)
        #
        if settings.grad_clip > 0.0:
            grads = opt.compute_gradients(loss)
            gradients, variables = zip(*grads)
            grads, _ = tf.clip_by_global_norm(gradients, settings.grad_clip)
            train_op = opt.apply_gradients(zip(grads, variables),
                                           global_step = global_step)
        else:
            train_op = opt.minimize(loss)
        #   
        return train_op, trainable_vars
        #
        
    #
    def _prepare_for_train_or_eval(self, tfrecord_filelist, dir_ckpt, flag_log_info):

        # graph
        self._graph = tf.Graph()
        with self._graph.as_default():
            #
            self._batch_iter = self.build_batch_iter(self.settings,
                                                     tfrecord_filelist,
                                                     Dataset)
            a, b = self.build_graph_all(self.settings,
                                        self._batch_iter,
                                        self.model_graph_builder)
            self._train_op, self.trainable_vars = a, b
            #
            self._lr = self._graph.get_tensor_by_name("lr:0")
            self._loss_tensor = self._graph.get_tensor_by_name(self.loss_name)
            self._metric_tensor = self._graph.get_tensor_by_name(self.metric_name)            
            #
            self._inputs_train = []
            for item in self.inputs_train_name:
                tensor = self._graph.get_tensor_by_name(item)
                self._inputs_train.append(tensor)
            #
            # print(tf.all_variables())
         
            # save info
            self._saver = tf.train.Saver()
            self._saver_best = tf.train.Saver()
            
            # sess
            self._sess = tf.Session(graph = self._graph, config = self.sess_config)

            # initialize the model
            self._sess.run(tf.global_variables_initializer())
            
            # params count
            self.param_num = sum([np.prod(self._sess.run(tf.shape(v))) \
                                  for v in self.trainable_vars])
            #
            if flag_log_info:
                str_info = 'Graph built, there are %d parameters in the model' % self.param_num
                self.log_info(str_info)
                print(str_info)
                #
                info_dict = self.settings.trans_info_to_dict()
                str_info = json.dumps(info_dict, ensure_ascii = False)
                self.log_info(str_info)
                # print(str_info)
                #
        #
        # load
        if dir_ckpt is not None:
            ckpt = tf.train.get_checkpoint_state(dir_ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                self._saver.restore(self._sess, ckpt.model_checkpoint_path)
                
    #
    def evaluate(self, eval_tfrecords, dir_ckpt = None, flag_log_info = False):
        
        if dir_ckpt is None: dir_ckpt = self.model_dir
        
        print('Creating model for evaluation ...')
        self._prepare_for_train_or_eval(eval_tfrecords, dir_ckpt, flag_log_info)
        
        total_batch = 0
        total_metric = 0.0  # accuracy, metric
        total_loss = 0.0
        batch_size = self.batch_size
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = self._sess, coord = coord)
        try:
            while not coord.should_stop():
                #
                metric, loss = self._sess.run([self._metric_tensor,
                                               self._loss_tensor] )
                total_batch += 1
                total_loss += loss * batch_size
                total_metric += metric * batch_size

        except tf.errors.OutOfRangeError:
            # print("done training")
            pass
        finally:
            coord.request_stop()
        coord.join(threads)
            
        #
        data_len = total_batch * batch_size
        loss_mean = total_loss / data_len
        metric_mean = total_metric / data_len            
        #
        str_info = "loss_eval, metric_eval: %.6f, %.4f" % (loss_mean, metric_mean)
        self.log_info(str_info)
        print(str_info)
        #
        return loss_mean, metric_mean
    
    def train_and_valid(self, train_tfrecords, valid_tfrecords, dir_ckpt = None):
        """ 
        """        
        if not os.path.exists(self.model_dir): os.mkdir(self.model_dir)
        if not os.path.exists(self.model_dir + '_best'): os.mkdir(self.model_dir + '_best')
        
        #start_time = time.time()
        total_batch = 0
        best_metric_val = 0.0 
        last_improved = 0

        #
        print("Creating model for training ...")
        self._prepare_for_train_or_eval(train_tfrecords, dir_ckpt, True)
        #        
        lr = self.learning_rate_base
        with self._graph.as_default():
            self._sess.run(tf.assign(self._lr, tf.constant(lr, dtype=tf.float32)))
        #
        
        # for eval
        settings_e = self.settings
        settings_e.keep_prob = 1.0
        settings_e.batch_size = settings_e.batch_size_eval
        settings_e.num_epochs = 1
        
        #
        print('Training ...')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = self._sess, coord = coord)
        try:
            while not coord.should_stop():
                #
                # valid
                if total_batch % self.valid_per_batch == 0:
                    #                    
                    model_e = ModelWrapper(settings_e)                                        
                    loss_val, metric_val = model_e.evaluate(valid_tfrecords)            
                    
                    # save best
                    if metric_val >= best_metric_val:  # >=
                        best_metric_val = metric_val
                        last_improved = total_batch
                        model_e._saver_best.save(model_e._sess,
                                                 os.path.join(model_e.model_dir + '_best',
                                                              model_e.model_name),
                                                 global_step = total_batch)
                        
                        # pb
                        constant_graph = graph_util.convert_variables_to_constants(
                                model_e._sess, model_e._sess.graph_def,
                                output_node_names = self.pb_outputs_name)
                        with tf.gfile.FastGFile(model_e.pb_file, mode='wb') as f:
                            f.write(constant_graph.SerializeToString())
                        #
                    
                    # decay
                    if total_batch - last_improved >= self.patience_decay:
                        lr *= self.ratio_decay
                        with self._graph.as_default():
                            self._sess.run(tf.assign(self._lr, tf.constant(lr, dtype=tf.float32)))
                        last_improved = total_batch
                        #
                        str_info = 'learning_rate DECAYED at total_batch: %d' % total_batch
                        self.log_info(str_info)
                        print(str_info)
                    
                    # time
                    # time_cost = time.time() - start_time
                    #
                    str_info = 'loss, metric, best_metric: %.6f, %.4f, %.4f' % (loss_val,
                                                                                metric_val,
                                                                                best_metric_val)
                    self.log_info(str_info)
                    # print(str_info)
                    #
                    str_info = 'curr_batch: %d, lr: %f' % (total_batch, lr)
                    self.log_info(str_info)
                    self.log_info("")
                    # print(str_info)

                # optim
                metric, loss, _ = self._sess.run([self._metric_tensor,
                                                  self._loss_tensor,
                                                  self._train_op])
                total_batch += 1
                #
                
                # save
                if total_batch % self.save_per_batch == 0:
                    #s = session.run(merged_summary, feed_dict=feed_dict)
                    #writer.add_summary(s, total_batch)
                    #
                    str_info = "loss_train, metric_train: %f, %f" % (loss, metric)
                    self.log_info(str_info)
                    # self.log_info("")
                    # print(str_info)
                    # print()
                    
                    self._saver.save(self._sess,
                                    os.path.join(self.model_dir, self.model_name),
                                    global_step = total_batch)
                    #
            #
            #
        except tf.errors.OutOfRangeError:
            #
            str_info = "training ended after %d batches" % total_batch
            self.log_info(str_info)
            self.log_info("")
            # print(str_info)
            # print()
            #
        finally:
            coord.request_stop()
        coord.join(threads)
        #        
    
    # for debug
    def get_model_graph_and_sess(self):
        """ for debug
        """
        return self._graph, self._sess
        #
        
    def feed_data_train(self, data_batch):        
        feed_dict = {}
        for idx, item in enumerate(self._inputs_train):
            feed_dict[item] = data_batch[idx]
        return feed_dict
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
  