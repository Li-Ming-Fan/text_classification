# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 12:57:35 2018

@author: limingfan

"""

import os
import time
import numpy as np
import json

import tensorflow as tf
from tensorflow.python.framework import graph_util

from data_set import Dataset


class ModelSettings(object):
    def __init__(self, vocab, model_tag = 'None'):
        
        self.num_classes = 2
        
        # model        
        self.min_seq_len = 5  #
        self.att_dim = 256
        #
        self.model_tag = model_tag
        #
        
        # vocab
        self.vocab = vocab
        #
        self.emb_tune = 0  # 1 for tune, 0 for not
        self.keep_prob = 0.7
        
        # train
        self.learning_rate = 0.001         
        self.ratio_decay = 0.9
        self.patience_decay = 1000
        
        self.num_epochs = 100     
        self.batch_size = 64
        self.batch_size_eval = 128 
        self.patience_stop = 5000
        
        self.save_per_batch = 100
        self.valid_per_batch = 100
        #
        
    def trans_info_to_dict(self):
                
        info_dict = {}
        for name,value in vars(self).items():
            if not isinstance(value, (int, float, str, bool, list, dict, tuple)):
                continue
            info_dict[str(name)] = value        
        return info_dict


class ModelWrapper(object):
    """
        This class is written as TASK-INDEPENDENT as possible,
        yet is kept as much readibility as possible.
        
        Only a few snippets need to be modified for different tasks.
        Mainly of these snippets are about the inputs/outputs,
        except that the one in __init__ is about the model graph.
        
        List of these snippets:
        0, __init__ : model graph
        1, prepare_for_prediction : inputs/outputs
        2, predict : data preprocess, inputs/outputs
        3, feed_data : feed data (inputs)
        4, prepare_graph_and_sess : inputs/outputs
        
        (These can also be implemented task-independent. In progress.)
        
        Prerequisites:
        1, feed data: arranged as (x_batch, y_batch) for a single batch
        2, implement data batching functions:
            valid_batches = Dataset.do_batching_data(valid_data, self.settings.batch_size_eval)
            valid_batches = Dataset.do_normalizing_batches(valid_batches, self.settings) # min_seq_len
        3, metric: accuracy-like metric
        
        # namedtuple
        from collections import namedtuple
        Settings = namedtuple('Settings', ['min_seq_len'])
        settings = Settings(5)
        
    """
    def __init__(self, settings):

        # model graph
        """ task-dependent
        """
        if settings.model_tag == 'cnn':
            from model_graph_cnn import build_graph
        elif settings.model_tag == 'rnn':
            from model_graph_rnn import build_graph
        elif settings.model_tag == 'csm':
            from model_graph_csm import build_graph
        else:
            assert False, 'ERROR: NOT supported model_tag: %s' % settings.model_tag
        #
        """ The codes after this in this function are task-independent,
            so they could work without modification for different tasks.
        """
        # config
        self.settings = settings
        self._build_graph = build_graph
        
        # model saving
        self.model_dir = './model_' + self.settings.model_tag
        self.model_name = 'model_' + self.settings.model_tag
        self.pb_file = os.path.join(self.model_dir + '_best', self.model_name + '.pb')
        
        # log
        self.log_dir = './log'
        str_datetime = str(time.strftime("%Y-%m-%d-%H-%M"))
        
        self.log_path = os.path.join(self.log_dir, 
                                     self.model_name + "_" + str_datetime +".txt")

        # session info
        self.sess_config = tf.ConfigProto()
        self.sess_config.gpu_options.allow_growth = True
        
    def log_info(self, str_info):
        """ task-independent
        """        
        if not os.path.exists(self.log_dir): os.mkdir(self.log_dir)
        with open(self.log_path, 'a', encoding='utf-8') as fp:
            if len(str_info.strip()) == 0:
                fp.write("\n")
            else:
                fp.write(time.strftime("%Y-%m-%d, %H:%M:%S: ") + str_info + "\n")
        #
        
    # prediction
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
            """ only the input/output variables
                are task-dependent
            """
            self._x = self._graph.get_tensor_by_name('input_x:0')
            #
            self._logits = self._graph.get_tensor_by_name('score/logits:0')
            # 
            print('Graph loaded for prediction')
            #
        #
        self._sess = tf.Session(graph = self._graph, config = self.sess_config)
        #
    
    def predict(self, list_texts):
        """ task-dependent
        
            data: list of text string, 
            returned result: logits        
        """
        list_converted = Dataset.preprocess_wholesuitely(self.settings.vocab, list_texts)
        data, seq_len = Dataset.do_padding_data_converted(list_converted, self.settings.min_seq_len)
        #
        feed_dict = {self._x: data}
        logits = self._sess.run(self._logits, feed_dict = feed_dict)
        
        return logits
    
    # train and validation
    def feed_data(self, x_batch, y_batch):
        """ task-dependent
        
            keep the arguments not changed as x_batch, y_batch,
            arrang the data to this form.
            feed data, a single batch
        """
        feed_dict = { self._input_x: x_batch,
                      self._input_y: y_batch }
        return feed_dict

    def prepare_graph_and_sess(self):

        # graph
        self._graph = tf.Graph()
        with self._graph.as_default():
            #
            self._learning_rate = tf.get_variable("lr", shape=[], dtype=tf.float32, trainable=False)
            #
            """ inputs/outputs, task-dependent            
            """
            self._input_x = tf.placeholder(tf.int32, [None, None], name='input_x')
            self._input_y = tf.placeholder(tf.int64, [None], name='input_y')
            #
            self._inputs = self._input_x, self._input_y, self._learning_rate
            #
            self._outputs = self._build_graph(self.settings, self._inputs)  # model graph
            #
            self._logits, self._metric, self._loss, self._optim = self._outputs
            #
            """ The codes after this in this file are task-independent,
                could work without modification for different tasks.
                Single GPU mode.
            """
            #
            # all params
            self.all_params = tf.trainable_variables()
            #
            # save info
            self._saver = tf.train.Saver()
            self._saver_best = tf.train.Saver()
            
            # sess
            self._sess = tf.Session(graph=self._graph, config = self.sess_config)

            # initialize the model
            self._sess.run(tf.global_variables_initializer())
            
            # params count
            self.param_num = sum([np.prod(self._sess.run(tf.shape(v))) for v in self.all_params])
            #
            str_info = 'Graph built, there are %d parameters in the model' % self.param_num
            self.log_info(str_info)
            print(str_info)
            #
            info_dict = self.settings.trans_info_to_dict()
            str_info = json.dumps(info_dict, ensure_ascii = False)
            self.log_info(str_info)
            print(str_info)
            #
    #
    def evaluate(self, eval_batches):
        
        data_len = 0        
        total_loss = 0.0
        total_acc = 0.0  # accuracy, metric
        curr = 0
        for x_batch, y_batch in eval_batches:
            batch_len = len(x_batch)
            data_len += batch_len
            feed_dict = self.feed_data(x_batch, y_batch)            
            loss, acc = self._sess.run([self._loss, self._metric], feed_dict = feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len            
            curr += 1
            
            #cls_pred = self.sess.run(self.y_pred_cls, feed_dict = feed_dict)
            #print(cls_pred)
            
        return total_loss / data_len, total_acc / data_len
    
    def train_and_valid(self, train_data, valid_data):
        
        if not os.path.exists(self.model_dir): os.mkdir(self.model_dir)
        if not os.path.exists(self.model_dir + '_best'): os.mkdir(self.model_dir + '_best')
        
        print('Training and evaluating...')
        #start_time = time.time()
        total_batch = 0
        best_acc_val = 0.0 
        last_improved = 0
        
        lr = self.settings.learning_rate
        with self._graph.as_default():
            self._sess.run(tf.assign(self._learning_rate, tf.constant(lr, dtype=tf.float32)))
        
        valid_batches = Dataset.do_batching_data(valid_data, self.settings.batch_size_eval)
        valid_batches = Dataset.do_normalizing_batches(valid_batches, self.settings)
        
        print('Creating model for evaluation ...')
        config_e = self.settings
        config_e.keep_prob = 1.0
        model_e = ModelWrapper(config_e)
        model_e.prepare_graph_and_sess()
        
        flag_stop = False
        for epoch in range(self.settings.num_epochs):
            print('Epoch: %d, training ...' % (epoch + 1) )
            
            train_batches = Dataset.do_batching_data(train_data, self.settings.batch_size)
            train_batches = Dataset.do_normalizing_batches(train_batches, self.settings)
            
            for x_batch, y_batch in train_batches:
                feed_dict = self.feed_data(x_batch, y_batch)
                
                # valid
                if total_batch % self.settings.valid_per_batch == 0:
                    # load
                    ckpt = tf.train.get_checkpoint_state(self.model_dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        model_e._saver.restore(model_e._sess, ckpt.model_checkpoint_path)
                    #print('evaluation-model created')                                        
                    loss_val, acc_val = model_e.evaluate(valid_batches)                    
                    #print('evaluated')
                    
                    # save best
                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        last_improved = total_batch
                        model_e._saver_best.save(model_e._sess,
                                                 os.path.join(model_e.model_dir + '_best', model_e.model_name),
                                                 global_step = total_batch)
                        
                        # pb
                        constant_graph = graph_util.convert_variables_to_constants(
                                model_e._sess, model_e._sess.graph_def, output_node_names = ['score/logits'])
                        with tf.gfile.FastGFile(model_e.pb_file, mode='wb') as f:
                            f.write(constant_graph.SerializeToString())
                        #
                    
                    # stop
                    if total_batch - last_improved >= self.settings.patience_stop:
                        str_info = "no improvement for a long time, stop optimization at curr_batch: %d" \
                                    % total_batch
                        self.log_info(str_info)
                        print(str_info)
                        #
                        flag_stop = True
                        break # for batch
                    
                    # decay
                    if total_batch - last_improved >= self.settings.patience_decay:
                        lr *= self.settings.ratio_decay
                        with self._graph.as_default():
                            self._sess.run(tf.assign(self._learning_rate, tf.constant(lr, dtype=tf.float32)))
                        last_improved = total_batch
                        #
                        str_info = 'learning_rate DECAYED at total_batch: %d' % total_batch
                        self.log_info(str_info)
                        print(str_info)
                    
                    # time
                    # time_cost = time.time() - start_time
                    #
                    str_info = 'loss, metric, best_metric: %.6f, %.4f, %.4f' % (loss_val,
                                                                                acc_val, best_acc_val)
                    self.log_info(str_info)
                    # print(str_info)
                    #
                    str_info = 'curr_batch: %d, lr: %f' % (total_batch, lr)
                    self.log_info(str_info)
                    # print(str_info)

                # optim
                self._sess.run(self._optim, feed_dict = feed_dict)
                total_batch += 1
                    
                # save
                if total_batch % self.settings.save_per_batch == 0:
                    #s = session.run(merged_summary, feed_dict=feed_dict)
                    #writer.add_summary(s, total_batch)
                    loss, acc = self._sess.run([self._loss, self._metric], feed_dict = feed_dict)
                    #
                    str_info = "epoch: %d" % (epoch + 1)
                    self.log_info(str_info)
                    # print(str_info)
                    #
                    str_info = "loss, metric of train: %f, %f" % (loss, acc)
                    self.log_info(str_info)
                    self.log_info("")
                    # print(str_info)
                    # print()
                    
                    self._saver.save(self._sess,
                                    os.path.join(self.model_dir, self.model_name),
                                    global_step = total_batch)
                #
            #
            if flag_stop: break # for epoch
            #
            

            
            
