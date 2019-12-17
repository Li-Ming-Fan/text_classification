# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:05:25 2018

@author: limingfan

"""

import tensorflow as tf

from Zeras.model_baseboard import ModelBaseboard


def cnn_rnf_layer(seq, seq_len, R, rnn_size, padding='valid', scope='cnn_rnf'):
    """ seq: [B, T, D]
    """
    seq_shape = seq.get_shape().as_list()

    T = tf.reduce_max(seq_len)
    D = seq_shape[2]
    U = T-R+1
    
    # if U <= 0, there will be InvalidArgumentError ! 
    
    # --> [ B*(T-R+1), R, D]
    chunks_ta = tf.TensorArray(size = U, dtype = tf.float32)    
    time = tf.constant(0)
    def condition(time, chunks_ta_d):
        return tf.less(time, U)
    
    def body(time, chunks_ta_d):
        chunk = seq[:, time:time+R, :]        
        chunks_ta_d = chunks_ta_d.write(time, chunk)        
        return (time + 1, chunks_ta_d)
        
    t, seq_ta = tf.while_loop(cond = condition, body = body,
                              loop_vars = (time, chunks_ta) )
    
    seq_s = seq_ta.stack()                      # [U, B, R, D]
    seq_s = tf.transpose(seq_s, [1, 0, 2, 3])   # [B, U, R, D]
    seq_s = tf.reshape(seq_s, [-1, R, D])       # [B*U, R, D]
    
    # go through rnn
    # seq_s_len = tf.reduce_max(tf.reduce_max(seq_s, 2), 1) * 0 + 1
    seq_s_len = seq_s[:,0,0] * 0 + 1
    seq_s_len = tf.multiply(seq_s_len, R)
    
    weight_initializer = tf.truncated_normal_initializer(stddev = 0.01)
    
    cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer = weight_initializer)

    outputs, last_state = tf.nn.dynamic_rnn(cell = cell,
                                            inputs = seq_s,                                             
                                            sequence_length = seq_s_len,
                                            time_major = False,
                                            dtype = tf.float32,
                                            scope = scope)
    # [B*U, rnn_size] --> [B, U, rnn_size]
    h = tf.reshape(last_state.h, [-1, U, rnn_size])
    
    return h
    
#

class ModelRNF(ModelBaseboard):
    """
    """
    def __init__(self, settings):
        """
        """
        super(ModelRNF, self).__init__(settings)

        # input/output tensors
        self.pb_input_names = {"input_x": "input_x:0"}
        self.pb_output_names = {"logits": "vs_gpu/score/logits:0"}
        self.pb_save_names = ["vs_gpu/score/logits"]
        #
        self.debug_tensor_names = ["vs_gpu/score/logits:0",
                                   "vs_gpu/score/logits:0"]
    #
    def build_placeholder(self):
        """
        """         
        input_x = tf.placeholder(tf.int32, [None, None], name='input_x')
        input_y = tf.placeholder(tf.int64, [None], name='input_y')
        
        #        
        print(input_x)
        print(input_y)
        #
        input_tensors = {"input_x": input_x}
        label_tensors = {"input_y": input_y}
        #
        return input_tensors, label_tensors

    def build_inference(self, input_tensors):
        """
        """
        settings = self.settings
        input_x = input_tensors["input_x"]
        #
        keep_prob = tf.get_variable("keep_prob", shape=[], dtype=tf.float32, trainable=False)
        #   
        with tf.device('/cpu:0'):
            emb_mat = tf.get_variable('embedding',
                                      [settings.vocab.size(), settings.vocab.emb_dim],
                                      initializer=tf.constant_initializer(settings.vocab.embeddings),
                                      trainable = settings.emb_tune,
                                      dtype=tf.float32)
            seq_emb = tf.nn.embedding_lookup(emb_mat, input_x)
            
            seq_mask = tf.cast(tf.cast(input_x, dtype = tf.bool), dtype = tf.int32)
            seq_len = tf.reduce_sum(seq_mask, 1)
    
        with tf.name_scope("cnn"):
            #
            conv1_5 = tf.layers.conv1d(seq_emb, 128, 5, padding='same', name='conv1_5')
            conv1_3 = tf.layers.conv1d(seq_emb, 128, 3, padding='same', name='conv1_3')
            conv1_2 = tf.layers.conv1d(seq_emb, 128, 2, padding='same', name='conv1_2')
            
            # 有[PAD]影响计算结果的问题，需要解决            
            # max_pooling, 最大值采提
            feat1 = tf.reduce_max(conv1_5, reduction_indices=[1], name='feat1')
            feat2 = tf.reduce_max(conv1_3, reduction_indices=[1], name='feat2')
            feat3 = tf.reduce_max(conv1_2, reduction_indices=[1], name='feat3')
            
            #
            crnf_5 = cnn_rnf_layer(seq_emb, seq_len, 5, 128, padding='valid', scope='cnn_rnf')
            
            # 有[PAD]影响计算结果的问题，需要解决            
            # max_pooling, 最大值采提
            feat_r = tf.reduce_max(crnf_5, 1)
            
            feat = tf.concat([feat1, feat2, feat3, feat_r], 1)
    
        with tf.name_scope("score"):
            #
            fc = tf.nn.dropout(feat, keep_prob)
            fc = tf.layers.dense(fc, 128, name='fc1')            
            fc = tf.nn.relu(fc)
            
            fc = tf.nn.dropout(fc, keep_prob)
            logits = tf.layers.dense(fc, settings.num_classes, name='fc2')
            
            normed_logits = tf.nn.softmax(logits, name='logits')
            
        #
        print(normed_logits)
        #
        output_tensors = {"normed_logits": normed_logits,
                          "logits": logits }
        #   
        return output_tensors
    
    def build_loss_and_metric(self, output_tensors, label_tensors):
        """
        """
        settings = self.settings

        logits = output_tensors["logits"]
        input_y = label_tensors["input_y"] 
                
        y_pred_cls = tf.argmax(logits, 1, name='pred_cls')
        
        with tf.name_scope("loss"):
            #
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,
                                                                           labels = input_y)
            loss = tf.reduce_mean(cross_entropy, name = 'loss')
    
        with tf.name_scope("accuracy"):
            #
            correct_pred = tf.equal(input_y, y_pred_cls)
            acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = 'metric')
            
        #
        print(loss)
        print(acc)
        #
        loss_and_metric = {"loss_model": loss,
                           "metric": acc}
        #
        return loss_and_metric
        #
        
