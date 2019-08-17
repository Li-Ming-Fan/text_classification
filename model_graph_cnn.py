#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 09:30:05 2019

@author: li-ming-fan
"""

import tensorflow as tf


def conv1d_layer(seq_emb, params):
    '''define a convolutional layer with params'''
    #
    # 输入数据维度为 3-D tensor: [batch_size, time, channels]
    #
    # params = [filters, kernel_size, padding, name]
    #
    # conv = tf.layers.conv1d(seq_emb, 128, 5, padding='same', name='conv1_5')
    output = tf.layers.conv1d(seq_emb,
                              params[0], params[1],
                              padding=params[2], name=params[3])
    #
    # kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    # kernel_initializer = tf.contrib.layers.xavier_initializer()    
    # bias_initializer = tf.constant_initializer(value=0.0)
    #
    return output
    #
    
def module_cnn(seq_emb, scope):
    """
    """
    with tf.variable_scope(scope):
        
        conv1_3 = conv1d_layer(seq_emb, [128, 3, 'valid', 'conv1_3'])
        feat1_3 = tf.reduce_max(conv1_3, axis=1, name="feat1_3")
        
        conv1_2 = conv1d_layer(seq_emb, [128, 2, 'valid', 'conv1_2'])
        feat1_2 = tf.reduce_max(conv1_2, axis=1, name="feat1_2")
        
        conv1_1 = conv1d_layer(seq_emb, [128, 1, 'valid', 'conv1_1'])
        feat1_1 = tf.reduce_max(conv1_1, axis=1, name="feat1_1")
        
        feat = tf.concat([feat1_1, feat1_2, feat1_3], axis=-1, name="feat")
    
    return feat
    


class ModelGraph():
    
    @staticmethod
    def build_placeholder(settings):
        
        input_x = tf.placeholder(tf.int32, [None, None], name='input_x')
        input_y = tf.placeholder(tf.int64, [None], name='input_y')
        
        #        
        print(input_x)
        print(input_y)
        #
        input_tensors = (input_x, )
        label_tensors = (input_y, )
        #
        return input_tensors, label_tensors
    
    @staticmethod
    def build_inference(settings, input_tensors):
        
        input_x = input_tensors[0]

        #
        keep_prob = tf.get_variable("keep_prob", shape=[], dtype=tf.float32, trainable=False)
        #   
        with tf.device('/cpu:0'):
            emb_mat = tf.get_variable('embedding',
                                      [settings.vocab.size(), settings.vocab.emb_dim],
                                      initializer=tf.constant_initializer(settings.vocab.embeddings),
                                      trainable = settings.emb_tune,
                                      dtype=tf.float32)
            #
            seq_emb = tf.nn.embedding_lookup(emb_mat, input_x)
            #
    
        with tf.variable_scope("cnn"):
            #
            feat = module_cnn(seq_emb, "input_x")
            #
    
        with tf.variable_scope("score"):
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
        output_tensors = normed_logits, logits
        #   
        return output_tensors
    
    @staticmethod
    def build_loss_and_metric(settings, output_tensors, label_tensors):
        
        normed_logits, logits = output_tensors
        input_y = label_tensors[0]
        
        
        y_pred_cls = tf.argmax(logits, 1, name='pred_cls')
        
        with tf.variable_scope("loss"):
            #
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,
                                                                           labels = input_y)
            loss = tf.reduce_mean(cross_entropy, name = 'loss')
    
        with tf.variable_scope("accuracy"):
            #
            correct_pred = tf.equal(input_y, y_pred_cls)
            acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = 'metric')
            
        #
        print(loss)
        print(acc)
        #
        return loss, acc
        #
        