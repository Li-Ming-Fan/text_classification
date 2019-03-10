# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 15:33:42 2018

@author: limingfan

"""

import tensorflow as tf

# from zoo_layers import rnn_layer
from zoo_layers import gru_layer as rnn_layer
from zoo_layers import att_pool_layer


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
                                      trainable = settings.emb_tune)
            seq_emb = tf.nn.embedding_lookup(emb_mat, input_x)
            
            seq_mask = tf.cast(tf.cast(input_x, dtype = tf.bool), dtype = tf.int32)
            seq_len = tf.reduce_sum(seq_mask, 1)
    
        with tf.name_scope("rnn"):
            
            seq_e = rnn_layer(seq_emb, seq_len, 128, keep_prob,
                              activation = tf.nn.relu, concat = True, scope = 'bi-lstm-1')
            
            # attention-pooling, 注意力加权采提
            #
            B = tf.shape(seq_e)[0]
            query = tf.get_variable("query", [settings.att_dim],
                                    initializer = tf.ones_initializer())
            query = tf.tile(tf.expand_dims(query, 0), [B, 1])
    
            feat = att_pool_layer(query, seq_e, seq_mask, settings.att_dim,
                                  keep_prob, scope="att_pooling")
            
            #feat = seq_e[:,-1,:]
    
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
        output_tensors = normed_logits, logits
        #   
        return output_tensors
    
    @staticmethod
    def build_loss_and_metric(settings, output_tensors, label_tensors):
        
        normed_logits, logits = output_tensors
        input_y = label_tensors[0]
        
        
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
        return loss, acc
        #
