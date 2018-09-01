# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 15:33:42 2018

@author: limingfan

"""

import tensorflow as tf

from zoo_layers import rnn_layer


def build_graph(config):
    
    input_x = tf.placeholder(tf.int32, [None, None], name='input_x')
    input_y = tf.placeholder(tf.int64, [None], name='input_y')

    with tf.device('/cpu:0'):
        embedding = tf.get_variable('embedding',
                                    [config.vocab.size(), config.vocab.emb_dim],
                                    initializer=tf.constant_initializer(config.vocab.embeddings),
                                    trainable = config.emb_tune)
        embedding_inputs = tf.nn.embedding_lookup(embedding, input_x)
        
        seq_mask = tf.cast(tf.cast(input_x, dtype = tf.bool), dtype = tf.int32)
        seq_len = tf.reduce_sum(seq_mask, 1)

    with tf.name_scope("rnn"):
        
        seq_e = rnn_layer(embedding_inputs, seq_len, 128, config.keep_prob,
                          concat = True, scope = 'bi-lstm-1')        
        '''        
        B = tf.shape(seq_e)[0]
        query = tf.get_variable("query", [config.att_dim])
        query = tf.tile(tf.expand_dims(query, 0), [B, 1])

        feat = att_pool_layer(seq_e, query, seq_mask, config.att_dim,
                              config.keep_prob, is_train=None, scope="att_pooling")
        '''
        feat = seq_e[:,-1,:]

    with tf.name_scope("score"):
        #
        #fc = tf.contrib.layers.dropout(feat, config.keep_prob)
        #fc = tf.layers.dense(fc, 128, name='fc1')            
        #fc = tf.nn.relu(fc)
        
        #fc = tf.contrib.layers.dropout(fc, config.keep_prob)
        logits = tf.layers.dense(feat, config.num_classes, name='fc2')
        # logits = tf.nn.sigmoid(fc)
        
        normed_logits = tf.nn.softmax(logits, name='logits')          
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
    print(normed_logits)
    print(acc)
    print(loss)
    print()
    #

