# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 13:49:51 2018

@author: limingfan

"""

import tensorflow as tf

def build_graph(config):

    input_x = tf.placeholder(tf.int32, [None, None], name='input_x')
    input_y = tf.placeholder(tf.int64, [None], name='input_y')

    with tf.device('/cpu:0'):
        emb_mat = tf.get_variable('embedding',
                                  [config.vocab.size(), config.vocab.emb_dim],
                                  initializer=tf.constant_initializer(config.vocab.embeddings),
                                  trainable = config.emb_tune)
        seq_emb = tf.nn.embedding_lookup(emb_mat, input_x)

    with tf.name_scope("cnn"):
        #
        conv1_5 = tf.layers.conv1d(seq_emb, 128, 5, padding='same', name='conv1_5')
        conv1_3 = tf.layers.conv1d(seq_emb, 128, 3, padding='same', name='conv1_3')
        conv1_2 = tf.layers.conv1d(seq_emb, 128, 2, padding='same', name='conv1_2')
        
        conv1 = tf.concat([conv1_5, conv1_3, conv1_2], -1)
        
        conv2_5 = tf.layers.conv1d(conv1, 128, 5, name='conv2_5')
        conv2_3 = tf.layers.conv1d(conv1, 128, 3, name='conv2_3')
        conv2_2 = tf.layers.conv1d(conv1, 128, 2, name='conv2_2')
        
        feat1 = tf.reduce_max(conv2_5, reduction_indices=[1], name='feat1')
        feat2 = tf.reduce_max(conv2_3, reduction_indices=[1], name='feat2')
        feat3 = tf.reduce_max(conv2_2, reduction_indices=[1], name='feat3')
        
        feat = tf.concat([feat1, feat2, feat3], 1)

    with tf.name_scope("score"):
        #
        fc = tf.nn.dropout(feat, config.keep_prob)
        fc = tf.layers.dense(fc, 128, name='fc1')            
        fc = tf.nn.relu(fc)
        
        fc = tf.nn.dropout(fc, config.keep_prob)
        logits = tf.layers.dense(fc, config.num_classes, name='fc2')
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

