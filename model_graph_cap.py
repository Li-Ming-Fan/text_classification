# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 07:18:29 2019

@author: limingfan
"""


import tensorflow as tf

from zoo_capsules import capsule_layer


def build_graph(settings):

    input_x = tf.placeholder(tf.int32, [None, None], name='input_x')
    input_y = tf.placeholder(tf.int64, [None], name='input_y')
    
    keep_prob = tf.get_variable("keep_prob", shape=[], dtype=tf.float32, trainable=False)
    #

    with tf.device('/cpu:0'):
        emb_mat = tf.get_variable('embedding',
                                  [settings.vocab.size(), settings.vocab.emb_dim],
                                  initializer=tf.constant_initializer(settings.vocab.embeddings),
                                  trainable = settings.emb_tune)
        seq_emb = tf.nn.embedding_lookup(emb_mat, input_x)
        
        seq_mask = tf.cast(tf.cast(input_x, dtype = tf.bool), dtype = tf.int32)

    with tf.variable_scope("feat"):
        
        seq_e = seq_emb
        
        B = tf.shape(seq_e)[0]
    
        #
        num_caps = 3
        cap_dim = 64
        num_iter = 3
        
        caps_initial_state = tf.get_variable('caps_state', shape = (num_caps, cap_dim),
                                             initializer = tf.truncated_normal_initializer() )
        caps_initial_state = tf.tile(tf.expand_dims(caps_initial_state, 0), [B, 1, 1])
        
        mask_t = tf.cast(seq_mask, dtype=tf.float32)    
        cap_d = capsule_layer(seq_e, mask_t, num_caps, cap_dim, num_iter = num_iter,
                              keep_prob = keep_prob, caps_initial_state = caps_initial_state,
                              scope="capsules")
        cap_d = tf.nn.relu(cap_d)
        #
        feat = tf.reshape(cap_d, [-1, num_caps * cap_dim])
        #

    with tf.name_scope("score"):
        #
        fc = tf.nn.dropout(feat, keep_prob)
        fc = tf.layers.dense(fc, 128, name='fc1')            
        fc = tf.nn.relu(fc)
        
        fc = tf.nn.dropout(fc, keep_prob)
        logits = tf.layers.dense(fc, settings.num_classes, name='fc2')

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
