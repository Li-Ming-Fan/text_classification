# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 06:23:53 2019

@author: limingfan
"""

import tensorflow as tf


def squash_layer(x, name="squash"):
    """ squash the last dim
    """
    x_norm_s = tf.reduce_sum(x * x, axis = -1, keepdims=True)
    x_norm = tf.sqrt(x_norm_s + 1e-9)
    p = x_norm_s / (1 + x_norm_s)
    v = x / x_norm
    output = p * v
    return output, p, v

def capsule_layer(x, x_mask_2d, num_caps, cap_dim, num_iter = 3, keep_prob = 1.0,
                  caps_initial_state = None, scope="capsules"):
    """ 0, initialize the correlation matrix, calculate the information base matrix,
        1, calculate normalized correlations,
        2, calculate information to be incorporated,
        3, squash incorporated information,
        4, update correlation matrix,
        5, iterate from 1 to 4 for num_iter times,
        
        x: from where information is extracted, [B, TX, D]
        x_mask_2d: [B, TX]
        caps_initial_state: [B, num_caps, cap_dim]
    """
    with tf.variable_scope(scope):
        x_mask_3d = tf.expand_dims(x_mask_2d, -1)    # [B, TX, 1]
        x_mask_4d = tf.expand_dims(x_mask_3d, -1)    # [B, TX, 1, 1]
        #
        B = tf.shape(x)[0]
        TX = tf.shape(x)[1]
        #
        # x = tf.nn.dropout(x, keep_prob)
        #
        # [B, TX, num_caps, cap_dim]
        activation = tf.tanh
        info_base = tf.layers.dense(x, num_caps * cap_dim, activation=activation)
        info_base = tf.reshape(info_base, shape = [B, TX, num_caps, cap_dim],
                               name = "info_base")
        #
        info_base = info_base * x_mask_4d
        #
        # [B, TX, num_caps]
        if caps_initial_state is None:
            cor_mat = tf.zeros([B, TX, num_caps], dtype = tf.float32, name = "cor_mat")
        else:
            output_e = tf.expand_dims(caps_initial_state, 1)   # [B, 1, num_caps, cap_dim]
            cor_mat = tf.reduce_sum(output_e * info_base, -1, name = "cor_mat")
        #
        # iterations
        for itr in range(num_iter):
            cor_normed = tf.nn.softmax(cor_mat, -1)  # [B, TX, num_caps]
            cor_4d = tf.expand_dims(cor_normed, -1)  # [B, TX, num_caps, 1]
            info_mat = cor_4d * info_base
            info_sum = tf.reduce_sum(info_mat, 1)    # [B, num_caps, cap_dim]
            # output = info_sum
            output, p, v = squash_layer(info_sum)    # [B, num_caps, cap_dim]
            #
            # update, [B, TX, num_caps]
            output_e = tf.expand_dims(output, 1)     # [B, 1, num_caps, cap_dim]
            cor_delta = tf.reduce_sum(output_e * info_base, -1)
            cor_mat = cor_mat + cor_delta
            #
        #
        return output
    
"""
    with tf.variable_scope("feat"):
        
        seq_e = enc_t
    
        #
        num_caps = 3
        cap_dim = 64
        num_iter = 3
        
        mask_t = tf.cast(mask_t, dtype=tf.float32)    
        cap_d = capsule_layer(seq_e, mask_t, num_caps, cap_dim, num_iter = num_iter,
                              keep_prob = keep_prob, caps_initial_state = None, scope="capsules")
        cap_d = tf.nn.relu(cap_d)
        #
        feat = tf.reshape(cap_d, [-1, num_caps * cap_dim])
        #
"""
