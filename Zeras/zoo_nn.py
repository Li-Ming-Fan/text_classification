# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 17:14:19 2018

@author: limingfan
"""

import tensorflow as tf


#
def get_shape_list(tensor):
    """
    """    
    shape = tensor.shape.as_list()
    
    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None: non_static_indexes.append(index)
        
    if not non_static_indexes: return shape
    
    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


#
def get_emb_positioned(x, token_emb, position_emb):
    """ x: [None, None]
    """
    posi = tf.range(tf.shape(x)[-1])
    
    seq_emb_t = tf.nn.embedding_lookup(token_emb, x)
    seq_emb_p = tf.nn.embedding_lookup(position_emb, posi)
    
    return seq_emb_t + seq_emb_p

#
def get_mask_mat_subsequent(size, name="mask_subsequent"):
    """ subsequent mask
    """
    mask_tensor = tf.constant(1.0, shape = (1, size, size), dtype=tf.float32)
    mask_tensor = tf.linalg.band_part(mask_tensor,
                                      num_lower = -1,
                                      num_upper = 0,
                                      name = name)
    return mask_tensor
    
#
def get_tensor_expanded(x, dim, dtype=None):
    """
    """
    x = tf.expand_dims(x, dim)
    if dtype is not None:
        x = tf.cast(x, dtype=dtype)
    #
    return x

#
def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh((0.79788456 * (x + 0.044715 * tf.pow(x, 3)) )))
    return x * cdf


#
def dropout(inputs, keep_prob, feature_stick=True, mode="recurrent"):
    #
    if feature_stick is False: return tf.nn.dropout(inputs, keep_prob)
    #
    shape = tf.shape(inputs)
    if mode == "embedding" and len(inputs.get_shape().as_list()) == 2:
        noise_shape = [shape[0], 1]
        scale = keep_prob
        out = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape) * scale
    elif mode == "recurrent" and len(inputs.get_shape().as_list()) == 3:     
        noise_shape = [shape[0], 1, shape[-1]]  # batch_major
        out = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)
    else: 
        out = tf.nn.dropout(inputs, keep_prob, noise_shape=None)
    return out

#
def get_label_smoothened(onehot_label, num_classes, delta):
    """
    """
    new_label = (1.0 - delta) * onehot_label + delta / num_classes
    return new_label


