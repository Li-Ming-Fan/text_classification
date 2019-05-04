# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 17:14:19 2018

@author: limingfan
"""

import numpy as np
import tensorflow as tf

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
def get_position_emb_mat(max_seq_len, posi_emb_dim, posi_emb_model,
                         trainable = False, name="position_embeddings"):
    """
    """
    d_model_recip_2 = 2.0 / posi_emb_model
    
    arg_mat = np.zeros((max_seq_len, posi_emb_dim), dtype=np.float32)
    for idx in range(max_seq_len):
        for idm in range(posi_emb_dim):
            arg_mat[idx, idm] = idx * 1e-4**(d_model_recip_2 * idm)
    #
    pe_sin = np.sin(arg_mat)
    pe_cos = np.cos(arg_mat)
    #    
    pe_sin = np.expand_dims(pe_sin, -1)  
    pe_cos = np.expand_dims(pe_cos, -1)
    pe_all = np.concatenate([pe_sin, pe_cos], -1)  # (T, D, 2)
    #
    pe_all = np.reshape(pe_all, [max_seq_len, -1])
    pe_all = pe_all[:, 0:posi_emb_dim]
    
    #
    # tf.Tensor
    pe_mat = tf.get_variable(name, shape = (max_seq_len, posi_emb_dim),
                             initializer = tf.constant_initializer(pe_all),
                             trainable = trainable)
        
    return pe_mat

def get_emb_positioned(x, token_emb, position_emb):
    """ x: [None, None]
    """
    posi = tf.range(tf.shape(x)[-1])
    
    seq_emb_t = tf.nn.embedding_lookup(token_emb, x)
    seq_emb_p = tf.nn.embedding_lookup(position_emb, posi)
    
    return seq_emb_t + seq_emb_p

#
def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh((0.79788456 * (x + 0.044715 * tf.pow(x, 3)) )))
    return x * cdf

#
def get_label_smoothened(onehot_label, num_classes, delta):
    """
    """
    new_label = (1.0 - delta) * onehot_label + delta / num_classes
    return new_label
    
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

