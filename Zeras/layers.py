# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 17:14:19 2018

@author: limingfan
"""

import numpy as np
import tensorflow as tf


class Dense():
    """
    """
    def __init__(self, input_size, output_size, weight_mat=None,
                 use_bias=True, bias_init_value=0.0, scope="dense"):
        """
        """
        wb = create_dense_vars(input_size, output_size,
                               weight_mat=weight_mat, use_bias=use_bias,
                               bias_init_value=bias_init_value, scope=scope)
        #
        self.wb = wb
    
    def __call__(self, inputs, transpose_b=False):
        """
        """
        out = dense_with_vars(inputs, self.wb, transpose_b=transpose_b)
        return out

class LayerNorm():
    """
    """
    def __init__(self, num_units, epsilon=1e-6, scope="layer_norm"):
        """
        """
        with tf.variable_scope(scope):
            self.beta = tf.get_variable('layer_norm_beta', [num_units],
                                        initializer=tf.ones_initializer(),
                                        trainable=True)
            self.gamma = tf.get_variable('layer_norm_gamma', [num_units],
                                         initializer=tf.zeros_initializer(),
                                         trainable=True)
            self.eps = epsilon
    
    def __call__(self, x):
        """
        """
        mean, std = tf.nn.moments(x, [-1], keep_dims=True, name='moments')        
        return self.beta * (x - mean)/ (std + self.eps) + self.gamma
    
class Dropout():
    """
    """
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
        
    def __call__(self, x):        
        return tf.nn.dropout(x, self.keep_prob)
    
#
def dense_with_w(inputs, hidden, weights, transpose_b=False):
    """
    """
    shape_list = inputs.get_shape().as_list()
    if len(shape_list) == 2:
        out = tf.matmul(inputs, weights, transpose_b=transpose_b)
        return out
    #
    shape = tf.shape(inputs)
    out_shape = [shape[idx] for idx in range(len(shape_list) - 1)] + [hidden]
    input_size = shape_list[-1]
    flat_inputs = tf.reshape(inputs, [-1, input_size])
    out = tf.matmul(flat_inputs, weights, transpose_b = transpose_b)
    out = tf.reshape(out, out_shape)
    return out

def create_dense_vars(input_size, output_size, weight_mat=None,
                      use_bias=True, bias_init_value=0.0, scope="dense"):
    """
    """
    with tf.variable_scope(scope):
        if weight_mat is None:
            W = tf.get_variable("kernel", [input_size, output_size],
                                initializer = tf.variance_scaling_initializer())
        else:
            W = weight_mat
        if use_bias:
            b = tf.get_variable("bias", [output_size],
                                initializer = tf.constant_initializer(bias_init_value))
        else:
            b = None
    #
    return W, b

def dense_with_vars(inputs, Wb, transpose_b=False):
    """
    """
    shape_list = inputs.get_shape().as_list()
    if len(shape_list) == 2:
        out = tf.matmul(inputs, Wb[0], transpose_b=transpose_b)
        if Wb[1] is not None: out = tf.nn.bias_add(out, Wb[1])
        return out
    #
    input_size = shape_list[-1]
    shape = tf.shape(inputs)
    if transpose_b:
        output_size = Wb[0].get_shape().as_list()[0]
    else:
        output_size = Wb[0].get_shape().as_list()[1]
    out_shape = [shape[idx] for idx in range(len(shape_list) - 1)] + [output_size]
    flat_inputs = tf.reshape(inputs, [-1, input_size])
    out = tf.matmul(flat_inputs, Wb[0], transpose_b=transpose_b)
    if Wb[1] is not None: out = tf.nn.bias_add(out, Wb[1])
    out = tf.reshape(out, out_shape)
    return out

def layer_norm(x, scope=None):
    num_units = tf.shape(x)[-1]
    ln = LayerNorm(num_units, scope=scope)
    return ln(x)
    
# 
def build_module_copies(module_class, class_args, N, scope="module_copies"):
    """ module_class: must be some class
        class_args: be the args for module_class init function
    """
    list_copies = []
    with tf.variable_scope(scope):
        for idx in range(N):
            with tf.variable_scope("copy_%d" % idx):
                module_copied = module_class(*class_args)
                list_copies.append(module_copied)
    return list_copies

#
def qkv_att_layer(query, key, value, mask_mat=None, keep_prob=1.0):
    """ batch_major
        query: [B, TQ, DQ]
        key: [B, TK, DK]    # DQ = DK
        value: [B, TV, DV]  # TK = TV
        mask_mat: [B, TQ, TK], or [1, TQ, TK]
        
        return: [B, TQ, DV]
        3-dim, or higher-dim
    """
    dim = query.get_shape().as_list()[-1]
    att_mat = tf.matmul(query, key, transpose_b=True) / (dim ** 0.5)
    #
    if mask_mat is not None:
        att_mat = tf.add(att_mat, 1e16 * (mask_mat - 1) )  # -inf   # [B, TQ, TM]
    #
    logits = tf.nn.softmax(att_mat)
    logits = tf.nn.dropout(logits, keep_prob)
    outputs = tf.matmul(logits, value)   # [B, TQ, DV]
    return outputs, logits

#
class MultiHeadAttention():
    """
    """
    def __init__(self, num_heads, num_units, keep_prob=1.0, scope="multi_head"):
        """
        """
        self.keep_prob = keep_prob
        self.num_heads = num_heads
        self.num_units = num_units
        self.dim_all = self.num_heads * self.num_units
        
        self.attention = None
        
        d_model = num_heads * num_units
        with tf.variable_scope(scope):
            self.dense_query = Dense(d_model, d_model, scope="dense_query")
            self.dense_key = Dense(d_model, d_model, scope="dense_key")
            self.dense_value = Dense(d_model, d_model, scope="dense_value")
            self.dense_trans = Dense(d_model, d_model, scope="dense_trans")
            
        
    def __call__(self, query, key, value, mask_mat=None):
        """
        """
        qd = self.dense_query(query)
        kd = self.dense_key(key)
        vd = self.dense_value(value)
        #
        spq = tf.shape(query)
        batch_size = spq[0]
        q_len = spq[1]
        k_len = tf.shape(key)[1]
        # v_len = spq[1]
        #
        qs = tf.reshape(qd, [batch_size, q_len, self.num_heads, self.num_units])
        ks = tf.reshape(kd, [batch_size, k_len, self.num_heads, self.num_units])
        vs = tf.reshape(vd, [batch_size, k_len, self.num_heads, self.num_units])
        #
        qe = tf.transpose(qs, [0, 2, 1, 3])   # to [B, H, T, D]
        ke = tf.transpose(ks, [0, 2, 1, 3])
        ve = tf.transpose(vs, [0, 2, 1, 3])
    
        # qkv
        if mask_mat is None:
            mask_mat_e = None
        else:
            mask_mat_e = tf.expand_dims(mask_mat, 1)
        #
        out, att = qkv_att_layer(qe, ke, ve, mask_mat_e, self.keep_prob)
        #
        self.attention = att
        #
        
        # concat
        out_c = tf.transpose(out, [0, 2, 1, 3])           # to [B, T, H, D]
        out_c = tf.reshape(out, [batch_size, q_len, self.dim_all])    
        
        # linear
        out_d = self.dense_trans(out_c)
        return out_d
    
class PositionwiseFeedForward():
    """
    """
    def __init__(self, num_dim_all, dim_middle, keep_prob, scope="pwff"):
        """
        """        
        with tf.variable_scope(scope):
            self.d1 = Dense(num_dim_all, dim_middle, scope="ff_d1")
            self.d2 = Dense(dim_middle, num_dim_all, scope="ff_d2")
            self.dropout = Dropout(keep_prob)
        
    def __call__(self, x):
        x = tf.nn.relu(self.d1(x))
        x = self.d2(self.dropout(x))
        return x

#
class SublayerWrapper():
    """
    """
    def __init__(self, num_units, keep_prob, sublayer_class, class_args,
                 scope="sublayer_wrapper"):
        """
        """
        with tf.variable_scope(scope):
            self.layer_norm = LayerNorm(num_units, scope="sublayer_wrapper")
            self.dropout = Dropout(keep_prob)
            self.sublayer = sublayer_class(*class_args)
    
    def __call__(self, x, sublayer_invoker):
        """
        """
        return x + self.dropout(sublayer_invoker(self.layer_norm(x)))
    
#
def get_mask_mat_from_mask_seq(mask_a, mask_b):
    """ mask_a: [B, TA]
    """
    mask_ae = tf.cast(tf.expand_dims(mask_a, 2), tf.float32)  # [B, TA, 1]
    mask_be = tf.cast(tf.expand_dims(mask_b, 1), tf.float32)  # [B, 1, TB]
    mask = mask_ae * mask_be    # [B, TA, TB]
    return mask

#
def get_mask_mat_subsequent(size, name="mask_subsequent"):
    """ subsequent mask
    """
    
    """
    mask_mat = np.zeros((1, size, size), dtype = np.float32)
    for idx in range(size):
        for idy in range(size):
            if idx <= idy: mask_mat[0, idx, idy] = 1.0
    #
    mask_tensor = tf.get_variable(name, shape = (1, size, size),
                                  initializer = tf.constant_initializer(mask_mat),
                                  trainable = False)
    """
    #
    mask_tensor = tf.constant(1.0, shape = (1, size, size), dtype=tf.float32)
    mask_tensor = tf.linalg.band_part(mask_tensor,
                                      num_lower = -1,
                                      num_upper = 0,
                                      name = name)
    return mask_tensor

def get_list_subs_masks(max_len, name="subs_mask"):
    """ subsequent masks
    """    
    list_masks = []
    for step in range(max_len):
        subs_mask = get_mask_mat_subsequent(step+1, name = name+"_%d" % step)
        list_masks.append(subs_mask)
    return list_masks

def get_list_dcd_crs_masks(src_mask_seq, max_len):
    """ decoder cross masks
    """    
    list_masks = []
    mask_be = tf.cast(tf.expand_dims(src_mask_seq, 1), tf.float32)  # [B, 1, TB]
    for step in range(max_len):        
        crs_mask = tf.tile(mask_be, [1, step+1, 1])
        list_masks.append(crs_mask)
    return list_masks
    
#
def get_position_emb_mat(max_seq_len, posi_emb_dim, posi_emb_model,
                         name="position_embeddings"):
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
                             trainable = False)
        
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

