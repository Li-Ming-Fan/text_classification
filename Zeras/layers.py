# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 17:14:19 2018

@author: limingfan
"""

import tensorflow as tf
# from tensorflow.python.ops import array_ops


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

#
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
def layer_norm(x, scope="layer_norm"):
    num_units = x.shape.as_list()[-1]
    ln = LayerNorm(num_units, scope=scope)
    return ln(x)

def dropout(x, keep_prob):
    drop_layer = Dropout(keep_prob)
    return drop_layer(x)
    
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
    logits = tf.nn.softmax(att_mat, -1)
    logits = tf.nn.dropout(logits, keep_prob)
    outputs = tf.matmul(logits, value)   # [B, TQ, DV]
    return outputs, logits


def multihead_attention_layer(num_heads, num_units,
                              query, key, value, mask_mat=None,
                              keep_prob=1.0):
    """
    """
    dim_all = num_heads * num_units
    
    qd = tf.layers.dense(query, dim_all, name="qd")
    kd = tf.layers.dense(key, dim_all, name="kd")
    vd = tf.layers.dense(value, dim_all, name="vd")
    
    """
    qs = array_ops.split(value = qd, num_or_size_splits = num_heads, axis = -1)
    ks = array_ops.split(value = kd, num_or_size_splits = num_heads, axis = -1)
    vs = array_ops.split(value = vd, num_or_size_splits = num_heads, axis = -1)
    
    # to [B, H, T, D]
    qe = tf.concat([tf.expand_dims(item, 1) for item in qs], 1)
    ke = tf.concat([tf.expand_dims(item, 1) for item in ks], 1)
    ve = tf.concat([tf.expand_dims(item, 1) for item in vs], 1)
    
    """
    
    spq = tf.shape(query)
    batch_size = spq[0]
    time_len = spq[1]
    qs = tf.reshape(qd, [batch_size, time_len, num_heads, num_units])
    ks = tf.reshape(kd, [batch_size, time_len, num_heads, num_units])
    vs = tf.reshape(vd, [batch_size, time_len, num_heads, num_units])
    
    qe = tf.transpose(qs, [0, 2, 1, 3])   # to [B, H, T, D]
    ke = tf.transpose(ks, [0, 2, 1, 3])
    ve = tf.transpose(vs, [0, 2, 1, 3])

    # qkv
    if mask_mat is None:
        mask_mat_e = None
    else:
        mask_mat_e = tf.expand_dims(mask_mat, 1)
    #
    out, att = qkv_att_layer(qe, ke, ve, mask_mat_e, keep_prob)
    #
    
    # concat
    # out_list = [ out[:,idx,:,:] for idx in range(num_heads) ]
    # out_c = tf.concat(out_list, -1)
    
    out_c = tf.transpose(out, [0, 2, 1, 3])           # to [B, T, H, D]
    out_c = tf.reshape(out, [batch_size, time_len, dim_all])
    
    # linear
    out_d = tf.layers.dense(out_c, dim_all, name="out_d")
    return out_d

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
        
        with tf.variable_scope(scope):
            self.dense_query = Dense(self.dim_all, self.dim_all, scope="dense_query")
            self.dense_key = Dense(self.dim_all, self.dim_all, scope="dense_key")
            self.dense_value = Dense(self.dim_all, self.dim_all, scope="dense_value")
            self.dense_trans = Dense(self.dim_all, self.dim_all, scope="dense_trans")
            
        
    def __call__(self, query, key, value, mask_mat=None):
        """
        """
        qd = self.dense_query(query)
        kd = self.dense_key(key)
        vd = self.dense_value(value)
        #
        
        shape_q = tf.shape(query)
        batch_size = shape_q[0]
        q_len = shape_q[1]
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
        
        """
        
        qs = array_ops.split(value = qd, num_or_size_splits = self.num_heads, axis = -1)
        ks = array_ops.split(value = kd, num_or_size_splits = self.num_heads, axis = -1)
        vs = array_ops.split(value = vd, num_or_size_splits = self.num_heads, axis = -1)
        
        # to [B, H, T, D]
        qe = tf.concat([tf.expand_dims(item, 1) for item in qs], 1)
        ke = tf.concat([tf.expand_dims(item, 1) for item in ks], 1)
        ve = tf.concat([tf.expand_dims(item, 1) for item in vs], 1)
        
        """
    
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
        
        # out_list = [ out[:,idx,:,:] for idx in range(self.num_heads) ]
        # out_c = tf.concat(out_list, -1)
        
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
    def __init__(self, sublayer_class, class_args, num_units, keep_prob,
                 scope="sublayer_wrapper"):
        """
        """
        with tf.variable_scope(scope):
            self.sublayer = sublayer_class(*class_args)
            self.dropout = Dropout(keep_prob)
            self.layer_norm = LayerNorm(num_units, scope="sublayer_wrapper")

    def __call__(self, x, sublayer_invoker):
        """ layer & drop & add & norm
        """
        # return x + self.dropout(sublayer_invoker(self.layer_norm(x)))
        return self.layer_norm(x + self.dropout(sublayer_invoker(x)))
    
#

    
    