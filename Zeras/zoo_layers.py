# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 17:14:19 2018

@author: limingfan
"""

import numpy as np
import tensorflow as tf
# from tensorflow.python.ops import array_ops


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

#
def create_dense_vars(input_size, output_size, weight_mat=None,
                      use_bias=True, bias_init_value=0.0, scope="dense"):
    """
    """
    with tf.variable_scope(scope):
        if weight_mat is None:
            W = tf.get_variable("kernel", [input_size, output_size],
                                initializer = tf.variance_scaling_initializer(),
                                dtype = tf.float32)
        else:
            W = weight_mat
        if use_bias:
            b = tf.get_variable("bias", [output_size],
                                initializer = tf.constant_initializer(bias_init_value),
                                dtype = tf.float32)
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
    #
    out_shape = [shape[idx] for idx in range(len(shape_list) - 1)] + [output_size]
    flat_inputs = tf.reshape(inputs, [-1, input_size])
    out = tf.matmul(flat_inputs, Wb[0], transpose_b=transpose_b)
    if Wb[1] is not None: out = tf.nn.bias_add(out, Wb[1])
    out = tf.reshape(out, out_shape)
    return out


def dense(x, output_size, weight_mat=None, transpose_b=False,
          use_bias=True, bias_init_value=0.0, scope="dense"):
    """
    """
    input_size = x.get_shape().as_list()[-1]
    #
    wb = create_dense_vars(input_size, output_size,
                           weight_mat=weight_mat, use_bias=use_bias,
                           bias_init_value=bias_init_value, scope=scope)
    #
    out = dense_with_vars(x, wb, transpose_b=transpose_b)
    return out

#
def layer_norm(x, scope="layer_norm"):
    """
    """
    num_units = x.get_shape().as_list()[-1]
    #
    with tf.variable_scope(scope):
        beta = tf.get_variable('layer_norm_beta', [num_units],
                               initializer=tf.ones_initializer(),
                               trainable=True, dtype=tf.float32)
        gamma = tf.get_variable('layer_norm_gamma', [num_units],
                                initializer=tf.zeros_initializer(),
                                trainable=True, dtype=tf.float32)
        #
        mean, std = tf.nn.moments(x, [-1], keep_dims=True, name='moments')        
        return beta * (x - mean)/ (std + 1e-6) + gamma
    
def layer_norm_api(x, scope=None):
    """
    """
    out = tf.contrib.layers.layer_norm(inputs = x,
                                       begin_norm_axis = -1,
                                       begin_params_axis = -1,
                                       scope = scope)
    return out

#
def multihead_attention_layer(num_heads, num_units,
                              query, key, value, mask_mat=None,
                              keep_prob=1.0, scope="mha"):
    """
    """
    dim_all = num_heads * num_units
    
    with tf.variable_scope(scope):
        
        q_2d = tf.reshape(query, [-1, query.shape[-1]])
        k_2d = tf.reshape(key, [-1, key.shape[-1]])
        v_2d = tf.reshape(key, [-1, value.shape[-1]])
        
        qd = tf.layers.dense(q_2d, dim_all, name="query_d")
        kd = tf.layers.dense(k_2d, dim_all, name="key_d")
        vd = tf.layers.dense(v_2d, dim_all, name="value_d")   
        
        #
        shape_q = tf.shape(query)
        batch_size = shape_q[0]
        seq_len_q = shape_q[1]
        
        shape_k = tf.shape(key)
        seq_len_k = shape_k[1]
        
        #
        qd_4d = tf.reshape(qd, [batch_size, seq_len_q, num_heads, num_units])
        kd_4d = tf.reshape(kd, [batch_size, seq_len_k, num_heads, num_units])
        vd_4d = tf.reshape(vd, [batch_size, seq_len_k, num_heads, num_units])
        
        qd_4d = tf.transpose(qd_4d, [0, 2, 1, 3])
        kd_4d = tf.transpose(kd_4d, [0, 2, 1, 3])
        vd_4d = tf.transpose(vd_4d, [0, 2, 1, 3])
        
        #    
        att_scores = tf.matmul(qd_4d, kd_4d, transpose_b=True) / (num_units ** 0.5)
        
        if mask_mat is not None:
            mask_mat_e = tf.cast(tf.expand_dims(mask_mat, axis=[1]), tf.float32)
            att_scores += (mask_mat_e - 1.0) * 1000000.0
            
        att_probs = tf.nn.softmax(att_scores)
        att_probs = tf.nn.dropout(att_probs, keep_prob)
        
        #
        value_summ = tf.matmul(att_probs, vd_4d)
        
        value_summ = tf.transpose(value_summ, [0, 2, 1, 3])
        value_summ = tf.reshape(value_summ, [batch_size, seq_len_q, dim_all])
    
        # linear
        value_summ = tf.layers.dense(value_summ, dim_all, name="out_d")
        
        return value_summ

#
def att_qkv_layer(inputs, memory, values, mask_m, att_dim, keep_prob=1.0, scope="qkv"):
    """ batch_major
        inputs: [B, TQ, DQ]
        memory: [B, TM, DM]
        values: [B, TV, DV]  # TM = TV
    """
    with tf.variable_scope(scope):
        d_inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)  # [B, TQ, DQ]
        d_memory = tf.nn.dropout(memory, keep_prob=keep_prob)
        #
        inputs_d = dense(d_inputs, att_dim, use_bias=False, scope="inputs")            
        memory_d = dense(d_memory, att_dim, use_bias=False, scope="memory")
        # inputs_d = tf.nn.relu(inputs_d)
        # memory_d = tf.nn.relu(memory_d)
        #
        # [B, TQ, TM]
        att_mat = tf.matmul(inputs_d, tf.transpose(memory_d, [0, 2, 1])) / (att_dim ** 0.5)
        # 
        mask_3d = tf.cast(tf.expand_dims(mask_m, axis=1), tf.float32) # [B, 1, TM]
        att_masked = tf.add(att_mat, 1e30 * (mask_3d - 1) )  # -inf   # [B, TQ, TM]
        logits = tf.nn.softmax(att_masked)
        #
        d_values = tf.nn.dropout(values, keep_prob=keep_prob)  # [B, TM, DV]
        values_d = dense(d_values, att_dim, use_bias=False, scope="values")
        # values_d = tf.nn.relu(values_d)
        #
        outputs = tf.matmul(logits, values_d)   # [B, TQ, DV_d]
    return outputs
    
def qk_mat_layer(inputs, memory, att_dim, keep_prob=1.0, scope="qk_mat"):
    with tf.variable_scope(scope):
        d_inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)  # [B, TQ, D]
        d_memory = tf.nn.dropout(memory, keep_prob=keep_prob)
        #
        inputs_d = dense(d_inputs, att_dim, use_bias=False, scope="inputs")            
        memory_d = dense(d_memory, att_dim, use_bias=False, scope="memory")
        # inputs_d = tf.nn.relu(inputs_d)
        # memory_d = tf.nn.relu(memory_d)
        #
        # [B, TQ, TM]
        att_mat = tf.matmul(inputs_d, tf.transpose(memory_d, [0, 2, 1])) / (att_dim ** 0.5)
    return att_mat

def qk_value_pool_layer(qk_mat, values, mask_k, hidden, keep_prob=1.0, scope="qk_pool"):
    with tf.variable_scope(scope):
        # 
        mask_3d = tf.cast(tf.expand_dims(mask_k, axis=1), tf.float32) # [B, 1, TM]
        att_masked = tf.add(qk_mat, 1e30 * (mask_3d - 1) )  # -inf   # [B, TQ, TM]
        logits = tf.nn.softmax(att_masked)
        #
        d_values = tf.nn.dropout(values, keep_prob=keep_prob)  # [B, TM, DV]
        values_d = dense(d_values, qk_mat, use_bias=False, scope="values")
        # values_d = tf.nn.relu(values_d)
        outputs = tf.matmul(logits, values_d)   # [B, TQ, DV_d]
    return outputs

    
def att_pool_layer(query, seq, seq_mask, att_dim, keep_prob=1.0, scope="att_pooling"):
    """ batch_major
        query: [B, DQ]
        seq: [B, TM, DM]
        seq_mask: [B, TM]
    """
    with tf.variable_scope(scope):
        #
        query = tf.expand_dims(query, 1)  # [B, 1, DQ], TQ = 1
        #
        d_inputs = tf.nn.dropout(query, keep_prob=keep_prob)  # [B, TQ, DQ]
        d_memory = tf.nn.dropout(seq, keep_prob=keep_prob)    # [B, TM, DM]
        #
        inputs_d = dense(d_inputs, att_dim, use_bias=False, scope="inputs")            
        memory_d = dense(d_memory, att_dim, use_bias=False, scope="memory")
        # inputs_d = tf.nn.relu(inputs_d)
        # memory_d = tf.nn.relu(memory_d)
        #
        # [B, TQ, TM]        
        att_mat = tf.matmul(inputs_d, tf.transpose(memory_d, [0, 2, 1])) / (att_dim ** 0.5)
        # 
        mask_3d = tf.cast(tf.expand_dims(seq_mask, axis=1), tf.float32) # [B, 1, TM]
        att_masked = tf.add(att_mat, 1e30 * (mask_3d - 1) )  # -inf   # [B, TQ, TM]
        logits = tf.nn.softmax(att_masked)
        #
        d_values = tf.nn.dropout(seq, keep_prob=keep_prob)  # [B, TM, DV]
        values_d = dense(d_values, att_dim, use_bias=False, scope="values")
        # values_d = tf.nn.relu(values_d)
        #
        outputs = tf.matmul(logits, values_d)   # [B, TQ, DV_d]
        outputs = tf.squeeze(outputs, 1)        # [B, DV_d]
    return outputs

#
def rnn_layer(input_sequence, sequence_length, rnn_size,
              keep_prob = 1.0, activation = None,
              concat = True, scope = 'bi-lstm'):
    '''build bidirectional lstm layer'''
    #
    # time_major = False
    #
    input_sequence = tf.nn.dropout(input_sequence, keep_prob)
    #
    weight_initializer = tf.truncated_normal_initializer(stddev = 0.01)
    act = activation or tf.nn.tanh
    #
    cell_fw = tf.contrib.rnn.LSTMCell(rnn_size, activation = act,
                                      initializer = weight_initializer)
    cell_bw = tf.contrib.rnn.LSTMCell(rnn_size, activation = act,
                                      initializer = weight_initializer)
    #
    #cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=dropout_rate)
    #cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=dropout_rate)
    #
    #cell_fw = MyLSTMCell(rnn_size, keep_prob, initializer = weight_initializer)
    #cell_bw = MyLSTMCell(rnn_size, keep_prob, initializer = weight_initializer)
    #
    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_sequence,
                                                    sequence_length = sequence_length,
                                                    time_major = False,
                                                    dtype = tf.float32,
                                                    scope = scope)
    #
    if concat:
        rnn_output = tf.concat(rnn_output, 2, name = 'output')
    else:
        rnn_output = tf.multiply(tf.add(rnn_output[0], rnn_output[1]), 0.5, name = 'output')
    #
    return rnn_output
    #

def gru_layer(input_sequence, sequence_length, rnn_size,
              keep_prob = 1.0, activation = None,
              concat = True, scope = 'bi-gru'):
    '''build bidirectional gru layer'''
    #
    # time_major = False
    #
    input_sequence = tf.nn.dropout(input_sequence, keep_prob)
    #
    act = activation or tf.nn.tanh
    #
    cell_fw = tf.nn.rnn_cell.GRUCell(rnn_size, activation = act)
    cell_bw = tf.nn.rnn_cell.GRUCell(rnn_size, activation = act)
    #
    # cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=dropout_rate)
    # cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=dropout_rate)
    #
    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_sequence,
                                                    sequence_length = sequence_length,
                                                    time_major = False,
                                                    dtype = tf.float32,
                                                    scope = scope)
    #
    if concat:
        rnn_output = tf.concat(rnn_output, 2, name = 'output')
    else:
        rnn_output = tf.multiply(tf.add(rnn_output[0], rnn_output[1]), 0.5, name = 'output')
    #
    return rnn_output
    #     

#
def gather_and_pad_layer(x, num_items):
    """ x: (BS', D)
        num_items : (B,)
        
        returning: (B, S, D), (B, S)
    """
    B = tf.shape(num_items)[0]
    T = tf.reduce_max(num_items)
    
    pad_item = tf.zeros(shape = tf.shape(x[0:1,:]) )
    one_int32 = tf.ones(shape = (1,), dtype = tf.int32)
    zero_int32 = tf.zeros(shape = (1,), dtype = tf.int32)
    
    bsd_ta = tf.TensorArray(size = B, dtype = tf.float32)
    mask_ta = tf.TensorArray(size = B, dtype = tf.int32)
    time = tf.constant(0)
    posi = tf.constant(0)
    
    def condition(time, posi_s, bsd_s, mask_s):
        return tf.less(time, B)
    
    def body(time, posi_s, bsd_s, mask_s):        
        posi_e = posi_s + num_items[time]        
        chunk = x[posi_s:posi_e, :]
        #
        mask_c = tf.tile(one_int32, [ num_items[time] ] )
        #
        d = T - num_items[time]
        chunk, mask_c = tf.cond(d > 0,
                                lambda: (tf.concat([chunk, tf.tile(pad_item, [d, 1])], 0),
                                         tf.concat([mask_c, tf.tile(zero_int32, [d])], 0) ),
                                lambda: (chunk, mask_c) )
        #
        bsd_s = bsd_s.write(time, chunk)
        mask_s = mask_s.write(time, mask_c)
        return (time + 1, posi_e, bsd_s, mask_s)
        
    t, p, bsd_w, mask_w = tf.while_loop(cond = condition, body = body,
                                        loop_vars = (time, posi, bsd_ta, mask_ta) )
    bsd = bsd_w.stack()
    mask = mask_w.stack()
    
    return bsd, mask

    