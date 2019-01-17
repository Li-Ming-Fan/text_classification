# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 17:14:19 2018

@author: limingfan
"""


import tensorflow as tf

#
def dense(inputs, hidden, use_bias=True, scope="dense"):
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        shape_list = inputs.get_shape().as_list()
        dim = shape_list[-1]
        out_shape = [shape[idx] for idx in range(len(shape_list) - 1)] + [hidden]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("W", [dim, hidden],
                            initializer = tf.variance_scaling_initializer())
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable("b", [hidden],
                                initializer = tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res
    
def dropout(inputs, keep_prob, mode="recurrent"):
    noise_shape = None
    scale = 1.0
    shape = tf.shape(inputs)
    if mode == "embedding":
        noise_shape = [shape[0], 1]
        scale = keep_prob
    if mode == "recurrent" and len(inputs.get_shape().as_list()) == 3:
        # batch_major
        noise_shape = [shape[0], 1, shape[-1]]
    inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape) * scale
    
    return inputs

#
def do_mask_padding_elems(x, mask):
    # make padding elements in x to -inf,
    # for next step of softmax,
    # (batch, time), or (time, batch), or
    # (batch, time, units), or (time, batch, units)
    return tf.add(x, 1e30 * tf.cast(mask - 1, dtype=tf.float32) )

def dot_att_layer(query, memory, mask_m_2d, att_dim,
                  keep_prob=1.0, gating=False, scope="dot_attention"):
    """ batch_major,
        query: [B, TQ, DQ]
        memory: [B, TM, DM]
        mask_m_2d: [B, TM]
    """
    with tf.variable_scope(scope):
        # TQ = tf.shape(query)[1]
        with tf.variable_scope("attention"):
            d_query = tf.nn.dropout(query, keep_prob=keep_prob)  # [B, TQ, D]
            d_memory = tf.nn.dropout(memory, keep_prob=keep_prob)
            #
            inputs_r = tf.nn.relu(dense(d_query, att_dim, use_bias=False, scope="inputs"))
            memory_r = tf.nn.relu(dense(d_memory, att_dim, use_bias=False, scope="memory"))
            # [B, TQ, TM]
            att_mat = tf.matmul(inputs_r, tf.transpose(memory_r, [0, 2, 1])) / (att_dim ** 0.5)
            # 
            mask_3d = tf.expand_dims(mask_m_2d, axis=1)  # [B, 1, TM]
            logits = tf.nn.softmax(do_mask_padding_elems(att_mat, mask_3d)) # [B, TQ, TM]
            #
            d_memory = tf.nn.dropout(memory, keep_prob=keep_prob)  # [B, TM, DM]
            outputs = tf.matmul(logits, memory)   # [B, TQ, DM]
            #
            result = tf.concat([query, outputs], axis=2)  # [B, TQ, DQ + DM]
            
        if gating:
            with tf.variable_scope("gate"):
                dim = result.get_shape().as_list()[-1]
                d_result = tf.nn.dropout(result, keep_prob=keep_prob)
                gate = tf.nn.sigmoid(dense(d_result, dim, use_bias=False))
                result = tf.multiply(result, gate)
        
        return result
    
def att_pool_layer(seq, query, seq_mask, att_dim,
                   keep_prob=1.0, is_train=None, scope="att_pooling"):
    with tf.variable_scope(scope):
        # batch_major
        # seq: [B, T, D]
        # query: [B, DQ]
        # seq_mask: [B, T]
        d_seq = tf.nn.dropout(seq, keep_prob=keep_prob)
        seq_shape = tf.shape(seq)
        T = seq_shape[1]
        D = seq_shape[2]
        with tf.variable_scope("attention"):
            d_seq = tf.nn.tanh(dense(d_seq, att_dim, scope="att_dense"))
            #
            q_tile = tf.tile(tf.expand_dims(query, 1), [1, T, 1])  # [B, T, DQ]
            att_value = tf.reduce_sum(tf.multiply(d_seq, q_tile), 2)  # [B, T]
            # mask = tf.tile(tf.expand_dims(seq_mask, axis=1), [1, T, 1])  # [B, T, 1]
            # att_value = tf.matmul(d_seq, tf.transpose(query, [1, 0])) / (hidden ** 0.5)
            #
            logits = tf.nn.softmax(do_mask_padding_elems(att_value, seq_mask))  # [B, T]
            logits_s = tf.tile(tf.expand_dims(logits, 2), [1, 1, D])  # [B, T, D]
            vec_pooled = tf.reduce_sum(tf.multiply(logits_s, seq), 1)  #  [B, D]
        #      
        return vec_pooled
        #
        # seq_mask 相当于seq_length一样的作用，因为seq里有padding_token
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

