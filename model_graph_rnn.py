# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 15:33:42 2018

@author: limingfan

"""

import tensorflow as tf

def dense(inputs, hidden, use_bias=True, scope="dense"):
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(
            len(inputs.get_shape().as_list()) - 1)] + [hidden]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("W", [dim, hidden])
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable(
                "b", [hidden], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res
    
def dropout(inputs, keep_prob, mode="recurrent"):
    if keep_prob is not None and keep_prob < 1.0:
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

def rnn_layer(input_sequence, sequence_length, rnn_size,
              keep_prob = 1.0, concat = True, scope = 'bi-lstm'):
    '''build bidirectional lstm layer'''
    #
    # time_major = False
    #
    if keep_prob < 1.0:
        input_sequence = dropout(input_sequence, keep_prob)
    #
    # to time_major from batch_major
    input_sequence = tf.transpose(input_sequence, [1,0,2])
    #
    weight_initializer = tf.truncated_normal_initializer(stddev = 0.01)
    #
    cell_fw = tf.contrib.rnn.LSTMCell(rnn_size, initializer = weight_initializer)
    cell_bw = tf.contrib.rnn.LSTMCell(rnn_size, initializer = weight_initializer)
    #
    #cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=dropout_rate)
    #cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=dropout_rate)
    #
    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_sequence,
                                                    sequence_length = sequence_length,
                                                    time_major = True,
                                                    dtype = tf.float32,
                                                    scope = scope)
    #
    if concat:
        rnn_output = tf.concat(rnn_output, 2, name = 'output')
    else:
        rnn_output = tf.multiply(tf.add(rnn_output[0], rnn_output[1]), 0.5, name = 'output')
    #
    # to batch_major from time_major
    rnn_output = tf.transpose(rnn_output, [1,0,2])
    #
    return rnn_output
    #
   
def do_mask_padding_elems(x, mask):
    # make padding elements in x to -inf,
    # for next step of softmax,
    # mask: 1 for kept-elements, 0 for drop-elements,
    # x: [batch, time, dim], mask: [batch, time], when batch_major
    # or ... when time_major
    return tf.add(x, 1e30 * (tf.cast(mask, tf.float32) - 1) )

def att_pool_layer(seq, query, seq_mask, att_dim,
                   keep_prob=1.0, is_train=None, scope="att_pooling"):
    with tf.variable_scope(scope):
        # batch_major
        # seq: [B, T, D]
        # query: [B, DQ]
        # seq_mask: [B, T]
        d_seq = dropout(seq, keep_prob=keep_prob)
        seq_shape = tf.shape(seq)
        T = seq_shape[1]
        D = seq_shape[2]
        with tf.variable_scope("attention"):
            d_seq = tf.nn.relu(dense(d_seq, att_dim, use_bias=False, scope="att_dense"))
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

def build_graph(config):
    
    input_x = tf.placeholder(tf.int32, [None, None], name='input_x')
    input_y = tf.placeholder(tf.int64, [None], name='input_y')

    with tf.device('/cpu:0'):
        embedding = tf.get_variable('embedding',
                                    [config.vocab.size(), config.vocab.emb_dim],
                                    initializer=tf.constant_initializer(config.vocab.embeddings),
                                    trainable = config.emb_tune)
        embedding_inputs = tf.nn.embedding_lookup(embedding, input_x)
        
        # self._input_x = tf.placeholder(tf.int32, [None, None], name='input_x')
        # self._input_y = tf.placeholder(tf.int64, [None], name='input_y')
        
        seq_mask = tf.cast(tf.cast(input_x, dtype = tf.bool), dtype = tf.int32)
        seq_len = tf.reduce_sum(seq_mask, 1)

    with tf.name_scope("rnn"):
        
        seq_e = rnn_layer(embedding_inputs, seq_len, 256, config.keep_prob,
                          concat = True, scope = 'bi-lstm-1')
        
        B = tf.shape(seq_e)[0]
        query = tf.get_variable("query", [config.att_dim])
        query = tf.tile(tf.expand_dims(query, 0), [B, 1])

        feat = att_pool_layer(seq_e, query, seq_mask, config.att_dim,
                              config.keep_prob, is_train=None, scope="att_pooling")

    with tf.name_scope("score"):
        #
        fc = tf.contrib.layers.dropout(feat, config.keep_prob)
        fc = tf.layers.dense(fc, 128, name='fc1')            
        fc = tf.nn.relu(fc)
        
        fc = tf.contrib.layers.dropout(fc, config.keep_prob)
        fc = tf.layers.dense(fc, config.num_classes, name='fc2')
        logits = tf.nn.sigmoid(fc)
        
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

