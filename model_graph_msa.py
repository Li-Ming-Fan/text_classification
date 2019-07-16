# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 04:34:03 2019

@author: limingfan
"""


import tensorflow as tf


from zoo_nn import get_emb_positioned
from zoo_nn import get_tensor_expanded
from zoo_nn import gelu, dropout

from zoo_layers import get_position_emb_mat
from zoo_layers import layer_norm
from zoo_layers import att_pool_layer

from zoo_layers import multihead_attention_layer



class ModelGraph():
    
    @staticmethod
    def build_placeholder(settings):
        
        input_x = tf.placeholder(tf.int32, [None, None], name='input_x')
        input_y = tf.placeholder(tf.int64, [None], name='input_y')
        
        #        
        print(input_x)
        print(input_y)
        #
        input_tensors = (input_x, )
        label_tensors = (input_y, )
        #
        return input_tensors, label_tensors
    
    @staticmethod
    def build_inference(settings, input_tensors):
        
        input_x = input_tensors[0]

        #
        keep_prob = tf.get_variable("keep_prob", shape=[], dtype=tf.float32, trainable=False)
        #   
        with tf.device('/cpu:0'):
            emb_mat = tf.get_variable('embedding',
                                      [settings.vocab.size(), settings.vocab.emb_dim],
                                      initializer=tf.constant_initializer(settings.vocab.embeddings),
                                      trainable = settings.emb_tune,
                                      dtype=tf.float32)
            emb_dim = settings.vocab.emb_dim
            
        with tf.variable_scope("mask"):
            
            mask_t = tf.cast(tf.cast(input_x, dtype = tf.bool), dtype = tf.int32)
            # seq_len = tf.reduce_sum(mask_t, 1)
            mask_mat = get_tensor_expanded(mask_t, 1, tf.float32)
            
        with tf.variable_scope("emb"):
            
            posi_emb_max_len = 512
            posi_emb_dim = emb_dim
            posi_emb_model = 1024
            
            posi_emb_mat = get_position_emb_mat(posi_emb_max_len, posi_emb_dim,
                                                posi_emb_model)
            
            #
            emb_x = get_emb_positioned(input_x, emb_mat, posi_emb_mat)            
            emb_all_dim = emb_dim
            
            #
            seq_input = emb_x
            dim_all = emb_all_dim
            #
    
        #
        # transformers
        #
        num_layers_trans = 2
        num_heads = 2
        num_units = int(emb_dim / num_heads)
        #
        dim_middle = emb_dim * 2
        #
        activation_type = "gelu"
        #
        for lid in range(num_layers_trans):
    
            with tf.variable_scope("self_att_%d" % lid):
                
                # sublayer-0
                # attention
                seq = multihead_attention_layer(num_heads, num_units,
                                                seq_input, seq_input, seq_input,
                                                mask_mat = mask_mat,
                                                keep_prob = keep_prob)
                #
                # drop
                seq = dropout(seq, keep_prob=keep_prob)
                #
                # add & norm
                seq_input = layer_norm(seq_input + seq, scope="layer_norm_att")
                #
                
                #
                # sublayer-1
                # dense
                if activation_type == "relu":
                    act = tf.nn.relu
                else:
                    act = gelu            
                seq = tf.layers.dense(seq_input, dim_middle, activation = act)
                seq = tf.layers.dense(seq, dim_all)
                #
                # drop
                seq = dropout(seq, keep_prob=keep_prob)
                #
                # add & norm
                seq_input = layer_norm(seq_input + seq, scope="layer_norm_ff")
                #
    
        with tf.variable_scope("feat"):
            """ attention-pooling, 注意力加权采提
            """        
            B = tf.shape(seq_input)[0]
            query = tf.get_variable("query", [settings.att_dim],
                                    initializer = tf.ones_initializer())
            query = tf.tile(tf.expand_dims(query, 0), [B, 1])
    
            feat = att_pool_layer(query, seq_input, mask_t, settings.att_dim,
                                  keep_prob, scope="att_pooling")
            
        with tf.variable_scope("score"):
            #
            fc = tf.nn.dropout(feat, keep_prob)
            fc = tf.layers.dense(fc, 128, name='fc1')            
            fc = tf.nn.relu(fc)
            
            fc = tf.nn.dropout(fc, keep_prob)
            logits = tf.layers.dense(fc, settings.num_classes, name='fc2')
            
            normed_logits = tf.nn.softmax(logits, name='logits')
            
        #
        print(normed_logits)
        #
        output_tensors = normed_logits, logits
        #   
        return output_tensors
    
    @staticmethod
    def build_loss_and_metric(settings, output_tensors, label_tensors):
        
        normed_logits, logits = output_tensors
        input_y = label_tensors[0]
        
        
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
        print(loss)
        print(acc)
        #
        return loss, acc
        #

        