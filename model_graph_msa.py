# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 04:34:03 2019

@author: limingfan
"""


import tensorflow as tf


from zoo_layers import dropout

from zoo_layers import get_posi_emb
from zoo_layers import att_qkv_layer, att_pool_layer


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
                                      trainable = settings.emb_tune)
            
        with tf.variable_scope("emb"):
            
            emb_dim = settings.vocab.emb_dim
            
            emb_x = tf.nn.embedding_lookup(emb_mat, input_x)
            
            mask_t = tf.cast(tf.cast(input_x, dtype = tf.bool), dtype = tf.int32)
            # seq_len = tf.reduce_sum(mask_t, 1)
            
        with tf.variable_scope("posi_emb"):
    
            d_posi_emb = 64
            d_model = 1024
            
            posi_emb_x = get_posi_emb(input_x, d_posi_emb, d_model)
            
            emb_x = tf.concat([emb_x, posi_emb_x], -1)
            
            emb_all_dim = emb_dim + d_posi_emb * 2
            
            #
            enc_t = emb_x
            enc_dim = emb_all_dim
    
        #
        # transformers
        #
        num_layers_trans = 2
        #
        for lid in range(num_layers_trans):
    
            with tf.variable_scope("self_att_%d" % lid):
    
                num_head = 2
                num_hidden = int(128 / num_head)
    
                sat_t = []
                for idx in range(num_head):
                    sat_t_c = att_qkv_layer(enc_t, enc_t, enc_t, mask_t, num_hidden,
                                            keep_prob = keep_prob, scope = "t_%d" % idx)
                    sat_t.append(sat_t_c)
                #
                sat_t = tf.concat(sat_t, -1)
                #
                # add & norm
                sat_t = dropout(sat_t, keep_prob=keep_prob)
                sat_t = tf.layers.dense(sat_t, enc_dim)
                #
                enc_t = enc_t + sat_t
                enc_t = tf.contrib.layers.layer_norm(enc_t)
                #
                """
                # dense
                ffn_t = dropout(enc_t, keep_prob=keep_prob)
                ffn_t = tf.layers.dense(ffn_t, enc_dim, activation=tf.nn.relu)
                ffn_t = tf.layers.dense(ffn_t, enc_dim)
                #
                # add & norm
                enc_t = enc_t + ffn_t
                enc_t = tf.contrib.layers.layer_norm(enc_t)
                #
                """
    
        with tf.variable_scope("feat"):
            """ attention-pooling, 注意力加权采提
            """        
            B = tf.shape(enc_t)[0]
            query = tf.get_variable("query", [settings.att_dim],
                                    initializer = tf.ones_initializer())
            query = tf.tile(tf.expand_dims(query, 0), [B, 1])
    
            feat = att_pool_layer(query, enc_t, mask_t, settings.att_dim,
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

        