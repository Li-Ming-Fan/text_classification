# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 13:49:51 2018

@author: limingfan

"""

import tensorflow as tf


debug_tensor_name = "to_be_assigned"


def build_model_graph(settings, data):
    
    input_x, input_y = data

    with tf.device('/cpu:0'):
        
        input_x = tf.identity(input_x, name = "input_x")
        input_y = tf.identity(input_y, name = "input_y")
        
        #
        emb_mat = tf.get_variable('embedding',
                                  [settings.vocab.size(), settings.vocab.emb_dim],
                                  initializer=tf.constant_initializer(settings.vocab.embeddings),
                                  trainable = settings.emb_tune)
        seq_emb = tf.nn.embedding_lookup(emb_mat, input_x)

    with tf.name_scope("cnn"):
        #
        conv1_5 = tf.layers.conv1d(seq_emb, 128, 5, padding='same', name='conv1_5')
        conv1_3 = tf.layers.conv1d(seq_emb, 128, 3, padding='same', name='conv1_3')
        conv1_2 = tf.layers.conv1d(seq_emb, 128, 2, padding='same', name='conv1_2')
        
        conv1 = tf.concat([conv1_5, conv1_3, conv1_2], -1)
        
        conv2_5 = tf.layers.conv1d(conv1, 128, 5, name='conv2_5')
        conv2_3 = tf.layers.conv1d(conv1, 128, 3, name='conv2_3')
        conv2_2 = tf.layers.conv1d(conv1, 128, 2, name='conv2_2')
        
        feat1 = tf.reduce_max(conv2_5, reduction_indices=[1], name='feat1')
        feat2 = tf.reduce_max(conv2_3, reduction_indices=[1], name='feat2')
        feat3 = tf.reduce_max(conv2_2, reduction_indices=[1], name='feat3')
        
        feat = tf.concat([feat1, feat2, feat3], 1)

    with tf.name_scope("score"):
        #
        fc = tf.nn.dropout(feat, settings.keep_prob)
        fc = tf.layers.dense(fc, 128, name='fc1')            
        fc = tf.nn.relu(fc)
        
        fc = tf.nn.dropout(fc, settings.keep_prob)
        logits = tf.layers.dense(fc, settings.num_classes, name='fc2')
        # logits = tf.nn.sigmoid(fc)
        
        normed_logits = tf.nn.softmax(logits, name='logits')
        
    with tf.name_scope("loss_infer"):
        #
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,
                                                                       labels = input_y)
        loss_infer = tf.reduce_mean(cross_entropy, name = 'loss_infer')

    with tf.name_scope("accuracy"):
        #
        y_pred_cls = tf.argmax(logits, 1, name='pred_cls')
        correct_pred = tf.equal(input_y, y_pred_cls)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = 'metric')
    
    #
    if settings.keep_prob < 1.0:  # train, eval
        print(input_x)
        print(input_y)
        #
        print(normed_logits)
        print(acc)
        print(loss_infer)
        print()
    #

    #
    debug_tensor = normed_logits
    #
    
    #
    global debug_tensor_name
    debug_tensor_name = debug_tensor.name
    #
    if settings.keep_prob < 1.0:
        print('debug_tensor_name: ' + debug_tensor_name)
        print(debug_tensor)
        print()
    #
    
    #
    return normed_logits, acc, loss_infer
    # results, metric, loss
    #


def debug_the_model(model, data_batches):
    
    model.log_info("begin debug ...")    
    model_graph, model_sess = model.get_model_graph_and_sess()
    
    idx_batch = 0
    
    data_batch = data_batches[idx_batch]
    
    print()
    for item in zip(*data_batch):
        #
        print(item[-1])
        print(model.vocab.convert_ids_to_tokens(item[0]) )
    print()
    
    #
    global debug_tensor_name
    tensor = model_graph.get_tensor_by_name(debug_tensor_name)
    #
    tensor_v = model_sess.run(tensor, feed_dict = model.feed_data_train(data_batch))    
    print(tensor_v)
    print(tensor_v.shape)
    
    loss = model_graph.get_tensor_by_name('loss/loss:0')
    loss_v = model_sess.run(loss, feed_dict = model.feed_data_train(data_batch))    
    print(loss_v)
    
    return tensor_v
    #

