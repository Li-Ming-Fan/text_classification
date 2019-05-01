# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 07:18:29 2019

@author: limingfan
"""

# import tensorflow as tf


class ModelGraphTemplate():
    """ Template for model graph.
        Three static methods
    """    
    @staticmethod
    def build_placeholder(settings):
        """
        """        
        # src_seq = tf.placeholder(tf.int32, [None, None], name='src_seq')  # id in vocab
        # src_seq_mask = tf.placeholder(tf.int32, [None, None], name='src_seq_mask')
        
        # dcd_seq = tf.placeholder(tf.int32, [None, None], name='dcd_seq')  # id in vocab
        # dcd_seq_mask = tf.placeholder(tf.int32, [None, None], name='dcd_seq_mask')

        # labels_seq = tf.placeholder(tf.int32, [None, None], name='labels_seq')  # id in vocab
        # labels_mask = tf.placeholder(tf.int32, [None, None], name='labels_mask')
        
        #
        # input sequence: could not prefix and suffix, when preparing examples
        # label sequence: suffix with a [end] token, then do [pad].
        #
        # decoder input seq: prefix with [start], suffix with [end], then do [pad].
        #
        # print(src_seq)
        #
        # input_tensors = (src_seq, src_seq_mask, dcd_seq, dcd_seq_mask)
        # label_tensors = (labels_seq, labels_mask)
        input_tensors = None
        label_tensors = None
        #
        return input_tensors, label_tensors
    
    #
    @staticmethod
    def build_inference(settings, input_tensors):
        """
        """
        #
        # print(logits_normed)
        # print(preds)
        #
        # output_tensors = logits_normed, logits, preds
        output_tensors = None
        #   
        return output_tensors
    
    #
    @staticmethod
    def build_loss_and_metric(settings, output_tensors, label_tensors):
        """
        """            
        #
        # print(loss)
        # print(metric)
        #
        loss = None
        metric = None
        #
        return loss, metric
        #
