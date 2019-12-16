# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 07:18:29 2019

@author: limingfan
"""

import tensorflow as tf


class ModelGraphTemplate():
    """ Template for model graph.
        Three static methods
        pb/debug tensor names
    """
    #    
    pb_input_names = {"input_x": "input_x:0"}    
    pb_output_names = {"logits": "vs_gpu/score/logits:0"}
    pb_save_names = ["vs_gpu/score/logits"]
    #
    debug_tensor_names = ["vs_gpu/score/logits:0",
                          "vs_gpu/score/logits:0"]
    #
    @staticmethod
    def build_placeholder(settings):
        """
        """
        input_x = None
        input_y = None
        #
        input_tensors = {"input_x": input_x}
        label_tensors = {"input_y": input_y}
        #
        return input_tensors, label_tensors
    
    #
    @staticmethod
    def build_inference(settings, input_tensors):
        """
        """
        #
        keep_prob = tf.get_variable("keep_prob", shape=[], dtype=tf.float32, trainable=False)
        print(keep_prob)
        #
        logits_normed = None
        #
        # print(logits_normed)
        # print(preds)
        #
        output_tensors = {"logits_normed": logits_normed}
        #   
        return output_tensors
    
    #
    @staticmethod
    def build_loss_and_metric(settings, output_tensors, label_tensors):
        """
        """            
        #
        loss_model = None
        metric = None
        #
        # print(loss)
        # print(metric)
        #
        loss_metric_tensors = {"loss_model": loss_model, "metric": metric}
        #
        return loss_metric_tensors
        #
