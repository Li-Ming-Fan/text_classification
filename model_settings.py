# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 21:12:23 2018

@author: limingfan
"""


class ModelSettings(object):
    def __init__(self, vocab, is_train = None):
        
        # model graph
        self.model_tag = None
        self.model_graph = None
        
        # model macro     
        self.min_seq_len = 5  #
        self.att_dim = 256
        #
        self.num_classes = 2
        
        # vocab
        self.vocab = vocab
        #
        self.emb_tune = 0  # 1 for tune, 0 for not
        self.keep_prob = 0.7
        
        # train
        self.num_epochs = 100     
        self.batch_size = 64
        self.batch_size_eval = 128 
        
        self.grad_clip = 8.0
        self.is_grad_clip = False
        self.learning_rate_base = 0.001         
        self.ratio_decay = 0.9
        self.patience_decay = 1000
        self.patience_stop = 5000
        
        self.save_per_batch = 100
        self.valid_per_batch = 100

        # inputs/outputs
        self.inputs_predict_name = ['input_x:0']     
        self.outputs_predict_name = ['score/logits:0']
        self.pb_outputs_name = ['score/logits']      
        self.is_train = is_train
        
        self.inputs_train_name = ['input_x:0', 'input_y:0']
        self.outputs_train_name = ['score/logits:0']
        self.loss_name = 'loss/loss:0'
        self.metric_name = 'accuracy/metric:0'
        self.use_metric = True
        #
        
    def check_settings(self):
        
        assert self.vocab is not None, 'vocab is None'
        
        """ 
            The following in this file is task-independent.
            
        """
        # assert
        if self.is_train is None:
            assert False, 'is_train not assigned'
        elif self.is_train == False:
            assert len(self.inputs_predict_name), 'inputs_predict_name is []'
            assert len(self.outputs_predict_name), 'outputs_predict_name is []'
        else:
            assert self.model_graph is not None, 'model_graph is None'
            assert len(self.inputs_train_name), 'inputs_train_name is []'
            assert len(self.outputs_train_name), 'outputs_train_name is []'
            assert self.loss_name is not None, 'loss_name is None'
            
        assert self.model_tag is not None, 'model_tag is None'
        
    def trans_info_to_dict(self):
                
        info_dict = {}
        for name,value in vars(self).items():
            if not isinstance(value, (int, float, str, bool, list, dict, tuple)):
                continue
            info_dict[str(name)] = value        
        return info_dict
        
        