# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 21:12:23 2018

@author: limingfan
"""

import os
import time


class ModelSettings(object):
    def __init__(self, vocab = None, is_train = None):

        # model macro     
        self.min_seq_len = 5  #
        self.att_dim = 256
        #
        self.num_classes = 2
        
        # vocab
        self.vocab = vocab
        self.emb_tune = 0  # 1 for tune, 0 for not
        
        # 
        # model graph
        self.model_tag = None
        self.model_graph = None
        
        # train
        self.num_epochs = 100     
        self.batch_size = 64
        self.batch_size_eval = 128 
        
        self.reg_lambda = 0.0001  # 0.0, 0.0001
        self.grad_clip = 8.0  # 0.0, 5.0, 8.0
        self.keep_prob = 0.7  # 1.0, 0.7, 0.5
        
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
        # save and log, if not set, default values will be used.
        self.model_dir = None
        self.model_name = None
        self.pb_file = None
        self.log_dir = None
        self.log_path = None
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
        if self.use_metric:
            assert self.metric_name is not None, 'metric_name is None'
            
        assert self.model_tag is not None, 'model_tag is None'
        
        # model dir
        if self.model_dir is None: self.model_dir = './model_' + self.model_tag
        if self.model_name is None: self.model_name = 'model_' + self.model_tag
        if self.pb_file is None: self.pb_file = os.path.join(self.model_dir + '_best',
                                                             self.model_name + '.pb')
        
        # log
        if self.log_dir is None: self.log_dir = './log'
        str_datetime = time.strftime("%Y-%m-%d-%H-%M")       
        if self.log_path is None: self.log_path = os.path.join(self.log_dir,
                                                   self.model_name + "_" + str_datetime +".txt")
        
        #
        self.display()
        #
        
    def display(self):
        
        print()
        for name,value in vars(self).items():
            if not isinstance(value, (int, float, str, bool, list, dict, tuple)):
                continue
            print(str(name) + ': ' + str(value))
        print()
        
    def trans_info_to_dict(self):
                
        info_dict = {}
        for name,value in vars(self).items():
            if not isinstance(value, (int, float, str, bool, list, dict, tuple)):
                continue
            info_dict[str(name)] = value        
        return info_dict
        
if __name__ == '__main__':
    
    sett = ModelSettings('vocab_placeholder', False)
    
    sett.model_tag = 'cnn'
    
    #print(dir(sett))    
    #l = [i for i in dir(sett) if inspect.isbuiltin(getattr(sett, i))]
    #l = [i for i in dir(sett) if inspect.isfunction(getattr(sett, i))]
    #l = [i for i in dir(sett) if not callable(getattr(sett, i))]
    
    sett.check_settings()
    
    print(sett.__dict__.keys())
    print()
    

    