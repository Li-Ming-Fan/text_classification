# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 21:12:23 2018

@author: limingfan
"""

import os
import time
import logging


class ModelSettings(object):
    def __init__(self, vocab = None, is_train = None):
        
        # model graph
        self.model_tag = None
        self.model_graph = None        
        #
        
        # data macro     
        self.min_seq_len = 5      #
        self.max_seq_len = 300   #
        
        # model macro
        self.att_dim = 128
        #
        self.num_classes = 2
        
        # vocab
        self.vocab = vocab
        self.emb_tune = 0  # 1 for tune, 0 for not
        
        # train
        self.gpu_mem_growth = True
        self.log_device = False
        
        self.with_bucket = False
        self.is_train = is_train
        
        self.num_epochs = 100     
        self.batch_size = 32
        self.batch_size_eval = 32 
        
        self.reg_lambda = 0.0001  # 0.0, 0.0001
        self.grad_clip = 2.0  # 0.0, 5.0, 8.0, 2.0
        self.keep_prob = 0.8  # 1.0, 0.7, 0.5
        
        self.optimizer_type = 'adam'  # adam, momentum, sgd
        self.momentum = 0.9
        self.learning_rate_base = 0.001   #
        self.ratio_decay = 0.9
        self.patience_decay = 3000
        self.learning_rate_minimum = 0.000001
        
        self.save_period_batch = 100
        self.valid_period_batch = 100
        #
        
        # inputs/outputs        
        self.inputs_train_name = ['input_x:0', 'input_y:0']
        self.outputs_train_name = ['score/logits:0']
        
        self.inputs_predict_name = ['input_x:0']     
        self.outputs_predict_name = ['score/logits:0']
        
        self.pb_outputs_name = ['score/logits']
        self.is_train = is_train
        #
        self.loss_name = 'loss/loss:0'
        self.metric_name = 'accuracy/metric:0'
        self.use_metric = True
        #
        self.debug_tensors_name = ['score/logits:0']
        #
       
        #
        # save and log, if not set, default values will be used.
        self.model_dir = None
        self.model_name = None
        self.pb_file = None
        self.log_dir = None
        self.log_path = None
        #
    
    def check_settings(self):
        """ assert and make directories
        """
        assert self.vocab is not None, 'vocab is None'
        
        # assert         
        assert self.is_train is not None, 'is_train not assigned'               
        assert self.model_tag is not None, 'model_tag is None'
        
        # model dir
        if self.model_dir is None: self.model_dir = './model_' + self.model_tag
        if self.model_name is None: self.model_name = 'model_' + self.model_tag
        if self.pb_file is None: self.pb_file = os.path.join(self.model_dir + '_best',
                                                             self.model_name + '.pb')
        
        if not os.path.exists(self.model_dir): os.mkdir(self.model_dir)
        if not os.path.exists(self.model_dir + '_best'): os.mkdir(self.model_dir + '_best')
        
        # log dir
        if self.log_dir is None: self.log_dir = './log'
        str_datetime = time.strftime("%Y-%m-%d-%H-%M")       
        if self.log_path is None: self.log_path = os.path.join(
                self.log_dir, self.model_name + "_" + str_datetime +".txt")
        
        if not os.path.exists(self.log_dir): os.mkdir(self.log_dir)
        #
        # logger
        self.logger = logging.getLogger(self.log_path)  # use log_path as log_name
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        # self.logger.info('test')
        #
        self.display()
        #
        
    def create_or_reset_log_file(self):        
        with open(self.log_path, 'w', encoding='utf-8'):
            pass
        
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
    
