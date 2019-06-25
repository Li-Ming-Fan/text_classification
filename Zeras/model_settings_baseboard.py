# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 21:12:23 2018

@author: limingfan
"""

import os
import time
import json

import logging


class ModelSettingsBaseboard(object):
    """
    """
    def __init__(self):

        # model
        self.model_tag = None     # str
        self.is_train = None      # bool
        
        # model graph macro parameters
        #
        
        # other macro parameters
        #
        
        #
        # train
        self.gpu_available = "0"      # could be specified in args
        self.gpu_batch_split = None   # list, for example, [12, 20]; if None, batch split evenly
        #
        self.gpu_mem_growth = True
        self.log_device = False
        self.soft_placement = True

        self.num_epochs = 100     
        self.batch_size = 32
        self.batch_size_eval = 6
        self.max_batches_eval = 20
        
        self.reg_lambda = 0.001  # 0.0, 0.01
        self.reg_exclusions = ["embedding", "bias", "layer_norm", "LayerNorm"]
        self.grad_clip = 8.0  # 0.0, 5.0, 8.0, 2.0
        self.keep_prob = 0.8  # 1.0, 0.7, 0.5
        self.label_smoothing = 0.01
        
        self.optimizer_type = 'adam'  # adam, momentum, sgd, customized
        self.momentum = 0.9
        self.learning_rate_base = 0.001   #        
        self.learning_rate_minimum = 0.000001
        self.warmup_steps = 1000
        self.decay_steps = 1000
        self.decay_rate = 0.99
        self.staircase = True
        
        self.check_period_batch = 100
        self.valid_period_batch = 100
        #

        # inputs/outputs
        self.vs_str_multi_gpu = "vs_multi_gpu"
        #
        self.inputs_predict_name = ['src_seq:0', 'src_seq_mask:0']
        self.outputs_predict_name = ['vs_multi_gpu/logits:0']
        self.pb_outputs_name = ['vs_multi_gpu/logits']
                
        self.inputs_train_name = ['src_seq:0', 'src_seq_mask:0',
                                  'dcd_seq:0', 'dcd_seq_mask:0',
                                  'labels_seq:0', 'labels_mask:0']
        self.outputs_train_name = ['vs_multi_gpu/logits:0']
        self.use_metric = True
        
        self.debug_tensors_name = ['vs_multi_gpu/loss/loss:0',
                                   'vs_multi_gpu/logits:0',
                                   'vs_multi_gpu/preds:0'
                                   #'encoder/encoder_rnn_1/bw/bw/sequence_length:0',
                                   #'inputs_len:0'
                                   ]
        
        #
        # save and log, if not set, default values will be used.
        self.base_dir = './task_results'
        self.model_dir = None
        self.model_name = None
        self.model_dir_best = None
        self.pb_file = None
        self.log_dir = None
        self.log_path = None
        #
    
    def check_settings(self):
        """ assert and make directories
        """        
        # assert         
        assert self.is_train is not None, 'is_train not assigned'
        
        # gpu
        num_gpu = len(self.gpu_available.split(","))
        if num_gpu > 1:
            #
            if self.gpu_batch_split is None:
                self.gpu_batch_split = [self.batch_size//num_gpu] * num_gpu
            #
            str_info = "make sure that num_gpu == len(self.gpu_batch_split)"
            assert num_gpu == len(self.gpu_batch_split), str_info
            str_info = "make sure that self.batch_size == sum(self.gpu_batch_split)"
            assert self.batch_size == sum(self.gpu_batch_split), str_info
            
        # directories
        if self.model_dir is None:
            self.model_dir = os.path.join(self.base_dir, 'model_' + self.model_tag)
        if self.model_dir_best is None: self.model_dir_best = self.model_dir + "_best"
        if self.log_dir is None: self.log_dir = os.path.join(self.base_dir, 'log')
        #
        if not os.path.exists(self.base_dir): os.mkdir(self.base_dir)
        if not os.path.exists(self.model_dir): os.mkdir(self.model_dir)
        if not os.path.exists(self.model_dir_best): os.mkdir(self.model_dir_best)
        if not os.path.exists(self.log_dir): os.mkdir(self.log_dir)
        #
        # files
        if self.model_name is None: self.model_name = 'model_' + self.model_tag
        if self.pb_file is None: self.pb_file = os.path.join(self.model_dir_best, 
                                                             'model_saved.pb')
        #
        # logger
        str_datetime = time.strftime("%Y-%m-%d-%H-%M")       
        if self.log_path is None: self.log_path = os.path.join(
                self.log_dir, self.model_name + "_" + str_datetime +".txt")
        #
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
        
    def close_logger(self):
        for item in self.logger.handlers: item.close()
    
    #
    def display(self):
        """
        """        
        print()
        for name,value in vars(self).items():
            if not isinstance(value, (int, float, str, bool, list, dict, tuple)):
                continue
            print(str(name) + ': ' + str(value))
        print()
    
    #
    def trans_info_to_dict(self):
        """
        """                
        info_dict = {}
        for name,value in vars(self).items():
            if not isinstance(value, (int, float, str, bool, list, dict, tuple)):
                continue
            info_dict[str(name)] = value        
        return info_dict
    
    def assign_info_from_dict(self, info_dict):
        """
        """
        for key in info_dict:
            value = info_dict[key]
            setattr(self, key, value)
        #
        
    def save_to_json_file(self, file_path):
        """
        """
        info_dict = self.trans_info_to_dict()
        #
        info_dict["model_dir"] = None
        info_dict["model_name"] = None
        info_dict["model_dir_best"] = None
        info_dict["pb_file"] = None
        info_dict["log_dir"] = None
        info_dict["log_path"] = None
        #
        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(info_dict, fp, ensure_ascii=False, indent=4)
        #       
    
    def load_from_json_file(self, file_path):
        """
        """
        if file_path is None:
            print("model settings file: %s NOT found, using default settings"
                  % file_path)
            return
        #
        with open(file_path, "r", encoding="utf-8") as fp:
            info_dict = json.load(fp)
        #
        self.assign_info_from_dict(info_dict)
        #
        
        
    
        
if __name__ == '__main__':
    
    sett = ModelSettingsBaseboard()
    
    sett.model_tag = 'cnn'
    sett.is_train = False
    
    #print(dir(sett))
    #l = [i for i in dir(sett) if inspect.isbuiltin(getattr(sett, i))]
    #l = [i for i in dir(sett) if inspect.isfunction(getattr(sett, i))]
    #l = [i for i in dir(sett) if not callable(getattr(sett, i))]
    
    sett.check_settings()
    
    print(sett.__dict__.keys())
    print()
    
    info_dict = sett.trans_info_to_dict()
    print(info_dict)
    print()
    
    #
    info_dict["model_tag"] = "transformer"
    sett.assign_info_from_dict(info_dict)
    
    #
    info_dict = sett.trans_info_to_dict()
    print(info_dict)
    print()
    #
    
    file_path = "./settings_template.json"
    sett.save_to_json_file(file_path)    
    sett.load_from_json_file(file_path)
    
    #
    info_dict = sett.trans_info_to_dict()
    print(info_dict)
    print()
    #

    #    
    sett.close_logger()
    #