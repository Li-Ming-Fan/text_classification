# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 21:12:23 2018

@author: limingfan
"""

from Zeras.model_settings_template import ModelSettingsTemplate

class ModelSettings(ModelSettingsTemplate):
    """
    """
    def __init__(self):
        """
        """
        super(ModelSettings, self).__init__()
        
        # model graph
        self.model_tag = None
        self.model_graph = None
        self.is_train = None        
        #
        
        # data macro     
        self.min_seq_len = 5      #
        self.max_seq_len = 300   #
        
        # model macro
        self.att_dim = 128
        #
        self.num_classes = 2
        
        # vocab
        self.vocab = None
        self.emb_dim = 64
        self.emb_tune = 0  # 1 for tune, 0 for not
        
        # train
        self.gpu_available = "0"  # specified in args
        self.gpu_batch_split = [12, 20]   # list; if None, batch split evenly
        #
        self.gpu_mem_growth = True
        self.log_device = False
        self.soft_placement = True
        
        self.with_bucket = False
        
        self.num_epochs = 100     
        self.batch_size = 32
        self.batch_size_eval = 32
        
        self.reg_lambda = 0.0001  # 0.0, 0.0001
        self.grad_clip = 5.0  # 0.0, 5.0, 8.0, 2.0
        self.keep_prob = 0.8  # 1.0, 0.7, 0.5
        
        self.optimizer_type = 'adam'  # adam, momentum, sgd
        self.momentum = 0.9
        self.learning_rate_base = 0.001   #
        self.ratio_decay = 0.9
        self.patience_decay = 3000
        self.learning_rate_minimum = 0.0001
        
        self.save_period_batch = 100
        self.valid_period_batch = 100
        #
        
        # inputs/outputs
        self.vs_str_multi_gpu = "vs_multi_gpu"
        #
        self.inputs_predict_name = ['input_x:0']     
        self.outputs_predict_name = ['vs_multi_gpu/score/logits:0']
        self.pb_outputs_name = ['vs_multi_gpu/score/logits']
        
        self.inputs_train_name = ['input_x:0', 'input_y:0']
        self.outputs_train_name = ['vs_multi_gpu/score/logits:0']
        self.use_metric = True
        
        self.debug_tensors_name = ['vs_multi_gpu/score/logits:0']
        #
       
        #
        # save and log, if not set, default values will be used.
        self.base_dir = '.'
        self.model_dir = None
        self.model_name = None
        self.pb_file = None
        self.log_dir = None
        self.log_path = None
        #

        
if __name__ == '__main__':
    
    sett = ModelSettings()
    
    sett.model_tag = 'cnn'
    sett.is_train = False
    
    #print(dir(sett))    
    #l = [i for i in dir(sett) if inspect.isbuiltin(getattr(sett, i))]
    #l = [i for i in dir(sett) if inspect.isfunction(getattr(sett, i))]
    #l = [i for i in dir(sett) if not callable(getattr(sett, i))]
    
    sett.check_settings()
    
    print(sett.__dict__.keys())
    print()
    
    sett.close_logger()
    
