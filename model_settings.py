# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 21:12:23 2018

@author: limingfan
"""

from Zeras.model_settings_baseboard import ModelSettingsBaseboard


class ModelSettings(ModelSettingsBaseboard):
    """
    """
    def __init__(self):
        """
        """
        super(ModelSettings, self).__init__()
        
        # model graph
        self.model_tag = None
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
        self.tokens_file = './vocab/vocab_tokens.txt'
        self.emb_file = None
        
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
        self.batch_size_eval = 1
        self.max_batches_eval = 20000
        
        self.reg_lambda = 0.0  # 0.0, 0.01
        self.grad_clip = 0.0  # 0.0, 5.0, 8.0, 2.0
        self.keep_prob = 0.8  # 1.0, 0.7, 0.5
        self.label_smoothing = 0.01
        
        self.optimizer_type = 'adam'  # adam, momentum, sgd, customized
        self.momentum = 0.9
        self.learning_rate_base = 0.001   #
        self.learning_rate_minimum = 0.000001
        self.warmup_steps = 1000
        self.decay_steps = 5000
        self.decay_rate = 0.99
        self.staircase = True
        
        self.check_period_batch = 100
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
        
        self.debug_tensors_name = ["vs_multi_gpu/posi_emb/concat:0",
                                   "vs_multi_gpu/self_att_0/t_0/MatMul_1:0"]
        #
       
        #
        # save and log, if not set, default values will be used.
        self.base_dir = './task_cls'
        self.model_dir = None
        self.model_name = None
        self.model_dir_best = None
        self.pb_file = None
        self.log_dir = None
        self.log_path = None
        #

        
if __name__ == '__main__':
    
    sett = ModelSettings()
    #
    sett.model_tag = 'cnn'
    sett.is_train = False
    #
    sett.check_settings()
    #
    
    #
    info_dict = sett.trans_info_to_dict()
    print("original:")
    print(info_dict)
    print()
    #
    info_dict["model_tag"] = "msa"
    sett.assign_info_from_dict(info_dict)
    #
    info_dict = sett.trans_info_to_dict()
    print("assigned:")
    print(info_dict)
    print()
    #
    
    #
    file_path = "./temp_settings.json"
    sett.save_to_json_file(file_path)    
    sett.load_from_json_file(file_path)
    #
    info_dict = sett.trans_info_to_dict()
    print("saved then loaded:")
    print(info_dict)
    print()
    #

    #    
    sett.close_logger()
    #
