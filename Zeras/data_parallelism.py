# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 04:11:16 2018

@author: limingfan
"""

import os
import pickle

# from multiprocessing import Process        # True Process
# from multiprocessing.dummy import Process  # A wrapper of Thread
# 多进程需要在 cmd 运行 python 文件.py，这样才有多进程的效果。


class DataParallelism(object):
    """
    """
    def __init__(self, num_workers, worker_type="thread"):
        """
        """
        self.num_workers = num_workers
        self.worker_type = worker_type  # "process", "thread"
        
    def do_processing(self, list_data, process_pipeline, args_rem):
        """
        """
        # data
        list_data_split = DataParallelism.split_data_list(list_data,
                                                          self.num_workers)
        # worker
        if self.worker_type == "process":
            from multiprocessing import Process        # True Process
        else:
            from multiprocessing.dummy import Process  # A wrapper of Thread
        self.worker = Process
        #
        print('parent process: %s.' % os.getpid())
        self._workers = []
        for idx in range(self.num_workers):
            p_curr = self.worker(target = process_pipeline,
                                 args = (list_data_split[idx],
                                         idx, args_rem) )
            p_curr.daemon = True
            #
            self._workers.append(p_curr)
            print("worker %d created" % idx)
            #
        #
        # doing
        for idx in range(self.num_workers):
            self._workers[idx].start()
        #
        for idx in range(self.num_workers):
            self._workers[idx].join()
        #
        print('data processing all finished')
        #
        
    
    @staticmethod
    def split_data_list(data_list, num_split):
        """
        """
        num_data_all = len(data_list)
        num_per_worker = num_data_all // num_split
        #
        list_data_split = []
        posi_start = 0
        posi_end = num_per_worker
        for idx in range(num_split):
            list_curr = data_list[posi_start:posi_end]
            list_data_split.append(list_curr)
            posi_start = posi_end
            posi_end += num_per_worker
        #
        if posi_end < num_data_all:
            list_data_split[-1].extend(data_list[posi_end:])
        #
        list_num_data = [len(item) for item in list_data_split]
        print("list_files split: {}".format(list_num_data))
        #
        return list_data_split
        #
        
    @staticmethod
    def get_files_with_ext(path, str_ext, flag_walk=False):
        """
        """
        list_all = []
        if flag_walk:
            # 列出目录下，以及各级子目录下，所有目录和文件
            for (root, dirs, files) in os.walk(path):            
                for filename in files:
                    file_path = os.path.join(root, filename) 
                    list_all.append(file_path)
        else:
            # 列出当前目录下，所有目录和文件
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                list_all.append(file_path)
        #
        file_list = [item for item in list_all if item.endswith(str_ext)]
        return file_list
        #

#
def save_data_to_pkl(data, file_path):
    with open(file_path, 'wb') as fp:
        pickle.dump(data, fp)
        
def load_data_from_pkl(file_path):
    with open(file_path, 'rb') as fp:
        data = pickle.load(fp)
    return data

    
    
#
if __name__ == '__main__':
    
    pass

