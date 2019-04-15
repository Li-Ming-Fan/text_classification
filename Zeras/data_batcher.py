# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
# Modifications Copyright 2018 Ming-Fan Li
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import time
import random

import queue as Queue

# from multiprocessing import Process        # True Process
# from multiprocessing.dummy import Process    # A wrapper of Thread

def get_generator_for_list(list_data):
    """
    """
    def gen(single_pass=True):
        while True:
            for item in list_data: yield item
            if single_pass: break
            #
    return gen

#
class DataBatcher(object):    
    """ This class is meant to be task-agnostic
    """    
    BATCH_QUEUE_MAX = 300       # max number of batches the batch_queue can hold
    BATCH_TIME_OUT = 5          # seconds
    EXAMPLE_TIME_OUT = 2
    
    def __init__(self, example_gen_or_list, batch_standardizer,
                 batch_size, single_pass, with_bucket=False, worker_type="thread"):
        """
        """
        if isinstance(example_gen_or_list, list):
            self.example_gen = get_generator_for_list(example_gen_or_list)
        else:
            self.example_gen = example_gen_or_list
        #
        self.batch_standardizer = batch_standardizer
        self.batch_size = batch_size
        self.with_bucket = with_bucket

        # queue
        self.single_pass = single_pass
        if single_pass:
            self.num_example_q_workers = 1 # one worker, so we read through the dataset just once
            self.num_batch_q_workers = 3   # num_workers to batch examples
            self.bucketing_cache_size = 1  # this essentially means no bucketing
        else:
            self.num_example_q_workers = 12 # num workers to fill example queue
            self.num_batch_q_workers = 12   # num workers to fill batch queue
            self.bucketing_cache_size = 200 # how many batches-worth of examples to load into cache before bucketing
        
        # worker_type
        self.worker_type = worker_type   # "process", "thread"
        #
        self.build_queue_and_workers(self.worker_type)
        #
        
    #
    def get_next_batch(self):
        """
        """        
        if self._batch_queue.qsize() == 0:
            print('batch_q_size: %i, example_q_size: %i' % (
                  self._batch_queue.qsize(), self._example_queue.qsize()) )
        #
        try:
            batch = self._batch_queue.get(timeout = self.BATCH_TIME_OUT) # get the next Batch
            return batch    
        except BaseException:
            return None
        
    #
    def build_queue_and_workers(self, worker_type="thread"):
        """
        """
        self.worker_type = worker_type
        #
        if worker_type == "process":
            from multiprocessing import Process        # True Process
        else:
            from multiprocessing.dummy import Process    # A wrapper of Thread
        #
        self.Process = Process
        #
        
        # queue
        self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self.batch_size)
        
        self._finished_reading = False
        self.count_put_examples = 0
        self.count_get_examples = 0 
        #
        # workers
        self._example_q_workers = []        
        for _ in range(self.num_example_q_workers):
            self._example_q_workers.append(self.Process(target=self.fill_example_queue))
            self._example_q_workers[-1].daemon = True
            self._example_q_workers[-1].start()
          
        self._batch_q_workers = []
        for _ in range(self.num_batch_q_workers):
            self._batch_q_workers.append(self.Process(target=self.fill_batch_queue))
            self._batch_q_workers[-1].daemon = True
            self._batch_q_workers[-1].start()

        if not self.single_pass:
            self._watch_worker = self.Process(target=self.watch_workers)
            self._watch_worker.daemon = True
            self._watch_worker.start()
            
    
    # internal executive functions
    # example
    def fill_example_queue(self):
        """
        """
        example_iter = self.example_gen(single_pass=self.single_pass)
        #
        while True:
            try:
                base_example = next(example_iter)
                # print(base_example)
                #
            except BaseException: # if there is no more example:                
                if self.single_pass:
                    print("fill_example_queue(), single_pass on, data finished, break loop")
                    self._finished_reading = True                    
                    break
                else:
                    raise Exception("single_pass mode is off but the example generator is out of data; ERROR!")
            #
            self._example_queue.put(base_example)
            self.count_put_examples += 1
            
    # batch    
    def fill_batch_queue(self):
        """
        """
        while True:           
            if self.with_bucket:                    
                inputs = []
                for _ in range(self.batch_size * self.bucketing_cache_size):
                    inputs.append(self._example_queue.get(timeout = self.EXAMPLE_TIME_OUT))
                inputs = sorted(inputs, key = lambda item: item.seq_len) 
                
                batches = []
                for i in range(0, len(inputs), self.batch_size):
                    batches.append(inputs[i:i + self.batch_size])
                if not self.single_pass:
                    random.shuffle(batches)
                for b in batches:
                    self._batch_queue.put(self.batch_standardizer(b))
            else:
                b = []
                flag_succeed = 1
                for eid in range(self.batch_size):
                    try:
                        example = self._example_queue.get(timeout = self.EXAMPLE_TIME_OUT)
                        b.append(example)
                        self.count_get_examples += 1
                    except BaseException:
                        flag_succeed = 0
                        break
                #
                if flag_succeed == 0:
                    for example in b:
                        self._example_queue.put(example)
                    print("fill_batch_queue(), examples reput, num_examples: %d." % len(b))
                    break
                #
                self._batch_queue.put(self.batch_standardizer(b)) 
                # print(b)
                #
    
    # watch
    def watch_workers(self):
        """
        """        
        while True:
            time.sleep(60)
            if self._finished_reading: break
            #                    
            for idx, t in enumerate(self._example_q_workers):
                if not t.is_alive(): # if the worker is dead
                    print('found example queue worker dead. Restarting.')
                    new_t = self.Process(target=self.fill_example_queue)
                    self._example_q_workers[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_workers):
                if not t.is_alive(): # if the worker is dead
                    print('found batch queue worker dead. Restarting.')
                    new_t = self.Process(target=self.fill_batch_queue)
                    self._batch_q_workers[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

# 
if __name__ == '__main__':
    
    pass

