# -*- coding:utf8 -*-

import numpy as np
import struct
import re

class Vocab(object):
    """
    """    
    pad_token = '[PAD]'
    unk_token = '[UNK]'
    num_predefined_tokens = 2
    
    delimiter_str = '[STRSEP]'
    
    def __init__(self, initial_tokens = [], lower = False):
        """
        """
        self.dict_id2token = {}
        self.dict_token2id = {}
        self.dict_token_cnt = {}
        self.lower = lower

        self.emb_dim = 64
        self.embeddings = None
        
        self.initial_tokens = initial_tokens

        self._add_predefined_and_initial_tokens()
        #

    def _add_predefined_and_initial_tokens(self):
        """
        """        
        self.add(self.pad_token, 10000)  # make pad_token id: 0
        self.add(self.unk_token, 10000)
        #
        for token in self.initial_tokens:
            self.add(token, cnt = 10000)
        #

    def size(self):
        return len(self.dict_id2token)

    def get_id(self, token):
        token = token.lower() if self.lower else token
        try:
            return self.dict_token2id[token]
        except KeyError:
            return self.dict_token2id[self.unk_token]

    def get_token(self, idx):
        try:
            return self.dict_id2token[idx]
        except KeyError:
            return self.unk_token

    def add(self, token, cnt=1):
        #
        token = token.lower() if self.lower else token
        if token in self.dict_token2id:
            idx = self.dict_token2id[token]
            # print('same token: %s' % token)
        else:
            idx = len(self.dict_id2token)
            self.dict_id2token[idx] = token
            self.dict_token2id[token] = idx
        if cnt > 0:
            if token in self.dict_token_cnt:
                self.dict_token_cnt[token] += cnt
            else:
                self.dict_token_cnt[token] = cnt
        return idx
        
    def add_tokens_from_vocab(self, vocab):
        """ add tokens from another vocab instance
        """
        for token in vocab.dict_token_cnt:
            self.add(token, vocab.dict_token_cnt[token])
        
    def add_tokens_from_corpus(self, corp):
        """ add tokens from corpus (list)
            with each item in the list as a list of tokens 
        """
        for item in corp:
            for word in item:
                self.add(word)
                
    def add_tokens_from_file(self, file_path):
        """ add tokens from file
            with one token in one line
        """
        fp = open(file_path, 'r', encoding='utf-8')
        lines = fp.readlines()
        fp.close()
        self.add_tokens_from_lines(lines)
        
    def add_tokens_from_lines(self, lines):
        """ add tokens from lines
            with one token in one line
        """
        token = ' '      
        for idx, line in enumerate(lines):
            if line.startswith(' '):
                self.add(' ', 10000)
                print('WARNING: blank token')
                continue
            #            
            str_arr = line.strip().split()  #
            # print(str_arr)
            if len(str_arr) == 0:
                print('WARNING: len(str_arr) == 0 with prev token: %s' % token)
                # continue
            #
            token = str_arr[0]      # .strip()
            # token = line.rstrip('\n')
            # print(token)
            if len(str_arr) > 1:
                cnt = int(str_arr[1].strip())
                self.add(token, cnt)
            else:
                self.add(token)
        #
        print('num lines: %d' % (idx + 1) )
        print('num tokens after loading: %d' % len(self.dict_id2token))
        #
        
    def save_tokens_to_file(self, file_path):
        """ save tokens to file
            with one token in one line
        """
        with open(file_path, 'w', encoding='utf-8') as fp:
            for tid in range(self.size()):
                token = self.dict_id2token[tid]
                fp.write(token + ' ' + str(self.dict_token_cnt[token]) + '\n')
                
    def save_tokens_to_file_no_count(self, file_path):
        """ save tokens to file
            with one token in one line
        """
        with open(file_path, 'w', encoding='utf-8') as fp:
            for tid in range(self.size()):
                token = self.dict_id2token[tid]
                fp.write(token + '\n')
    
    #
    def filter_tokens_by_cnt(self, min_cnt):
        #
        #filtered_tokens = [token for token in self.token2id if self.token_cnt[token] >= min_cnt]
        filtered_tokens = [self.dict_id2token[idd] for idd in range(self.size())
                                   if self.dict_token_cnt[self.dict_id2token[idd]] >= min_cnt]
        # rebuild the token ~ id map
        self.dict_token2id = {}
        self.dict_id2token = {}
        self._add_predefined_and_initial_tokens()
        for token in filtered_tokens:
            self.add(token, cnt=0)
            
    def remove_specified_tokens(self, list_tokens_to_remove, regex_to_remove=[]):
        """ pattern_int = re.compile(r'^[0-9]+$')
            list_patterns = [re.compile(item) for item in regx_to_remove]
        """
        list_patterns = [re.compile(item) for item in regex_to_remove]
        #
        filtered_tokens = []
        for idd in range(self.size()):
            token = self.dict_id2token[idd]
            #
            if token in list_tokens_to_remove: continue
            for pattern in list_patterns:
                if pattern.match(token): continue
            #
            filtered_tokens.append(token)
            #
        # rebuild the token ~ id map
        self.dict_token2id = {}
        self.dict_id2token = {}
        self._add_predefined_and_initial_tokens()
        for token in filtered_tokens:
            self.add(token, cnt=0)
        
    #
    def randomly_init_embeddings(self, emb_dim):
        #
        self.emb_dim = emb_dim
        self.embeddings = np.random.rand(self.size(), emb_dim).astype(np.float32)
        #for token in [self.pad_token, self.unk_token]:
        #for token in [self.pad_token]: self.embeddings[self.get_id(token)] = np.zeros([self.emb_dim])
        
    def load_pretrained_embeddings(self, emb_path, load_all = False):
        #
        if emb_path is None:
            self.randomly_init_embeddings(self.emb_dim)
            return
        #
        if emb_path.endswith('bin'):
            self._load_pretrained_embeddings_bin(emb_path, load_all)
        elif emb_path.endswith('txt'):
            self._load_pretrained_embeddings_txt(emb_path, load_all)
        #
        
    def _load_pretrained_embeddings_bin(self, emb_path, load_all):
        """
        """
        num_tokens = 0
        emb_dim = 0
        trained_embeddings = {}
        #
        fp = open(emb_path,'rb')
        data_byte = fp.read()
        # filesize = fp.tell()
        fp.close()
        #
        posi = 0
        #
        # num_tokens
        str_t = ''
        while True:
            b = data_byte[posi]
            posi += 1
            #
            if b == 32: break
            str_t += chr(b)
        #
        num_tokens = int(str_t)
        #
        # emb_dim
        str_t = ''
        while True:
            b = data_byte[posi]
            posi += 1
            #
            if b == 10: break
            str_t += chr(b)
        #
        emb_dim = int(str_t)
        #
        # emb_mat
        for idx in range(num_tokens):
            #
            posi_start = posi
            count = 0
            while True:
                b = data_byte[posi]
                posi += 1
                #
                if b == 32: break
                count += 1
            #
            if count > 0:
                token = data_byte[posi_start:posi-1].decode('utf-8')                
            else:
                token = ' '
                posi += 1
            #
            emb_list = []
            for d in range(emb_dim):
                data = data_byte[posi:posi+4]
                posi += 4
                elem, = struct.unpack("f", data)
                emb_list.append(elem)
            #
            posi += 1
            #
            trained_embeddings[token] = np.array(emb_list, dtype=np.float32)
            # del emb_list
            #
        #
        del data_byte
        #
        self.emb_dim = emb_dim
        #
        if load_all:
            for token in trained_embeddings.keys():
                self.add(token)
        #
        # initiate embeddings
        self.randomly_init_embeddings(self.emb_dim)
        # self.embeddings = np.random.rand(self.size(), self.emb_dim)
        # for token in [self.pad_token]: self.embeddings[self.get_id(token)] = np.zeros([self.emb_dim])
        # self.embeddings = np.zeros([self.size(), self.emb_dim])
        #
        # load embeddings        
        for token in self.dict_token2id.keys():
            if token in trained_embeddings:
                self.embeddings[self.get_id(token)] = trained_embeddings[token]
        #
        del trained_embeddings
        #
    
    def _load_pretrained_embeddings_txt(self, emb_path, load_all):
        """
        """        
        with open(emb_path, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
        #
        trained_embeddings = {}
        for line in lines:
            contents = line.strip().split()
            #
            if len(contents) < 3: continue  #
            #
            if line.startswith(' '):
                token = ' '
                emb_list = list(map(float, contents[0:]))
                trained_embeddings[token] = np.array(emb_list, dtype=np.float32)
                continue            
            #                
            token = contents[0]
            emb_list = list(map(float, contents[1:]))
            trained_embeddings[token] = np.array(emb_list, dtype=np.float32)
            #
        #
        del lines
        #
        for token in trained_embeddings.keys():
            emb = trained_embeddings[token]
            self.emb_dim = len(emb)
            break
        #
        if load_all:
            for token in trained_embeddings.keys():
                self.add(token)
        #
        # initiate embeddings
        self.randomly_init_embeddings(self.emb_dim)
        # self.embeddings = np.random.rand(self.size(), self.emb_dim)
        # for token in [self.pad_token]: self.embeddings[self.get_id(token)] = np.zeros([self.emb_dim])
        # self.embeddings = np.zeros([self.size(), self.emb_dim])
        #
        # load embeddings        
        for token in self.dict_token2id.keys():
            if token in trained_embeddings:
                self.embeddings[self.get_id(token)] = trained_embeddings[token]
        #
        del trained_embeddings
        #
                
    def save_embeddings_to_file(self, emb_path):
        #
        if emb_path.endswith('bin'):
            self._save_embeddings_to_file_bin(emb_path)
        elif emb_path.endswith('txt'):
            self._save_embeddings_to_file_txt(emb_path)
        #
        
    def _save_embeddings_to_file_bin(self, emb_path):
        """
        """        
        num_words_str = str(self.size()) + ' '
        emb_dim_str = str(self.emb_dim) + '\n'
        with open(emb_path, 'wb') as fp:
            #
            fp.write(num_words_str.encode('utf-8'))  # str to bytes
            fp.write(emb_dim_str.encode('utf-8'))
            #
            for idx in range(self.size()):
                word_and_space = self.dict_id2token[idx] + ' '                
                # fp.write(bytes(word_and_space, encoding = "utf-8"))  # str to bytes
                fp.write(word_and_space.encode('utf-8'))
                
                emb_value = self.embeddings[idx]
                for item in emb_value:
                    write_buf = struct.pack('f', item)   # float to bytes
                    fp.write(write_buf)
                #
                a = struct.pack('B', ord('\n') )   # char to byte
                fp.write(a)
                #
                
    def _save_embeddings_to_file_txt(self, emb_path):
        #
        with open(emb_path, 'w', encoding = 'utf-8') as fp:
            for idd in range(self.size()):
                token = self.dict_id2token[idd]
                emb_str = map(str, self.embeddings[idd])
                line = token + ' ' + ' '.join(emb_str) + '\n'
                fp.write(line)

    def convert_tokens_to_ids(self, tokens):
        """ convert a list of tokens to a list of ids,
            tokens: a list of tokens
        """
        vec = [self.get_id(label) for label in tokens]
        return vec

    def convert_ids_to_tokens(self, ids, stop_id = None):
        """ convert a list of ids to a list of tokens,
            break when stop_id is turning up in ids,
            ids: a list of ids
        """
        tokens = []
        for i in ids:            
            if stop_id is not None and i == stop_id:
                break
            tokens += [self.get_token(i)]
        return tokens

#
if __name__ == '__main__':
    
    import time
    
    vocab = Vocab()
    
    # vocab.add_tokens_from_file('../vocab/vocab_tokens.txt')
    # print('tokens loaded from vocab_tokens.txt')
    # print(vocab.size())
    
    #
    # vocab.randomly_init_embeddings(64)
    # print(vocab.embeddings[vocab.get_id('完成')])
    # print()
    
    #
    # vocab.save_embeddings_to_file('../vocab/vocab_emb_test.txt')
    # print('emb saved to vocab_emb.txt')
    
    s = time.time()
    vocab.load_pretrained_embeddings('../vocab/vocab_emb.txt', load_all=True)
    e = time.time()    
    print('emb loaded from vocab_emb.txt')
    print('time cost: %g' % (e-s))
    
    print("完成" in vocab.dict_token2id)
    print(vocab.get_id('完成'))
    print(vocab.embeddings[vocab.get_id('完成')])
    print()
    
    #
    print(" " in vocab.dict_token2id)
    print(vocab.get_id(' '))
    print(vocab.embeddings[vocab.get_id(' ')])
    print()
    
    #
    vocab.save_embeddings_to_file('../vocab/vocab_emb_test.bin')
    print('emb saved to vocab_emb_test.bin')
    print()
    
    s = time.time()
    vocab.load_pretrained_embeddings('../vocab/vocab_emb_test.bin', load_all=True)
    e = time.time()
    print('emb loaded from vocab_emb_test.bin')
    print('time cost: %g' % (e-s))
    
    print(vocab.embeddings[vocab.get_id('完成')])
    print()
    
    s = time.time()
    vocab.load_pretrained_embeddings('../vocab/vocab_emb_test.bin', load_all=True)
    e = time.time()
    print('emb loaded from vocab_emb_test.bin')
    print('time cost: %g' % (e-s))
    
    print(vocab.embeddings[vocab.get_id('完成')])
    print()
    
    
    # print(vocab.embeddings[vocab.get_id(' ')])
    # print()
    
    # print(vocab.embeddings[vocab.get_id(' ')])
    # print()
    
    #
    print(" " in vocab.dict_token2id)
    print(vocab.get_id(' '))
    print(vocab.embeddings[vocab.get_id(' ')])
    print()
    
    #
    vocab.save_embeddings_to_file('../vocab/vocab_emb_test.txt')
    print('emb saved to vocab_emb_test.txt')
    
    s = time.time()
    vocab.load_pretrained_embeddings('../vocab/vocab_emb_test.txt', load_all=True)
    e = time.time()    
    print('emb loaded from vocab_emb_test.txt')
    print('time cost: %g' % (e-s))
    
    print(vocab.embeddings[vocab.get_id(' ')])
    print()
    
    #
    vocab.save_embeddings_to_file('../vocab/vocab_emb_test.bin')
    print('emb saved to vocab_emb_test.bin')
    print()
    
    s = time.time()
    vocab.load_pretrained_embeddings('../vocab/vocab_emb_test.bin', load_all=True)
    e = time.time()
    print('emb loaded from vocab_emb_test.bin')
    print('time cost: %g' % (e-s))
    
    print(vocab.get_id(' '))
    print(vocab.embeddings[vocab.get_id(' ')])
    print()
    
    s = time.time()
    vocab.load_pretrained_embeddings('../vocab/vocab_emb_test.bin', load_all=True)
    e = time.time()
    print('emb loaded from vocab_emb_test.bin')
    print('time cost: %g' % (e-s))
    
    print(vocab.embeddings[vocab.get_id(' ')])
    print()
    
    print(vocab.embeddings[vocab.get_id('[UNK]')])
    print()
    
    
    # print(vocab.embeddings[vocab.get_id(' ')])
    # print()
    
    # print(vocab.embeddings[vocab.get_id(' ')])
    # print()


