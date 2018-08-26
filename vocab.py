# -*- coding:utf8 -*-

import numpy as np


class Vocab(object):
    
    def __init__(self, lower=False):
        self.id2token = {}
        self.token2id = {}
        self.token_cnt = {}
        self.lower = lower

        self.emb_dim = None
        self.embeddings = None

        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        
        self.initial_tokens = []
        self.initial_tokens.extend([self.pad_token, self.unk_token, self.sos_token, self.eos_token])
        for token in self.initial_tokens:
            self.add(token, cnt = 10000)

    def size(self):
        return len(self.id2token)

    def get_id(self, token):
        token = token.lower() if self.lower else token
        try:
            return self.token2id[token]
        except KeyError:
            return self.token2id[self.unk_token]

    def get_token(self, idx):
        try:
            return self.id2token[idx]
        except KeyError:
            return self.unk_token

    def add(self, token, cnt=1):
        #
        token = token.lower() if self.lower else token
        if token in self.token2id:
            idx = self.token2id[token]
        else:
            idx = len(self.id2token)
            self.id2token[idx] = token
            self.token2id[token] = idx
        if cnt > 0:
            if token in self.token_cnt:
                self.token_cnt[token] += cnt
            else:
                self.token_cnt[token] = cnt
        return idx

    def filter_tokens_by_cnt(self, min_cnt):
        #
        #filtered_tokens = [token for token in self.token2id if self.token_cnt[token] >= min_cnt]
        filtered_tokens = [self.id2token[idd] for idd in range(self.size()) \
                                  if self.token_cnt[self.id2token[idd]] >= min_cnt]
        # rebuild the token x id map
        self.token2id = {}
        self.id2token = {}
        for token in self.initial_tokens:
            self.add(token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)
            
    def load_tokens_from_corpus(self, corp):
        """ load tokens from corpus (list)
            with each item in the list as a list of tokens 
        """
        for item in corp:
            for word in item:
                self.add(word)
                
    def load_tokens_from_file(self, file_path):
        """ load tokens from file
            with one token in one line
        """
        for line in open(file_path, 'r', encoding='utf-8'):
            token = line.rstrip('\n')
            self.add(token)
            
    def save_tokens_to_file(self, file_path):
        """ save tokens to file
            with one token in one line
        """
        with open(file_path, 'w', encoding='utf-8') as fp:
            #for token in self.token2id.keys():
            for idd in range(self.size()):            
                fp.write(self.id2token[idd] + '\n')

    def randomly_init_embeddings(self, emb_dim):
        #
        self.emb_dim = emb_dim
        self.embeddings = np.random.rand(self.size(), emb_dim)
        #for token in [self.pad_token, self.unk_token]:
        for token in [self.pad_token]:
            self.embeddings[self.get_id(token)] = np.zeros([self.emb_dim])

    def load_pretrained_embeddings(self, embedding_path):
        #
        trained_embeddings = {}
        with open(embedding_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                contents = line.strip().split()
                token = contents[0] #.decode('utf8')
                #
                if token not in self.token2id: continue  # only existing tokens
                #
                try:
                    trained_embeddings[token] = list(map(float, contents[1:]))
                except BaseException:
                    print(contents)
                    continue
        #
        trained_tokens = list(trained_embeddings.keys() )
        #if self.emb_dim is None:
        self.emb_dim = len(trained_embeddings[trained_tokens[0]])
        #
        # rebuild the token x id map
        #
        # load embeddings
        self.embeddings = np.zeros([self.size(), self.emb_dim])
        for token in self.token2id.keys():
            if token in trained_embeddings:
                self.embeddings[self.get_id(token)] = trained_embeddings[token]
                
    def save_embeddings_to_file(self, embedding_path):
        """ save embeddings
        """
        with open(embedding_path, 'w', encoding = 'utf-8') as fp:
            for idd in range(self.size()):
                token = self.id2token[idd]
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
            tokens += [self.get_token(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens
        
        
        