#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 04:11:16 2018

@author: limingfan
"""

import pickle

import jieba

import xlrd


# task-related
def load_from_file_raw(file_raw):
        
    work_book = xlrd.open_workbook(file_raw)
    data_sheet = work_book.sheets()[0]
    text_raw = data_sheet.col_values(0)
    return text_raw
    
    """
    text_raw = []
    with open(file_raw, 'r', encoding = 'utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            if line.strip() != '':
                text_raw.append(line)
    #
    return text_raw
    
    #
    work_book = xlrd.open_workbook(file_raw)
    data_sheet = work_book.sheets()[0]
    queries = data_sheet.col_values(0)
    labels = data_sheet.col_values(2)
    return queries, labels
    """

#      
def clean_and_seg_single_text(text):        
    text = text.strip()
    #
    seg_list = jieba.cut(text, cut_all = False)
    tokens = list(seg_list)
    #
    #print(text)
    #tokens = list(text)   # cut chars
    #
    return tokens

#
def clean_and_seg_list_raw(data_raw):        
    data_seg = []
    for item in data_raw:
        text_tokens = clean_and_seg_single_text(item)
        data_seg.append(text_tokens)
    return data_seg
#
def convert_data_seg_to_ids(vocab, data_seg):
    data_converted = []
    for item in data_seg:
        ids = vocab.convert_tokens_to_ids(item)
        data_converted.append(ids)
    return data_converted


# task-independent
def save_data_to_pkl(data, file_path):
    with open(file_path, 'wb') as fp:
        pickle.dump(data, fp)
        
def load_data_from_pkl(file_path):
    with open(file_path, 'rb') as fp:
        return pickle.load(fp)
    
    
#
if __name__ == '__main__':
    
    pass

