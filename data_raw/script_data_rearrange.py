# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 04:11:16 2018

@author: limingfan
"""

import xlrd

import random


# task-related
def load_from_file_raw(file_raw):
        
    work_book = xlrd.open_workbook(file_raw)
    data_sheet = work_book.sheets()[0]
    text_raw = data_sheet.col_values(0)
    return text_raw
        
#
if __name__ == '__main__':
    
    file_pos = './pos.xls'
    file_neg = './neg.xls'
    
    data_pos = load_from_file_raw(file_pos)
    data_neg = load_from_file_raw(file_neg)
    
    label_1 = [1] * len(data_pos)
    label_0 = [0] * len(data_neg)
    
    data_all = data_pos + data_neg
    label_all = label_1 + label_0
    
    examples = list(zip(data_all, label_all)) 
    random.shuffle(examples)
    
    file_txt = './data_raw.txt'
    
    with open(file_txt, 'w', encoding = 'utf-8') as fp:     
        for item in examples:
            fp.write(str(item[1]) + '<label_text_delimiter>' + item[0].strip() + '\n')            


