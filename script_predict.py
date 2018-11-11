# -*- coding: utf-8 -*-

import os

from data_set import Dataset

from model_settings import ModelSettings
from model_wrapper import ModelWrapper


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#
model_tag = 'cnn'
#

if model_tag == 'cnn':
    from model_graph_cnn import build_model_graph
elif model_tag == 'csm':
    from model_graph_csm import build_model_graph
elif model_tag == 'rnn':
    from model_graph_rnn import build_model_graph
elif model_tag == 'rnf':
    from model_graph_rnf import build_model_graph
    
#
# data
dataset = Dataset()
dataset.load_vocab_tokens_and_emb()
#

#
settings = ModelSettings()
settings.vocab = dataset.vocab
settings.model_tag = model_tag
settings.model_graph_builder = build_model_graph
settings.is_train = False
settings.check_settings()
#
model = ModelWrapper(settings)
model.prepare_for_prediction()
#


text_raw = ["这本书不错" ]

"""
work_book = xlrd.open_workbook(file_raw)
data_sheet = work_book.sheets()[0]
text_raw = data_sheet.col_values(0)
"""

#
preds_list = []
logits_list = []
#
for item in text_raw:
    out = model.predict([ (item, 0) ])
    print(out)
    
    logits = out[0]
    pred = list(logits[0]).index(max(logits[0]))
    #
    logits_list.append(logits[0])
    preds_list.append(pred)
#
list_pair = zip(preds_list, text_raw)
#
for item in list_pair:
    print(item)


"""
workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('My Worksheet')

for idx, item in enumerate(items_err):
    label = 'a' if item[1] == 1 else 'b'
    worksheet.write(idx, 0, ''.join(dataset.vocab.recover_from_ids(item[0]) ) )
    worksheet.write(idx, 2, label)

filepath = 'ab.xls'
if os.path.exists(filepath):
    os.remove(filepath)
workbook.save(filepath)

"""





