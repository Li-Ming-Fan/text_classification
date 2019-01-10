# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 13:00:01 2018

@author: limingfan

"""

list_a = [0, 1, 3, 2]

print(list_a.index(max(list_a)))


#
import tensorflow as tf

tf.reset_default_graph()


a = tf.get_variable('a', shape = (2,4,5),  initializer = tf.random_normal_initializer() )
b = tf.get_variable('b', shape = (2,3,5), initializer = tf.random_normal_initializer() )

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(a))
print(sess.run(b))

c = tf.matmul(a, tf.transpose(b, [0,2,1]))
print(c)


print()
print(sess.run(tf.cast([0, 1.2, -1.0], dtype = tf.bool)) )


import pkuseg

seg = pkuseg.pkuseg()                                  # 以默认配置加载模型
text = seg.cut('我爱北京天安门')                        # 进行分词
print(text)

lexicon = ['北京大学', '北京天安门']                     # 希望分词时用户词典中的词固定不分开
seg = pkuseg.pkuseg(user_dict=lexicon)                  # 加载模型，给定用户词典
text = seg.cut('我爱北京天安门')                         # 进行分词
print(text)

