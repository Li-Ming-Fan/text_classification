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
