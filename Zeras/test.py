#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

def test_tensorflow_gpu():
    """
    """
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print(sess.run(c))

#
if __name__ == "__main__":
    #
    test_tensorflow_gpu()
    #