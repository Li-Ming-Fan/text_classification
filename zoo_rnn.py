# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 17:13:09 2018

@author: limingfan
"""

import tensorflow as tf

from zoo_layers import dropout

#
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
# from tensorflow.python.ops import variable_scope as vs

# higher version
from tensorflow.python.ops.rnn_cell import RNNCell  # tf.contrib.rnn.RNNCell
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple


_WEIGHTS_VARIABLE_NAME = "kernel"
_BIAS_VARIABLE_NAME = "bias"

#
class MyGRUCell(RNNCell):
    def __init__(self, num_units, keep_prob=1.0,
                 activation=None, reuse=None,
                 kernel_initializer=None, bias_initializer=None):
        super(MyGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        
        self._gate_linear = None
        self._candidate_linear = None        
        self._keep_prob = keep_prob

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        '''

        d_state = dropout(state, self._keep_prob, mode="embedding")

        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):
                self._gate_linear = _Linear([inputs, d_state], 2 * self._num_units, True,
                                            bias_initializer=bias_ones,
                                            kernel_initializer=self._kernel_initializer)
                
        value = math_ops.sigmoid(self._gate_linear([inputs, d_state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        
        r_state = r * d_state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear([inputs, r_state], self._num_units, True,
                                                 bias_initializer=self._bias_initializer,
                                                 kernel_initializer=self._kernel_initializer)
                
        c = self._activation(self._candidate_linear([inputs, r_state]))
        new_h = u * state + (1 - u) * c
        '''
        new_h = state

        return new_h, new_h
    
    
class MyLSTMCell(RNNCell):
    """ Basic LSTM recurrent network cell.
    """
    def __init__(self, num_units, keep_prob=1.0,
                 initializer=None, activation=None, forget_bias=1.0, 
                 state_is_tuple=True, reuse=None, name=None):
        """ Initialize the basic LSTM cell.
        """
        super(MyLSTMCell, self).__init__(_reuse=reuse, name=name)
        if not state_is_tuple:
            print("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
            
        # Inputs must be 2-dimensional.
        self._num_units = num_units
        
        self._weight_initializer = initializer or init_ops.variance_scaling_initializer()
        self._activation = activation or math_ops.tanh
        
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        
        # drop
        self._keep_prob = keep_prob
        self._ones = tf.Variable(tf.ones(shape=(num_units, num_units), dtype=tf.float32),
                                 trainable = False, name="ones_mask")
        self._drop_mask = self._ones
        self._noise_shape = [self._num_units, 1]
                
    def renew_drop_mask(self):        
        self._drop_mask = tf.nn.dropout(self._ones, self._keep_prob,
                                        noise_shape = self._noise_shape)
    #
        
    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)
        
    @property
    def output_size(self):
        return self._num_units
        
    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)
            
        input_depth = inputs_shape[1].value
        h_depth = self._num_units
        self._kernel = self.add_variable(_WEIGHTS_VARIABLE_NAME,
                                         shape = [input_depth + h_depth, 4 * self._num_units],
                                         initializer = self._weight_initializer)
        self._bias = self.add_variable(_BIAS_VARIABLE_NAME,
                                       shape = [4 * self._num_units],
                                       initializer = init_ops.zeros_initializer(dtype=self.dtype))
        
        self.built = True
    
    def call(self, inputs, state):
        """ Long short-term memory cell (LSTM).
        """
        sigmoid = math_ops.sigmoid
        # one = constant_op.constant(1, dtype=dtypes.int32)
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1) #one)
            
        # Note that using `add` and `multiply` instead of `+` and `*` gives a
        # performance improvement. So using those at the cost of readability.
        add = math_ops.add
        multiply = math_ops.multiply
        
        if self._keep_prob < 1.0: h = multiply(h, self._drop_mask)
        
        gate_inputs = math_ops.matmul(array_ops.concat([inputs, h], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
        
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=1) #one)
        
        forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
        
        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                    multiply(sigmoid(i), self._activation(j)))
        new_h = multiply(self._activation(new_c), sigmoid(o))
        
        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state

#
def rnn_layer(input_sequence, sequence_length, rnn_size,
              keep_prob = 1.0, concat = True, scope = 'bi-lstm'):
    '''build bidirectional lstm layer'''
    #
    # time_major = False
    #
    if keep_prob < 1.0:
        input_sequence = dropout(input_sequence, keep_prob)
    #
    # to time_major from batch_major
    input_sequence = tf.transpose(input_sequence, [1,0,2])
    #
    weight_initializer = tf.truncated_normal_initializer(stddev = 0.01)
    #
    #cell_fw = tf.contrib.rnn.LSTMCell(rnn_size, initializer = weight_initializer)
    #cell_bw = tf.contrib.rnn.LSTMCell(rnn_size, initializer = weight_initializer)
    #
    #cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=dropout_rate)
    #cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=dropout_rate)
    #
    cell_fw = MyLSTMCell(rnn_size, keep_prob, initializer = weight_initializer)
    cell_bw = MyLSTMCell(rnn_size, keep_prob, initializer = weight_initializer)
    #
    cell_fw.renew_drop_mask()
    cell_bw.renew_drop_mask()
    #
    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_sequence,
                                                    sequence_length = sequence_length,
                                                    time_major = True,
                                                    dtype = tf.float32,
                                                    scope = scope)
    #
    if concat:
        rnn_output = tf.concat(rnn_output, 2, name = 'output')
    else:
        rnn_output = tf.multiply(tf.add(rnn_output[0], rnn_output[1]), 0.5, name = 'output')
    #
    # to batch_major from time_major
    rnn_output = tf.transpose(rnn_output, [1,0,2])
    #
    return rnn_output
    #

       