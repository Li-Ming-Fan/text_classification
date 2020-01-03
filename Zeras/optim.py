

import re

import tensorflow as tf
from tensorflow.python.ops import state_ops

#
def linear_warmup_and_exp_decayed_lr(settings, global_step):
    """ settings.learning_rate_base
        settings.warmup_steps
        settings.decay_steps
        settings.decay_rate
        settings.staircase    
    """
    learning_rate = tf.constant(value = settings.learning_rate_base,
                                shape = [], dtype = tf.float32)
        
    if settings.warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(settings.warmup_steps, dtype=tf.int32)
        
        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
        
        step_surplus = global_steps_int - warmup_steps_int
        learning_rate = tf.train.exponential_decay(learning_rate,
                                                   step_surplus,
                                                   settings.decay_steps,
                                                   settings.decay_rate,
                                                   settings.staircase)
        
        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = settings.learning_rate_base * warmup_percent_done
        
        learning_rate = tf.cond(global_steps_int < warmup_steps_int,
                                lambda: warmup_learning_rate,
                                lambda: learning_rate)
    #
    else:
        learning_rate = tf.train.exponential_decay(learning_rate,
                                                   global_step,
                                                   settings.decay_steps,
                                                   settings.decay_rate,
                                                   settings.staircase)
    #
    return learning_rate
    #

def linear_warmup_and_polynomial_decayed_lr(settings, global_step):
    """ settings.learning_rate_base
        settings.warmup_steps
        settings.decay_steps
        settings.learning_rate_minimum
        settings.lr_power
        settings.lr_cycle
    """
    learning_rate = tf.constant(value = settings.learning_rate_base,
                                shape = [], dtype = tf.float32)
        
    if settings.warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(settings.warmup_steps, dtype=tf.int32)
        
        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
        
        step_surplus = global_steps_int - warmup_steps_int
        learning_rate = tf.train.polynomial_decay(learning_rate,
                                                  step_surplus,
                                                  settings.decay_steps,
                                                  end_learning_rate=settings.learning_rate_minimum,
                                                  power=settings.lr_power,
                                                  cycle=settings.lr_cycle)
        
        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = settings.learning_rate_base * warmup_percent_done
        
        learning_rate = tf.cond(global_steps_int < warmup_steps_int,
                                lambda: warmup_learning_rate,
                                lambda: learning_rate)
    #
    else:
        learning_rate = tf.train.polynomial_decay(learning_rate,
                                                  global_step,
                                                  settings.decay_steps,
                                                  end_learning_rate=settings.learning_rate_minimum,
                                                  power=settings.lr_power,
                                                  cycle=settings.lr_cycle)
    #
    return learning_rate
    #
    
def adam_optimizer(settings, learning_rate_tensor_or_value):
    """ 
        customized_optimizer = get_adam_optimizer
        self._opt = self.customized_optimizer(self.settings, self.learning_rate_tensor)
        
        grad_and_vars = self._opt.compute_gradients(self.loss_train_tensor)
        self.train_op = self._opt.apply_gradients(grad_and_vars, global_step = self.global_step)
    """
    opt = tf.train.AdamOptimizer(learning_rate_tensor_or_value, beta1 = settings.momentum)
    return opt
    #

class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """ A basic Adam optimizer that includes "correct" L2 weight decay.
    """
    def __init__(self, learning_rate, weight_decay_rate=0.0,
                       beta_1=0.9, beta_2=0.999, epsilon=1e-6,
                       exclusions_from_weight_decay=None,
                       name="AdamWeightDecayOptimizer"):
        """
        """
        super(AdamWeightDecayOptimizer, self).__init__(False, name)        
        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.exclusions_from_weight_decay = exclusions_from_weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon        

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        """
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name = param_name + "/adam_m",
                shape = param.shape.as_list(),
                dtype = tf.float32,
                trainable = False,
                initializer = tf.zeros_initializer())
            v = tf.get_variable(
                name = param_name + "/adam_v",
                shape = param.shape.as_list(),
                dtype = tf.float32,
                trainable = False,
                initializer = tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad) )
            next_v = (
                tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                            tf.square(grad)) )

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            next_param = param - self.learning_rate * update
            assignments.extend([ param.assign(next_param),
                                 m.assign(next_m), 
                                 v.assign(next_v) ])
        #
        assignments.append(global_step.assign(global_step + 1))
        #
        # global_step_add = state_ops.assign_add(global_step, 1, name=name)
        # assignments.append(global_step_add)
        #
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """
        """
        if not self.weight_decay_rate:
            return False
        if self.exclusions_from_weight_decay:
            for r in self.exclusions_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """ Get the variable name from the tensor name.
        """
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

def adam_wd_optimizer(settings, learning_rate_tensor_or_value):
    """
    """
    opt = AdamWeightDecayOptimizer(learning_rate_tensor_or_value,
                        weight_decay_rate = settings.reg_lambda,
                        exclusions_from_weight_decay = settings.reg_exclusions,
                        beta_1 = settings.beta_1, beta_2 = settings.beta_2,
                        epsilon = 1e-6)
    #
    return opt
    #


  