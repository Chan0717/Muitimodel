import numpy as np
import copy
import inspect
import types as python_types
import marshal
import sys
import warnings

from keras import backend as K
# from keras import activations, initializations, regularizers
from keras.engine.topology import Layer, InputSpec


# from keras.layers.wrappers import Wrapper, TimeDistributed
# from keras.layers.core import Dense
# from keras.layers.recurrent import Recurrent, time_distributed_dense


# Build attention pooling layer
class Attention(Layer):
    def __init__(self, op='attsum', activation='tanh', init_stdev=0.01, **kwargs):
        self.supports_masking = True
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):

        self.att_v = self.add_weight(name='att_v',
                                     shape=(input_shape[2], 1),
                                     initializer='uniform',
                                     trainable=True)

        self.att_W = self.add_weight(name='att_W',
                                     shape=(input_shape[2], input_shape[2]),
                                     initializer='uniform',
                                     trainable=True)

        super(Attention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        # u = K.dot(x, self.att_W)
        #
        # u = K.squeeze(u, -1)
        #
        # weights = K.tanh(u) * self.att_v
        # weights = K.exp(weights)
        # weights /= K.cast(K.sum(weights, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        #
        # out = x * weights
        # out = K.sum(out, axis=1)
        # return out
        y = K.dot(x, self.att_W)
        weights = K.dot(K.tanh(y), self.att_v)
        weights = K.squeeze(weights, axis=-1)
        # if not self.activation:
        #     if K.backend() == 'theano':
        #         weights = K.theano.tensor.tensordot(self.att_v, y, axes=[0, 2])
        #     elif K.backend() == 'tensorflow':
        #         weights = K.tensorflow.python.ops.math_ops.tensordot(self.att_v, y, axes=[0, 2])
        # elif self.activation == 'tanh':
        #     if K.backend() == 'theano':
        #         weights = K.theano.tensor.tensordot(self.att_v, K.tanh(y), axes=[0, 2])
        #     elif K.backend() == 'tensorflow':
        #         weights = K.tensorflow.python.ops.math_ops.tensordot(self.att_v, K.tanh(y), axes=[0, 2])
        weights = K.softmax(weights)
        out = x * K.permute_dimensions(K.repeat(weights, x.shape[2]), [0, 2, 1])
        if self.op == 'attsum':
            out = K.sum(out, axis=1)
        elif self.op == 'attmean':
            out = K.sum(out, axis=1) / K.sum(mask, axis=1, keepdims=True)
        return K.cast(out, K.floatx())

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])



