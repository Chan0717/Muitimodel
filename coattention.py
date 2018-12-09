from keras import backend as K
# from keras import activations, initializations, regularizers
from keras.engine.topology import Layer, InputSpec


# from keras.layers.wrappers import Wrapper, TimeDistributed
# from keras.layers.core import Dense
# from keras.layers.recurrent import Recurrent, time_distributed_dense


# Build attention pooling layer
class CoAttention(Layer):
    def __init__(self, op='attsum', activation='tanh', init_stdev=0.01, **kwargs):
        self.supports_masking = True
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        super(CoAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(name='b',
                                     shape=(1, ),
                                     initializer='uniform',
                                     trainable=True)

        self.W1 = self.add_weight(name='W1',
                                     shape=(input_shape[0][-1], 1),
                                     initializer='uniform',
                                     trainable=True)
        self.W2 = self.add_weight(name='W2',
                                     shape=(input_shape[0][-1], 1),
                                     initializer='uniform',
                                     trainable=True)
        self.W3 = self.add_weight(name='W3',
                                     shape=(input_shape[0][-1], 1),
                                     initializer='uniform',
                                     trainable=True)
        super(CoAttention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        text = x[0]
        img = x[1]
        v1 = K.dot(text, self.W1)
        v2 = K.dot(img, self.W2)
        text = K.expand_dims(text, axis=2)
        img = K.expand_dims(img, axis=1)
        v3 = K.dot(text * img, self.W3)
        v = v1 + K.permute_dimensions(v2, (0, 2, 1)) + K.squeeze(v3, axis=-1) + self.bias
        # v = text * img
        # v = K.sum(v, axis=-1)
        A_img = K.softmax(v, axis=2)
        A_text = K.softmax(K.max(v, axis=2), axis=1)
        text = K.squeeze(text, axis=2)
        img = K.squeeze(img, axis=1)
        text_re = K.batch_dot(text, A_text, (1, 1))
        text_re = K.expand_dims(text_re, axis=1)
        img_re = K.batch_dot(img, A_img, (1, 2))
        img_re = K.permute_dimensions(img_re, (0, 2, 1))
        G = K.concatenate([text, img_re, text * img_re, text * text_re])
        return G

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2] * 4)



