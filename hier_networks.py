# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2017-01-10 11:40:53
# @Last Modified by:   feidong1991
# @Last Modified time: 2017-06-18 16:08:08

from keras.models import *
from keras.optimizers import *
from keras.layers.core import *
from keras.layers import Input, Embedding, LSTM, GRU, Dense, Reshape
from keras.layers import TimeDistributed
from keras.layers.merge import concatenate, multiply

from keras.layers.convolutional import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D, GlobalMaxPooling2D
from keras.regularizers import l2

from softattention import Attention
from coattention import CoAttention
from zeromasking import ZeroMaskedEntries
from utils import get_logger
import time


logger = get_logger("Build model")

"""
Hierarchical networks, the following function contains several models:
(1)build_hcnn_model: hierarchical CNN model
(2)build_hrcnn_model: hierarchical Recurrent CNN model, LSTM stack over CNN,
 it supports two pooling methods
    (a): Mean-over-time pooling
    (b): attention pooling
"""
def build_model(opts, vocab_size=0, maxnum=50, maxlen=50, embedd_dim=50, embedding_weights=None, verbose=False, init_mean_value=None):
    N = maxnum
    L = maxlen

    p = Input(shape=(4, 2048), dtype='float32', name='p')
    # img_vector = Dense(name='img_vector', units=128)(p)

    word_input = Input(shape=(N * L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N * L, weights=embedding_weights,
                  mask_zero=True, trainable=False, name='x')(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

    cnn_e = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, border_mode='valid'), name='cnn_e')(resh_W)

    att_cnn_e = TimeDistributed(Attention(), name='att_cnn_e')(cnn_e)

    lstm_e = LSTM(opts.lstm_units, return_sequences=True, name='lstm_e')(att_cnn_e)

    G = CoAttention(name='essay')([lstm_e, p])
    avg = GlobalAveragePooling1D()(G)
    final_vec_drop = Dropout(rate=0.5, name='final_vec_drop')(avg)
    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(final_vec_drop)
    else:
        y = Dense(units=1, activation='sigmoid', name='output')(final_vec_drop)

    model = Model(input=[word_input, p], output=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value
                                     )

    if verbose:
        model.summary()

    start_time = time.time()
    model.compile(loss='mse', optimizer='adam')
    total_time = time.time() - start_time
    logger.info("Model compiled in %.4f s" % total_time)

    return model

def build_model_with_all(opts, vocab_size=0, maxnum=50, maxlen=50, maxnum_d=50, embedd_dim=50, embedding_weights=None, verbose=False, init_mean_value=None):

    p = Input(shape=(4, 2048), dtype='float32', name='p')
    img_vector = Attention(name='img_vector')(p)

    N = maxnum
    L = maxlen
    N_d = maxnum_d
    word_input = Input(shape=(N * L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N * L, weights=embedding_weights,
                  mask_zero=True, name='x')(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

    word_input_d = Input(shape=(N_d * L,), dtype='int32', name='word_input_d')
    x_d = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N_d * L, weights=embedding_weights,
                    mask_zero=True, name='x_d')(word_input_d)
    x_d_maskedout = ZeroMaskedEntries(name='x_d_maskedout')(x_d)
    drop_x_d = Dropout(opts.dropout, name='drop_x_d')(x_d_maskedout)

    resh_W_d = Reshape((N_d, L, embedd_dim), name='resh_W_d')(drop_x_d)

    cnn_e = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, border_mode='valid'), name='cnn_e')(resh_W)

    cnn_d = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, border_mode='valid'), name='cnn_d')(resh_W_d)

    att_cnn_e = TimeDistributed(Attention(), name='att_cnn_e')(cnn_e)
    att_cnn_d = TimeDistributed(Attention(), name='att_cnn_d')(cnn_d)

    lstm_e = LSTM(opts.lstm_units, return_sequences=True, name='lstm_e')(att_cnn_e)
    lstm_d = LSTM(opts.lstm_units, return_sequences=True, name='lstm_d')(att_cnn_d)

    essay = Attention(name='essay')(lstm_e)
    prompt = Attention(name='prompt')(lstm_d)

    final_vec = concatenate([essay, prompt, img_vector], name='final_vec')
    final_vec_drop = Dropout(rate=0.5, name='final_vec_drop')(final_vec)
    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(final_vec_drop)
    else:
        y = Dense(units=1, activation='sigmoid', name='output')(final_vec_drop)

    model = Model(input=[word_input, word_input_d, p], output=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    start_time = time.time()
    model.compile(loss='mse', optimizer='adam')
    total_time = time.time() - start_time
    logger.info("Model compiled in %.4f s" % total_time)

    return model

def build_model_with_topic(opts, vocab_size=0, maxnum=50, maxlen=50, maxnum_d=50, embedd_dim=50, embedding_weights=None, verbose=False, init_mean_value=None):

    N = maxnum
    L = maxlen
    N_d = maxnum_d
    word_input = Input(shape=(N * L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N * L, weights=embedding_weights,
                  mask_zero=True, name='x')(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

    word_input_d = Input(shape=(N_d * L,), dtype='int32', name='word_input_d')
    x_d = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N_d * L, weights=embedding_weights,
                    mask_zero=True, name='x_d')(word_input_d)
    x_d_maskedout = ZeroMaskedEntries(name='x_d_maskedout')(x_d)
    drop_x_d = Dropout(opts.dropout, name='drop_x_d')(x_d_maskedout)

    # resh_W_d = Reshape((N_d, L, embedd_dim), name='resh_W_d')(drop_x_d)

    cnn_e = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, border_mode='valid'), name='cnn_e')(resh_W)

    # cnn_d = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, border_mode='valid'), name='cnn_d')(resh_W_d)

    att_cnn_e = TimeDistributed(Attention(), name='att_cnn_e')(cnn_e)
    # att_cnn_d = TimeDistributed(Attention(), name='att_cnn_d')(cnn_d)

    att_cnn_e = Dropout(rate=0.5)(att_cnn_e)
    # att_cnn_d = Dropout(rate=0.5)(att_cnn_d)

    lstm_e = LSTM(opts.lstm_units, return_sequences=True, name='lstm_e')(att_cnn_e)
    lstm_d = LSTM(opts.lstm_units, return_sequences=True, name='lstm_d')(drop_x_d)

    # essay = Attention(name='essay')(lstm_e)
    # prompt = Attention(name='prompt')(lstm_d)

    # final_vec = concatenate([essay, prompt], name='final_vec')
    G = CoAttention(name='G')([lstm_e, lstm_d])
    avg = GlobalAveragePooling1D()(G)
    final_vec_drop = Dropout(rate=0.5, name='final_vec_drop')(avg)
    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(final_vec_drop)
    else:
        y = Dense(units=1, activation='sigmoid', name='output')(final_vec_drop)

    model = Model(input=[word_input, word_input_d], output=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    start_time = time.time()
    model.compile(loss='mse', optimizer='adam')
    total_time = time.time() - start_time
    logger.info("Model compiled in %.4f s" % total_time)

    return model

def cnn():
    model = Sequential()

    model.add(Convolution2D(input_shape=(256, 256, 3), filters=8, kernel_size=3, strides=1, padding='valid', activation='relu',
                                  data_format='channels_last', name='cnn2d'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid',
                                data_format='channels_last', name='max_pooling1'))
    model.add(Convolution2D(filters=16, kernel_size=3, strides=1, padding='valid', activation='relu',
                                  data_format='channels_last', name='cnn2d_1'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid',
                                data_format='channels_last', name='max_pooling2'))
    model.add(Convolution2D(filters=16, kernel_size=3, strides=1, padding='valid', activation='relu',
                                  data_format='channels_last', name='cnn2d_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid',
                                data_format='channels_last', name='max_pooling3'))
    model.add(Convolution2D(filters=32, kernel_size=3, strides=1, padding='valid', activation='relu',
                            data_format='channels_last', name='cnn2d_3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid',
                           data_format='channels_last', name='max_pooling4'))
    model.add(Convolution2D(filters=100, kernel_size=3, strides=1, padding='valid', activation='relu',
                            data_format='channels_last', name='cnn2d_4'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid',
                           data_format='channels_last', name='max_pooling5'))
    return model

def build_model_fusion(opts, vocab_size=0, maxnum=50, maxlen=50, embedd_dim=50, embedding_weights=None, verbose=False, init_mean_value=None):

    # p_input1 = Input(shape=(256, 256, 3), dtype='float32', name='p_input1')
    # p_input2 = Input(shape=(256, 256, 3), dtype='float32', name='p_input2')
    # p_input3 = Input(shape=(256, 256, 3), dtype='float32', name='p_input3')
    # p_input4 = Input(shape=(256, 256, 3), dtype='float32', name='p_input4')
    p = Input(shape=(256, 256, 3), dtype='float32', name='p')
    cnn_model = cnn()
    img = cnn_model(p)
    img = Reshape([6*6, 100])(img)
    # img1 = cnn_model(p_input1)
    # img2 = cnn_model(p_input2)
    # img3 = cnn_model(p_input3)
    # img4 = cnn_model(p_input4)
    # img1 = GlobalMaxPooling2D()(img1)
    # img2 = GlobalMaxPooling2D()(img2)
    # img3 = GlobalMaxPooling2D()(img3)
    # img4 = GlobalMaxPooling2D()(img4)

    # img = concatenate([img1, img2, img3, img4], axis=1)
    # img = Reshape((4, 100))(img)

    N = maxnum
    L = maxlen

    word_input = Input(shape=(N * L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N * L, weights=embedding_weights,
                  mask_zero=True, name='x')(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

    cnn_e = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, border_mode='valid', activation='tanh'), name='cnn_e')(resh_W)
    cnn_e = Dropout(rate=0.5)(cnn_e)
    att_cnn_e = TimeDistributed(Attention(), name='att_cnn_e')(cnn_e)
    att_cnn_e = Dropout(rate=0.5)(att_cnn_e)
    lstm_e = LSTM(opts.lstm_units, return_sequences=True, name='lstm_e')(att_cnn_e)
    lstm_e = Dropout(rate=0.5)(lstm_e)
    G = CoAttention(name='essay')([lstm_e, img])
    avg = GlobalAveragePooling1D()(G)
    final_vec_drop = Dropout(rate=0.5, name='final_vec_drop')(avg)

    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(final_vec_drop)
    else:
        y = Dense(units=1, activation='sigmoid', name='output')(final_vec_drop)

    # model = Model(input=[word_input, p_input1, p_input2, p_input3, p_input4], output=y)
    model = Model(input=[word_input, p], output=y)
    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    start_time = time.time()
    model.compile(loss='mse', optimizer='adam')
    total_time = time.time() - start_time
    logger.info("Model compiled in %.4f s" % total_time)

    return model

