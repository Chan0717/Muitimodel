import tensorflow as tf
from tensorflow.python.ops import math_ops
import numpy as np
import time
from utils import get_batch_index


class RAM(object):

    def __init__(self, config, sess):
        self.embedding_dim = config.embedding_dim
        self.batch_size = config.batch_size
        self.n_epoch = config.n_epoch
        self.n_hidden = config.n_hidden
        self.n_class = config.n_class
        self.n_hop = config.n_hop
        self.learning_rate = config.learning_rate
        self.l2_reg = config.l2_reg
        self.dropout = config.dropout

        self.word2id = config.word2id
        self.max_sentence_len = config.max_sentence_len
        self.max_aspect_len = config.max_aspect_len
        self.word2vec = config.word2vec
        self.sess = sess

        self.timestamp = str(int(time.time()))

    def build_model(self):
        with tf.name_scope('inputs'):
            self.sentences = tf.placeholder(tf.int32, [None, self.max_sentence_len])
            self.aspects = tf.placeholder(tf.int32, [None, self.max_aspect_len])
            self.sentence_lens = tf.placeholder(tf.int32, None)
            self.sentence_locs = tf.placeholder(tf.float32, [None, self.max_sentence_len])
            self.labels = tf.placeholder(tf.int32, [None, self.n_class])
            self.dropout_keep_prob = tf.placeholder(tf.float32)

            inputs = tf.nn.embedding_lookup(self.word2vec, self.sentences)
            inputs = tf.cast(inputs, tf.float32)
            inputs = tf.nn.dropout(inputs, keep_prob=self.dropout_keep_prob)
            aspect_inputs = tf.nn.embedding_lookup(self.word2vec, self.aspects)
            aspect_inputs = tf.cast(aspect_inputs, tf.float32)
            aspect_inputs = tf.reduce_mean(aspect_inputs, 1)

        with tf.name_scope('weights'):
            weights = {

            }
        with tf.name_scope('biases'):
            biases = {

            }
        with tf.name_scope('updates'):
            updates = {

            }
        with tf.name_scope('dynamic_rnn'):
            lstm_cell_fw = tf.contrib.rnn.LSTMCell(
                self.n_hidden,
                initializer=tf.orthogonal_initializer(),
            )
            lstm_cell_bw = tf.contrib.rnn.LSTMCell(
                self.n_hidden,
                initializer=tf.orthogonal_initializer(),
            )

            outputs, state, _ = tf.nn.static_bidirectional_rnn(
                lstm_cell_fw,
                lstm_cell_bw,
                tf.unstack(tf.transpose(inputs, perm=[1, 0, 2])),
                sequence_length=self.sentence_lens,
                dtype=tf.float32,
                scope='BiLSTM'
            )
            outputs = tf.reshape(tf.concat(outputs, 1), [-1, self.max_sentence_len, self.n_hidden * 2])
            outputs = tf.reshape(tf.concat(outputs, 1), [-1, self.max_sentence_len, self.n_hidden * 2])
            batch_size = tf.shape(outputs)[0]

            outputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            outputs_iter = outputs_iter.unstack(outputs)
            sentence_locs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            sentence_locs_iter = sentence_locs_iter.unstack(self.sentence_locs)
            sentence_lens_iter = tf.TensorArray(tf.int32, 1, dynamic_size=True, infer_shape=False)
            sentence_lens_iter = sentence_lens_iter.unstack(self.sentence_lens)
            memory = tf.TensorArray(size=batch_size, dtype=tf.float32)
            def body(i, memory):
                a = outputs_iter.read(i)
                b = sentence_locs_iter.read(i)
                c = sentence_lens_iter.read(i)
                weight = 1 - b
                memory = memory.write(i, tf.concat([tf.multiply(a, tf.tile(tf.expand_dims(weight, -1), [1, self.n_hidden * 2])), tf.reshape(b, [-1, 1])], 1))
                return (i + 1, memory)

            def condition(i, memory):
                return i < batch_size

            _, memory_final = tf.while_loop(cond=condition, body=body, loop_vars=(0, memory))