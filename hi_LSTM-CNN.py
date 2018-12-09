# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2017-01-10 11:57:22
# @Last Modified by:   feidong1991
# @Last Modified time: 2017-06-18 16:44:56


import os
import sys
import argparse
import random
import time
import numpy as np
from utils import *
from hier_networks import *
from process_picture import *
import data_prepare
from evaluator import Evaluator

logger = get_logger("Train sentence sequences Recurrent Convolutional model (LSTM stack over CNN)")
np.random.seed(100)

is_training = True
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def main(fold, p_id):
    parser = argparse.ArgumentParser(description="sentence Hi_CNN model")
    parser.add_argument('--embedding', type=str, default='glove', help='Word embedding type, word2vec, senna or glove')
    parser.add_argument('--embedding_dict', type=str, default='glove.6B.100d.txt', help='Pretrained embedding path')
    parser.add_argument('--embedding_dim', type=int, default=100, help='Only useful when embedding is randomly initialised')

    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of texts in each batch')
    parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000, help="Vocab size (default=4000)")

    parser.add_argument('--nbfilters', type=int, default=100, help='Num of filters in conv layer')

    parser.add_argument('--filter1_len', type=int, default=3, help='filter length in 1st conv layer')
    parser.add_argument('--rnn_type', type=str, default='LSTM', help='Recurrent type')
    parser.add_argument('--lstm_units', type=int, default=100, help='Num of hidden units in recurrent layer')

    # parser.add_argument('--project_hiddensize', type=int, default=100, help='num of units in projection layer')
    parser.add_argument('--optimizer', choices=['sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop'], help='updating algorithm', default='rmsprop')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for layers')
    parser.add_argument('--oov', choices=['random', 'embedding'], help="Embedding for oov word", default='random',
                        required=False)
    parser.add_argument('--l2_value', type=float, help='l2 regularizer value')
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint directory', default='./checkpoint')

    parser.add_argument('--train', default='prompt9_data/fold_' + str(fold) + '/train.tsv')  # "data/word-level/*.train"
    parser.add_argument('--dev', default='prompt9_data/fold_' + str(fold) + '/dev.tsv')
    parser.add_argument('--test', default='prompt9_data/fold_' + str(fold) + '/test.tsv')
    parser.add_argument('--prompt_id', type=int, default=p_id, help='prompt id of essay set')
    parser.add_argument('--init_bias', action='store_true',
                        help='init the last layer bias with average score of training data')
    parser.add_argument('--mode', type=str, choices=['mot', 'att', 'merged'], default='mot',
                        help='Mean-over-Time pooling or attention-pooling, or two pooling merged')

    args = parser.parse_args()



    d_path = 'prompt9_info/en.txt'
    datapaths = [args.train, args.dev, args.test, d_path]
    embedding_path = args.embedding_dict
    embedding = args.embedding
    embedd_dim = args.embedding_dim
    prompt_id = args.prompt_id


    (X_train, Y_train, D_train, mask_train, train_ids), (X_dev, Y_dev, D_dev, mask_dev, dev_ids), (
        X_test, Y_test, D_test, mask_test, test_ids), \
    vocab, vocab_size, embed_table, overal_maxlen, overal_maxnum, max_sentnum_d, init_mean_value = data_prepare.prepare_sentence_data(
        datapaths, embedding_path, embedding, embedd_dim, prompt_id, args.vocab_size, tokenize_text=True, to_lower=True,
        sort_by_len=False, vocab_path=None, score_index=6)

    # picture = np.loadtxt('img_feature')
    picture = get_picture()
    # new_pictures = []
    # for picture in pictures:
    #     picture = np.array(picture) / 255.0
    #     new_pictures.append(picture)
    # picture = np.array(new_pictures)
    print picture.shape
    # picture = get_picture('./prompt9_info/prompt9.png')
    train_num = X_train.shape[0]
    dev_num = X_dev.shape[0]
    test_num = X_test.shape[0]

    # p_train = np.empty(shape=[train_num, 4, 2048])
    # p_dev = np.empty(shape=[dev_num, 4, 2048])
    # p_test = np.empty(shape=[test_num, 4, 2048])

    img_size = 256
    # p_train = np.empty(shape=[train_num, 4, img_size, img_size, 3])
    # p_dev = np.empty(shape=[dev_num, 4, img_size, img_size, 3])
    # p_test = np.empty(shape=[test_num, 4, img_size, img_size, 3])
    p_train = np.empty(shape=[train_num, img_size, img_size, 3])
    p_dev = np.empty(shape=[dev_num, img_size, img_size, 3])
    p_test = np.empty(shape=[test_num, img_size, img_size, 3])
    for i in range(train_num):
        p_train[i] = picture

    for i in range(dev_num):
        p_dev[i] = picture
    for i in range(test_num):
        p_test[i] = picture


    embedd_dim = embed_table.shape[1]
    embed_table = [embed_table]

    max_sentnum = overal_maxnum
    max_sentlen = overal_maxlen


    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
    X_dev = X_dev.reshape((X_dev.shape[0], X_dev.shape[1]*X_dev.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

    D_train = D_train.reshape((D_train.shape[0], D_train.shape[1] * D_train.shape[2]))
    D_dev = D_dev.reshape((D_dev.shape[0], D_dev.shape[1] * D_dev.shape[2]))
    D_test = D_test.reshape((D_test.shape[0], D_test.shape[1] * D_test.shape[2]))
    logger.info("X_train shape: %s" % str(X_train.shape))

    model = build_model_fusion(args, vocab_size, max_sentnum, max_sentlen, embedd_dim, embed_table, True, init_mean_value)
    # model = build_model_with_topic(args, vocab_size, max_sentnum, max_sentlen, max_sentnum_d, embedd_dim, embed_table, True,
    #                    init_mean_value)

    evl = Evaluator(args.prompt_id, fold, X_train, X_dev, X_test, Y_train, Y_dev, Y_test, D_train, D_dev,
                    D_test, p_train, p_dev, p_test)

    # Initial evaluation
    if is_training:
        logger.info("Initial evaluation: ")
        # evl.evaluate(model, -1, print_info=True)
        logger.info("Train model")
        for ii in xrange(args.num_epochs):
            logger.info('Epoch %s/%s' % (str(ii+1), args.num_epochs))
            start_time = time.time()
            # model.fit({'word_input': X_train, 'word_input_d': D_train}, Y_train, batch_size=args.batch_size, epochs=1, verbose=0, shuffle=True)
            # model.fit({'word_input': X_train, 'p_input1': p_train[:, 0, :], 'p_input2': p_train[:, 1, :],
            #            'p_input3': p_train[:, 2, :], 'p_input4': p_train[:, 3, :]}, Y_train, batch_size=args.batch_size, epochs=1, verbose=0,
            #           shuffle=True)
            model.fit({'word_input': X_train, 'p': p_train}, Y_train,
                      batch_size=args.batch_size, epochs=1, verbose=0, shuffle=True)
            tt_time = time.time() - start_time
            logger.info("Training one epoch in %.3f s" % tt_time)
            evl.evaluate(model, ii+1)
            evl.print_info()



        evl.print_final_info()

if __name__ == '__main__':

    #main(0, 9)
    #main(1, 9)
    # main(2, 9)
    # main(3, 9)
    # main(4, 9)

    main(0, 9)
    main(1, 9)
    main(2, 9)
    main(3, 9)
    main(4, 9)


