# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2017-01-07 16:01:25
# @Last Modified by:   feidong1991
# @Last Modified time: 2017-02-12 21:36:02
import reader
import utils
import keras.backend as K
import numpy as np

logger = utils.get_logger("Prepare data ...")


def prepare_sentence_data(datapaths, embedding_path=None, embedding='word2vec', embedd_dim=100, prompt_id=1, vocab_size=0, tokenize_text=True, \
                         to_lower=True, sort_by_len=False, vocab_path=None, score_index=6):

    assert len(datapaths) == 4, "data paths should include train, dev and test path"
    (train_x, train_y, train_prompts, train_ids), (dev_x, dev_y, dev_prompts, dev_ids), (test_x, test_y, test_prompts, test_ids), vocab, overal_maxlen, overal_maxnum = \
        reader.get_data(datapaths, prompt_id, vocab_size, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=None, score_index=6)

    train_d, max_sentnum = reader.read_description(datapaths[3], vocab, len(train_x), tokenize_text=True, to_lower=True)
    dev_d, max_sentnum = reader.read_description(datapaths[3], vocab, len(dev_x), tokenize_text=True, to_lower=True)
    test_d, max_sentnum = reader.read_description(datapaths[3], vocab, len(test_x), tokenize_text=True, to_lower=True)


    X_train, y_train, mask_train = utils.padding_sentence_sequences(train_x, train_y, overal_maxnum, overal_maxlen, post_padding=True)
    X_dev, y_dev, mask_dev = utils.padding_sentence_sequences(dev_x, dev_y, overal_maxnum, overal_maxlen, post_padding=True)
    X_test, y_test, mask_test = utils.padding_sentence_sequences(test_x, test_y, overal_maxnum, overal_maxlen, post_padding=True)

    D_train, mask_d_train = utils.padding_des_sequences(train_d, max_sentnum, overal_maxlen, post_padding=True)
    D_dev, mask_d_dev = utils.padding_des_sequences(dev_d, max_sentnum, overal_maxlen, post_padding=True)
    D_test, mask_d_test = utils.padding_des_sequences(test_d, max_sentnum, overal_maxlen, post_padding=True)

    if prompt_id:
        train_pmt = np.array(train_prompts, dtype='int32')
        dev_pmt = np.array(dev_prompts, dtype='int32')
        test_pmt = np.array(test_prompts, dtype='int32')

    train_mean = y_train.mean(axis=0)
    train_std = y_train.std(axis=0)
    dev_mean = y_dev.mean(axis=0)
    dev_std = y_dev.std(axis=0)
    test_mean = y_test.mean(axis=0)
    test_std = y_test.std(axis=0)


    # We need the dev and test sets in the original scale for evaluation
    # dev_y_org = y_dev.astype(reader.get_ref_dtype())
    # test_y_org = y_test.astype(reader.get_ref_dtype())

    # Convert scores to boundary of [0 1] for training and evaluation (loss calculation)
    Y_train = reader.get_model_friendly_scores(y_train, prompt_id)
    Y_dev = reader.get_model_friendly_scores(y_dev, prompt_id)
    Y_test = reader.get_model_friendly_scores(y_test, prompt_id)
    scaled_train_mean = reader.get_model_friendly_scores(train_mean, prompt_id)
    # print Y_train.shape

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(X_train.shape))
    logger.info('  dev X shape:   ' + str(X_dev.shape))
    logger.info('  test X shape:  ' + str(X_test.shape))

    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev.shape))
    logger.info('  test Y shape:  ' + str(Y_test.shape))

    logger.info('  train_y mean: %s, stdev: %s, train_y mean after scaling: %s' %
                (str(train_mean), str(train_std), str(scaled_train_mean)))

    if embedding_path:
        embedd_dict, embedd_dim, _ = utils.load_word_embedding_dict(embedding, embedding_path, vocab, logger, embedd_dim)
        embedd_matrix = utils.build_embedd_table(vocab, embedd_dict, embedd_dim, logger, caseless=True)
    else:
        embedd_matrix = None

    return (X_train, Y_train, D_train, mask_train, train_ids), (X_dev, Y_dev, D_dev, mask_dev, dev_ids), (X_test, Y_test, D_test ,mask_test, test_ids), \
            vocab, len(vocab), embedd_matrix, overal_maxlen, overal_maxnum, max_sentnum, scaled_train_mean



