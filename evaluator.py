# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2017-02-10 14:56:57
# @Last Modified by:   feidong1991
# @Last Modified time: 2017-02-14 14:10:46
from utils import rescale_tointscore, get_logger
from metrics import *
import numpy as np
from keras.models import Model
from keras import backend as K

logger = get_logger("Evaluate stats")


class Evaluator():
    def __init__(self, prompt_id, fold, train_x, dev_x, test_x, train_y, dev_y, test_y, train_d,
                 dev_d, test_d, p_train, p_dev, p_test):
        # self.dataset = dataset

        self.fold = fold
        self.prompt_id = prompt_id
        self.train_x, self.dev_x, self.test_x = train_x, dev_x, test_x
        self.train_y, self.dev_y, self.test_y = train_y, dev_y, test_y
        self.train_d = train_d
        self.dev_d = dev_d
        self.test_d = test_d
        self.train_y_org = rescale_tointscore(train_y, self.prompt_id)
        self.dev_y_org = rescale_tointscore(dev_y, self.prompt_id)
        self.test_y_org = rescale_tointscore(test_y, self.prompt_id)

        self.p_train = p_train
        self.p_dev = p_dev
        self.p_test = p_test
        self.best_dev = [-1, -1, -1, -1]
        self.best_test = [-1, -1, -1, -1]

    def calc_correl(self, train_pred, dev_pred, test_pred):
        self.train_pr = pearson(self.train_y_org, train_pred)
        self.dev_pr = pearson(self.dev_y_org, dev_pred)
        self.test_pr = pearson(self.test_y_org, test_pred)

        self.train_spr = spearman(self.train_y_org, train_pred)
        self.dev_spr = spearman(self.dev_y_org, dev_pred)
        self.test_spr = spearman(self.test_y_org, test_pred)

    def calc_kappa(self, train_pred, dev_pred, test_pred, weight='quadratic'):
        train_pred_int = np.rint(train_pred).astype('int32')
        dev_pred_int = np.rint(dev_pred).astype('int32')
        test_pred_int = np.rint(test_pred).astype('int32')
        self.train_qwk = kappa(self.train_y_org, train_pred, weight)
        self.dev_qwk = kappa(self.dev_y_org, dev_pred, weight)
        self.test_qwk = kappa(self.test_y_org, test_pred, weight)

    def calc_rmse(self, train_pred, dev_pred, test_pred):
        self.train_rmse = root_mean_square_error(self.train_y_org, train_pred)
        self.dev_rmse = root_mean_square_error(self.dev_y_org, dev_pred)
        self.test_rmse = root_mean_square_error(self.test_y_org, test_pred)

    def evaluate(self, model, epoch, print_info=False):

        # train_pred = model.predict({'word_input': self.train_x, 'word_input_d': self.train_d}, batch_size=32).squeeze()
        # dev_pred = model.predict({'word_input': self.dev_x, 'word_input_d': self.dev_d}, batch_size=32).squeeze()
        # test_pred = model.predict({'word_input': self.test_x,  'word_input_d': self.test_d}, batch_size=32).squeeze()

        # train_pred = model.predict({'word_input': self.train_x, 'p_input1': self.p_train[:, 0, :], 'p_input2': self.p_train[:, 1, :],
        # 'p_input3': self.p_train[:, 2, :], 'p_input4': self.p_train[:, 3, :]}, batch_size=32).squeeze()
        # dev_pred = model.predict({'word_input': self.dev_x, 'p_input1': self.p_dev[:, 0, :], 'p_input2': self.p_dev[:, 1, :],
        # 'p_input3': self.p_dev[:, 2, :], 'p_input4': self.p_dev[:, 3, :]}, batch_size=32).squeeze()
        # test_pred = model.predict({'word_input': self.test_x, 'p_input1': self.p_test[:, 0, :], 'p_input2': self.p_test[:, 1, :],
        # 'p_input3': self.p_test[:, 2, :], 'p_input4': self.p_test[:, 3, :]}, batch_size=32).squeeze()
        train_pred = model.predict({'word_input': self.train_x, 'p': self.p_train}, batch_size=32).squeeze()
        dev_pred = model.predict({'word_input': self.dev_x, 'p': self.p_dev}, batch_size=32).squeeze()
        test_pred = model.predict({'word_input': self.test_x, 'p': self.p_test}, batch_size=32).squeeze()

        train_pred_int = rescale_tointscore(train_pred, self.prompt_id)
        dev_pred_int = rescale_tointscore(dev_pred, self.prompt_id)
        test_pred_int = rescale_tointscore(test_pred, self.prompt_id)

       # self.calc_correl(train_pred_int, dev_pred_int, test_pred_int)
        self.calc_kappa(train_pred_int, dev_pred_int, test_pred_int)
        #self.calc_rmse(train_pred_int, dev_pred_int, test_pred_int)

        if self.dev_qwk > self.best_dev[0]:
            self.best_dev = [self.dev_qwk, 0, 0, 0]
            self.best_test = [self.test_qwk, 0, 0, 0]
            self.best_dev_epoch = epoch

        if print_info:
            self.print_info()

    def print_info(self):
        logger.info('[DEV]   QWK:  %.3f, PRS: %.3f, SPR: %.3f, RMSE: %.3f, (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f)' % (
            self.dev_qwk, 0, 0, 0, self.best_dev_epoch,
            self.best_dev[0], self.best_dev[1], self.best_dev[2], self.best_dev[3]))
        logger.info('[TEST]  QWK:  %.3f, PRS: %.3f, SPR: %.3f, RMSE: %.3f (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f)' % (
            self.test_qwk, 0, 0, 0, self.best_dev_epoch,
            self.best_test[0], self.best_test[1], self.best_test[2], self.best_test[3]))

        logger.info(
            '--------------------------------------------------------------------------------------------------------------------------')

    def print_final_info(self):
        logger.info(
            '--------------------------------------------------------------------------------------------------------------------------')
        with open(str(self.prompt_id) + '_fold' + str(self.fold) + '.txt', 'w')as w:
            w.write('[DEV]QWK\n')
            w.write(str(self.best_dev[0]) + '\n')

            w.write('[TEST]QWK\n')
            w.write(str(self.best_test[0]) + '\n')
        # logger.info('Missed @ Epoch %i:' % self.best_test_missed_epoch)
        # logger.info('  [TEST] QWK: %.3f' % self.best_test_missed)
        logger.info('Best @ Epoch %i:' % self.best_dev_epoch)
        logger.info('  [DEV]  QWK: %.3f,  PRS: %.3f, SPR: %.3f, RMSE: %.3f' % (
        self.best_dev[0], self.best_dev[1], self.best_dev[2], self.best_dev[3]))
        logger.info('  [TEST] QWK: %.3f,  PRS: %.3f, SPR: %.3f, RMSE: %.3f' % (
        self.best_test[0], self.best_test[1], self.best_test[2], self.best_test[3]))



