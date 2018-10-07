# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
Binary classifier and corresponding datasets : MR, CR, SUBJ, MPQA
'''
from __future__ import absolute_import, division, unicode_literals

import io
import os
import numpy as np
import logging

from src_for_task.validation import InnerKFoldClassifier


class BinaryClassifierEval(object):
    def __init__(self, n_samples, seed=1111):
        self.seed = seed
        self.n_samples = n_samples



    def run(self, data, label):
        train_pos_emb, train_neg_emb, valid_pos_emb, valid_neg_emb = data
        enc_input_train = np.vstack((train_pos_emb, train_neg_emb))
        enc_input_valid = np.vstack((valid_pos_emb, valid_neg_emb))
        rng = np.random.RandomState(1234)
        perm = rng.permutation(label.shape[0])
        enc_input_train = enc_input_train[perm]
        enc_input_valid = enc_input_valid[perm]
        label = label[perm]

        classifier = {'nhid': 200, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': True,
                  'classifier': classifier,
                  'nhid':classifier['nhid'] , 'kfold': 10}
        clf = InnerKFoldClassifier(enc_input_train,enc_input_valid, label, config)
        devacc, testacc = clf.run()
        print('Dev acc : {0} Test acc : {1}\n'.format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc, 'ndev': self.n_samples,
                'ntest': self.n_samples}


class CREval(BinaryClassifierEval):
    def __init__(self, n_samples, seed=1111):
        print('***** Transfer task : CR *****\n\n')
        super(self.__class__, self).__init__(n_samples)


class MREval(BinaryClassifierEval):
    def __init__(self, n_samples, seed=1111):
        print('***** Transfer task : MR *****\n\n')
        super(self.__class__, self).__init__(n_samples)


class SUBJEval(BinaryClassifierEval):
    def __init__(self, n_samples, seed=1111):
        print('***** Transfer task : CR *****\n\n')
        super(self.__class__, self).__init__(n_samples)


class MPQAEval(BinaryClassifierEval):
    def __init__(self, n_samples, seed=1111):
        print('***** Transfer task : CR *****\n\n')
        super(self.__class__, self).__init__(n_samples)
