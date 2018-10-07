# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
SST - binary classification
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import logging
import numpy as np

from .validation import SplitClassifier


class SSTEval(object):
    def __init__(self,  nclasses=2, seed=1111):
        self.seed = seed

        # binary of fine-grained
        assert nclasses in [2, 5]
        self.nclasses = nclasses
        self.task_name = 'Binary' if self.nclasses == 2 else 'Fine-Grained'
        self.classifier = {'nhid': 200, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}



    def run(self, data):
        train_x,dev_x,test_x,train_y,dev_y,test_y=data

        config_classifier = {'nclasses': self.nclasses, 'seed': self.seed,
                             'usepytorch': True,
                             'classifier': self.classifier}

        clf = SplitClassifier(X={'train': train_x,
                                 'valid': dev_x,
                                 'test': test_x},
                              y={'train': train_y,
                                 'valid': dev_y,
                                 'test': test_y},
                              config =config_classifier)

        devacc, testacc = clf.run()
        print('\nDev acc : {0} Test acc : {1} for \
            SST {2} classification\n'.format(devacc, testacc, self.task_name))

        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(dev_y),
                'ntest': len(test_y)}
