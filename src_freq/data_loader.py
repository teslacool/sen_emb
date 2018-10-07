
import random
import numpy as np
import torch
from torch.autograd import Variable



class DataLoader(object):
    ''' For data iteration '''

    def __init__(self, params, src_sen_dico, tgt_sen_dico, batch_size=64, shuffle=True):
        self.src_dico = params.src_dico
        self.tgt_dico = params.tgt_dico
        self.src_sen_dico = src_sen_dico
        self.tgt_sen_dico = tgt_sen_dico
        self.idx = [ i for i in range(len(self.src_sen_dico))]
        assert len(src_sen_dico) == len(tgt_sen_dico)
        self._n_batch = int(np.ceil(len(src_sen_dico) / batch_size))
        self._batch_size = batch_size
        self._iter_count = 0
        assert len(self.src_dico)== len(self.tgt_dico)
        self.pad_index = len(self.src_dico)
        self._need_shuffle = shuffle
        if self._need_shuffle:
            self.shuffle()

    def shuffle(self):
        ''' Shuffle data for a brand new start '''

        random.shuffle(self.idx)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''



        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            sen_index = self.idx[start_idx:end_idx]
            src_insts = [self.src_sen_dico.id2vec[i] for i in sen_index]
            tgt_insts = [self.tgt_sen_dico.id2vec[i] for i in sen_index]
            # src_insts = self.src_sen_dico[src_insts]
            src = np.concatenate(src_insts,0)
            tgt = np.concatenate(tgt_insts,0)
            src = torch.from_numpy(src).cuda().float()
            tgt = torch.from_numpy(tgt).cuda().float()
            src.requires_grad=False
            tgt.requires_grad=False
            return src,tgt

        else:

            if self._need_shuffle:
                self.shuffle()

            self._iter_count = 0
            raise StopIteration()
