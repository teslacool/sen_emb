
import random
import numpy as np
import torch
from torch.autograd import Variable



class DataLoader(object):
    ''' For data iteration '''

    def __init__(self,  sen_dico, dico, batch_size=64, shuffle=True):
        self.dico = dico

        self.sen_dico = sen_dico
        self.idx = [ i for i in range(len(self.sen_dico))]

        self._n_batch = int(np.ceil(len(sen_dico) / batch_size))
        self._batch_size = batch_size
        self._iter_count = 0

        self.pad_index = len(self.dico)
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

        def pad_to_longest(src_insts):
            ''' Pad the instance to the max seq length in batch '''

            src_max_len = max(len(inst) for inst in src_insts)


            src_inst_data = np.array([
                inst + [self.pad_index] * (src_max_len - len(inst))
                for inst in src_insts])


            # inst_position = np.array([
            #     [pos_i + 1 if w_i != Constants.PAD else 0 for pos_i, w_i in enumerate(inst)]
            #     for inst in inst_data])

            src_inst_data_tensor = Variable(
                torch.LongTensor(src_inst_data))


            # inst_position_tensor = Variable(
            #     torch.LongTensor(inst_position), volatile=self.test)

            if torch.cuda.is_available():
                src_inst_data_tensor = src_inst_data_tensor.cuda()


                # inst_position_tensor = inst_position_tensor.cuda()
            return src_inst_data_tensor
        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            sen_index = self.idx[start_idx:end_idx]
            src_insts = [self.sen_dico[i] for i in sen_index]

            src_data = pad_to_longest(src_insts)

            return src_data

        else:

            if self._need_shuffle:
                self.shuffle()

            self._iter_count = 0
            raise StopIteration()
