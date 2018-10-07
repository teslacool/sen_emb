# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import numpy as np

logger = getLogger()


class Dictionary(object):



    def __init__(self, id2word, word2id,word2vec, lang):

        assert len(id2word) == len(word2id)
        self.id2word = id2word
        self.word2id = word2id
        self.total_cnt = int(0)
        self.word2vec = word2vec
        self.lang = lang
        self.check_valid()
        self.freq()



    def freq(self):
        self.id2cnt = {k: 0 for k, v in self.id2word.items()}
        self.id2freq = {k:0 for k,v in self.id2word.items() }

    def __len__(self):
        """
        Returns the number of words in the dictionary.
        """
        return len(self.id2word)

    def __getitem__(self, i):
        """
        Returns the word of the specified index.
        """
        return self.id2word[i]

    def __contains__(self, w):
        """
        Returns whether a word is in the dictionary.
        """
        return w in self.word2id

    def __eq__(self, y):
        """
        Compare the dictionary with another one.
        """
        self.check_valid()
        y.check_valid()
        if len(self.id2word) != len(y):
            return False
        return self.lang == y.lang and all(self.id2word[i] == y[i] for i in range(len(y)))

    def check_valid(self):
        """
        Check that the dictionary is valid.
        """
        assert len(self.id2word) == len(self.word2id)
        for i in range(len(self.id2word)):
            assert self.word2id[self.id2word[i]] == i

    def index(self, word):
        """
        Returns the index of the specified word.
        """
        return self.word2id[word]

    def freq_of_index(self,index):
        return self.id2freq[index]

    def prune(self, max_vocab):
        """
        Limit the vocabulary size.
        """
        assert max_vocab >= 1
        self.id2word = {k: v for k, v in self.id2word.items() if k < max_vocab}
        self.word2id = {v: k for k, v in self.id2word.items()}
        self.check_valid()

class SenDictionary(object):

    def __init__(self, id2sen, lang,dico:Dictionary, pos):

        self.id2sen = id2sen
        self.lang = lang
        self.dico = dico
        self.a = 0.001
        self.cal_vec_position(pos)




    def cal_vec_position(self,pos):
        vec=[]

        for id in range(len(self.id2sen)):
            vs = np.zeros(self.dico.word2vec[self.dico.id2word[0]].shape[1])
            posi_enc = np.zeros(self.dico.word2vec[self.dico.id2word[0]].shape[1])
            word_num = 0
            words = self.id2sen[id].strip().split(" ")
            for word in words:
                if word not in self.dico:
                    continue


                a_value = self.a/(self.a+self.dico.freq_of_index(self.dico.word2id[word]))
                # a_value = 1.
                vs = np.add(vs,np.multiply(a_value,self.dico.word2vec[word]))

                posi_enc = np.add(posi_enc,np.multiply(a_value,pos[word_num].reshape(self.dico.word2vec[self.dico.id2word[0]].shape[1])))
                word_num += 1
            if word_num==0:
                vs = np.add(vs, np.multiply(1, self.dico.word2vec['.']))
                posi_enc = np.zeros(self.dico.word2vec[self.dico.id2word[0]].shape[1])
            else:
                vs = np.divide(vs,word_num)
                posi_enc = np.divide(posi_enc,word_num)

            vec.append(np.add(vs,0.03*posi_enc[None,:self.dico.word2vec[self.dico.id2word[0]].shape[1]]))
            # vs = vs.reshape(1,300)
            # vec.append(vs)
        self.id2vec = {i:v for i,v in enumerate(vec)}

    def __len__(self):
        """
        Returns the number of words in the dictionary.
        """
        return len(self.id2sen)

    def __getitem__(self, i):
        """
        Returns the word of the specified index.
        """
        return self.id2sen[i]





def position_encode(pos,d_model):
    position_enc = np.array([pos/np.power(10000,2*(i // 2)/d_model) for i in range(d_model)])
    position_enc[0::2] = np.sin(position_enc[0::2])
    position_enc[1::2] = np.cos(position_enc[1::2])

    return position_enc

