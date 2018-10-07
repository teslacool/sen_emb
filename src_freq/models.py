# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
from torch import nn
import io
from .utils import load_embeddings, normalize_embeddings
from logging import getLogger
from .dictionary import SenDictionary
import pickle
logger = getLogger()
from sklearn.decomposition import TruncatedSVD
class Discriminator(nn.Module):

    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.emb_dim = params.emb_dim
        self.dis_layers = params.dis_layers
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout
        self.dis_input_dropout = params.dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)

def get_score(src_emb, tgt_emb,score_file):

    src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
    tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
    assert src_emb.size()[0] == tgt_emb.size()[0]
    # average_dist1 = torch.from_numpy(get_nn_avg_dist(tgt_emb, src_emb, 10))
    # average_dist2 = torch.from_numpy(get_nn_avg_dist(src_emb, tgt_emb, 10))
    # average_dist1 = average_dist1.type_as(src_emb)
    # average_dist2 = average_dist2.type_as(tgt_emb)
    n_src = src_emb.size(0)
    src_emb = src_emb.view(n_src, 1, -1)
    tgt_emb = tgt_emb.view(n_src, 1, -1)
    score = torch.bmm(src_emb, tgt_emb.transpose(1, 2)).squeeze()
    # score = 2 * score-average_dist1-average_dist2
    score = score.detach().cpu().numpy().tolist()
    with open(score_file, 'w') as f:
        for s in score:
            print(s, file=f)

def build_model(params, with_dis):
    """
    Build all components of the model.
    """
    # source embeddings
    params.pos_emb = position_encoding_init(1000,300)

    src_dico= load_embeddings(params, source=True)
    # torch.save(src_dico,'/home/jinhua/code/news_data_select/data/src_dico')
    # src_dico=torch.load('/home/jinhua/code/news_data_select/data/src_dico')
    update_dico_word_fren(src_dico,params,source=True)
    # with open("dumped/debug/%s_dic.pkl"% params.src_lang,'wb') as f:
    #     pickle.dump(src_dico,f)
    params.src_dico = src_dico
    print("load src sentence  from %s"%params.src_sen_path)
    src_sen_dic = read_sentence_embeddings(params,source=True,procru=False)
    src_emb = nn.Embedding(len(src_sen_dic), params.emb_dim, sparse=True)
    _src_emb=[src_sen_dic.id2vec[i] for i in range(len(src_sen_dic))]

    # _src_emb = np.concatenate(_src_emb,0)
    _src_emb = remove_pc(np.concatenate(_src_emb, 0))
    src_emb.weight.data.copy_(torch.from_numpy(_src_emb))
    # logger.info("load src sentence for procrustes from %s" % params.tgt_sen_path_pro)
    # src_sen_dic_ = read_sentence_embeddings(params, source=True,procru=True)
    # src_emb_ = nn.Embedding(len(src_sen_dic_), params.sen_emb_dim, sparse=True)
    # _src_emb_ = [src_sen_dic_.id2vec[i] for i in range(len(src_sen_dic_))]
    # _src_emb_ = remove_pc(np.concatenate(_src_emb_, 0))
    # src_emb_.weight.data.copy_(torch.from_numpy(_src_emb_) )

    if params.tgt_lang:

        tgt_dico = load_embeddings(params, source=False)
        # torch.save(tgt_dico, '/home/jinhua/code/news_data_select/data/tgt_dico')
        # tgt_dico = torch.load('/home/jinhua/code/news_data_select/data/tgt_dico')

        update_dico_word_fren(tgt_dico, params, source=False)
        params.tgt_dico = tgt_dico
        print("load tgt sentence from %s" % params.tgt_sen_path)
        tgt_sen_dic = read_sentence_embeddings(params,source=False,procru=False)
        tgt_emb = nn.Embedding(len(tgt_sen_dic),params.emb_dim,sparse=True)
        _tgt_emb = [tgt_sen_dic.id2vec[i] for i in range(len(tgt_sen_dic))]
        # _tgt_emb = np.concatenate(_tgt_emb, 0)
        _tgt_emb = remove_pc(np.concatenate(_tgt_emb,0))
        tgt_emb.weight.data.copy_(torch.from_numpy(_tgt_emb))
        # logger.info("load tgt sentence for procrustes from %s" % params.tgt_sen_path_pro)
        # tgt_sen_dic_ = read_sentence_embeddings(params, source=False, procru=True)
        # tgt_emb_ = nn.Embedding(len(tgt_sen_dic_), params.sen_emb_dim, sparse=True)
        # _tgt_emb_ = [tgt_sen_dic_.id2vec[i] for i in range(len(tgt_sen_dic_))]
        # _tgt_emb_ = remove_pc(np.concatenate(_tgt_emb_, 0))
        # tgt_emb_.weight.data.copy_(torch.from_numpy(_tgt_emb_))
    # else:
    #     tgt_emb = None
        # tgt_emb_ = None

    # mapping
    # mapping = nn.Linear(params.sen_emb_dim, params.sen_emb_dim, bias=False)
    # if getattr(params, 'map_id_init', True):
    #     mapping.weight.data.copy_(torch.diag(torch.ones(params.sen_emb_dim)))
    #
    # # discriminator
    # # discriminator = Discriminator(params) if with_dis else None
    #
    # # cuda
    # if params.cuda:
    #     src_emb.cuda()
    #     # src_emb_.cuda()
    #     if params.tgt_lang:
    #         tgt_emb.cuda()
            # tgt_emb_.cuda()
    #     mapping.cuda()
    #     # if with_dis:
        #     discriminator.cuda()

    # normalize embeddings
    # params.src_mean = normalize_embeddings(src_emb.weight.data, params.normalize_embeddings)
    # if params.tgt_lang:
    #     params.tgt_mean = normalize_embeddings(tgt_emb.weight.data, params.normalize_embeddings)

    return src_emb, tgt_emb,None,None, None


def position_encoding_init(n_position, d_pos_vec):

    #no padding
    position_enc = np.array([
        [pos / np.power(10000, 2. * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[0:, 0::2] = np.sin(position_enc[0:, 0::2]) # dim 2i
    position_enc[0:, 1::2] = np.cos(position_enc[0:, 1::2]) # dim 2i+1
    return position_enc


def read_sentence_embeddings(params, source,procru=False):
    """
    Reload pretrained embeddings from a text file.
    """
    id2sen = {}

    # load pretrained embeddings
    lang = params.src_lang if source else params.tgt_lang
    if not procru:
        sen_path = params.src_sen_path if source else params.tgt_sen_path
    else:
        sen_path = params.src_sen_path_pro if source else params.tgt_sen_path_pro
    with io.open(sen_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            sen = line.rstrip()
            sen =  sen.lower()

            if False:
                logger.warning("Sentence '%s' found twice in %s sentence  file"
                                   % (sen, 'source' if source else 'target'))
            else:
                id2sen[i] = sen

    logger.info("Loaded %i sentence." % len(id2sen))


    dico = SenDictionary(id2sen,lang,params.src_dico if source else params.tgt_dico,params.pos_emb)
    return dico

def remove_pc(X, npc=1):
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    pc = svd.components_
    if npc == 1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX

def update_dico_word_fren(dico,params,source=True):
    freq_file =  params.src_word_freq if source else params.tgt_word_freq
    with open(freq_file,'r') as f:
        for m,line in enumerate(f):
            words = line.strip().split(' ')
            for word in words:
                if word not in dico:
                    continue
                wordid = dico.word2id[word]
                dico.id2cnt[wordid] += 1
                dico.total_cnt += 1
            if m > 20000:
                break
    for i in range(len(dico)):
        dico.id2freq[i] =dico.id2cnt[i]/float(dico.total_cnt)
