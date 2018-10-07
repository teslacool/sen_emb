import torch
import torch.nn as nn
import numpy as np
from dictionary import Dictionary


class remove(nn.Module):
    def __init__(self,emb_dim):
        super(remove,self).__init__()
        self.emb_dim =emb_dim

        self.src_mapping = nn.Linear(emb_dim,emb_dim)



    def forward(self, src_sen_emb):
        src_engin = self.src_mapping(src_sen_emb)
        src_norm = src_engin.norm(2,dim=2).repeat(1,self.emb_dim).view_as(src_engin)
        src_engin = src_engin / src_norm
        src_pc = torch.bmm(src_sen_emb,src_engin.transpose(2,1)).repeat(1,1,self.emb_dim).view_as(src_engin)
        src_sen_emb = src_sen_emb - src_engin * src_pc




        return (src_sen_emb)





class attention(nn.Module):

    def __init__(self,params,n_src_vocab,src):


        super(attention,self).__init__()
        self.emb_dim = params.emb_dim
        self.src = src



        self.pos_emb = nn.Embedding(1000,params.emb_dim)
        self.pos_emb.weight.data = position_encoding_init(1000,params.emb_dim)

        self.src_word_emb = nn.Embedding(n_src_vocab+1,params.emb_dim, padding_idx=n_src_vocab,)
        self.src_word_emb.weight.data = torch.from_numpy(word_emb_init(params.src_dico,params.emb_dim)).float()
        self.src_word_emb.weight.requires_grad = False

        self.tgt_word_emb = nn.Embedding(n_src_vocab+1,params.emb_dim, padding_idx=n_src_vocab)
        self.tgt_word_emb.weight.data = torch.from_numpy(word_emb_init(params.tgt_dico,params.emb_dim)).float()
        self.tgt_word_emb.weight.requires_grad = False
        self.mapping = nn.Linear(params.emb_dim,500,bias=False)
        self.mapping1 = nn.Linear(500, 1, bias=False)
        self.remove = remove(self.emb_dim)

        reload_dir = "model/model_wiki_" + params.src_lang + params.tgt_lang
        # reload_dir = "model/model_eu_ende"

        self.mapping.load_state_dict(torch.load(reload_dir+'/mapping.dict'))
        self.mapping1.load_state_dict(torch.load(reload_dir+'/mapping1.dict'))
        mapping_dict = "/srcmapping.dict" if src else "/tgtmapping.dict"
        self.remove.src_mapping.load_state_dict(torch.load(reload_dir + mapping_dict))


        self.soft = nn.Softmax(dim=1)
        self.m = nn.Tanh()




    def forward(self, data):
        def where(cond, x_1, x_2):
            return (cond.float() * x_1) + ((1. - cond.float()) * x_2)
        if self.src:
            src_index = data
            src_pos_emb = torch.from_numpy(np.array([i for i in range(src_index.shape[1])]))[None, :].repeat(
                src_index.shape[0], 1).cuda()
            src_pos_emb = self.pos_emb(src_pos_emb)
            src_emb = self.src_word_emb(src_index)
            src_emb = src_emb + 0.03 * src_pos_emb
            src_ = self.mapping(src_emb)
            src_ = self.m(src_)
            src_ = self.mapping1(src_)
            src_ = self.m(src_)
            src_bias = where(torch.eq(
                torch.from_numpy(np.array([200000] * (src_index.shape[0] * src_index.shape[1]))).cuda().view_as(src_index),
                src_index)
                , float('-inf'), 0.)[:, :, None]
            src_bias[src_bias != src_bias] = 0
            src_ = src_ + src_bias
            src_ = self.soft(src_)
            src_ = src_.repeat(1, 1, self.emb_dim)
            src_sen_emb = torch.sum(src_ * src_emb, 1)
            sen_emb = src_sen_emb.view(-1, 1, self.emb_dim)
        else:
            tgt_index = data
            tgt_pos_emb = torch.from_numpy(np.array([i for i in range(tgt_index.shape[1])]))[None, :].repeat(
                tgt_index.shape[0], 1).cuda()
            tgt_pos_emb = self.pos_emb(tgt_pos_emb)
            tgt_emb = self.tgt_word_emb(tgt_index)
            tgt_emb = tgt_emb + 0.03 * tgt_pos_emb
            tgt_ = self.mapping(tgt_emb)
            tgt_ = self.m(tgt_)
            tgt_ = self.mapping1(tgt_)
            tgt_ = self.m(tgt_)
            tgt_bias = where(torch.eq(
                torch.from_numpy(np.array([200000] * (tgt_index.shape[0] * tgt_index.shape[1]))).cuda().view_as(tgt_index),
                tgt_index)
                             , float('-inf'), 0.)[:, :, None]
            tgt_bias[tgt_bias != tgt_bias] = 0
            tgt_ = tgt_ + tgt_bias
            tgt_ = self.soft(tgt_)
            tgt_ = tgt_.repeat(1, 1, self.emb_dim)
            tgt_sen_emb = torch.sum(tgt_ * tgt_emb, 1)
            sen_emb = tgt_sen_emb.view(-1, 1, self.emb_dim)


        sen_emb = self.remove(sen_emb)




        return sen_emb.squeeze()







def word_emb_init(dico:Dictionary,emb_dim):
    vec=[]
    for i in range(len(dico)):
        vec.append(dico.word2vec[dico.id2word[i]])
    vec.append(np.array([0.]*emb_dim).reshape((1,emb_dim)))
    return  np.concatenate(vec,0)

def position_encoding_init(n_position, d_pos_vec):

    #no padding
    position_enc = np.array([
        [pos / np.power(10000, 2. * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[0:, 0::2] = np.sin(position_enc[0:, 0::2]) # dim 2i
    position_enc[0:, 1::2] = np.cos(position_enc[0:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def roll(tensor ,shift ,axis):
    assert  shift >= 0
    if shift ==0:
        return tensor
    if axis < 0:
        axis += tensor.dim()
    dim_size = tensor.size(axis)
    after_start = dim_size - shift

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift )
    return torch.cat([after,before],axis)

