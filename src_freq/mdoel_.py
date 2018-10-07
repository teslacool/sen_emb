import torch
import torch.nn as nn
import numpy as np
from dictionary import Dictionary


class remove(nn.Module):
    def __init__(self,emb_dim):
        super(remove,self).__init__()
        self.emb_dim =emb_dim
        self.src_mapping = nn.Linear(emb_dim,emb_dim)
        self.tgt_mapping = nn.Linear(emb_dim,emb_dim)


    def forward(self, src_sen_emb,tgt_sen_emb):
        src_engin = self.src_mapping(src_sen_emb)
        src_norm = src_engin.norm(2,dim=2).repeat(1,self.emb_dim).view_as(src_engin)
        src_engin = src_engin / src_norm
        src_pc = torch.bmm(src_sen_emb,src_engin.transpose(2,1)).repeat(1,1,self.emb_dim).view_as(src_engin)
        src_sen_emb = src_sen_emb - src_engin * src_pc


        tgt_engin = self.tgt_mapping(tgt_sen_emb)
        tgt_norm = tgt_engin.norm(2,dim=2).repeat(1,self.emb_dim).view_as(tgt_engin)
        tgt_engin = tgt_engin / tgt_norm
        tgt_pc = torch.bmm(tgt_sen_emb,tgt_engin.transpose(2,1)).repeat(1,1,self.emb_dim).view_as(tgt_engin)
        tgt_sen_emb = tgt_sen_emb - tgt_engin * tgt_pc

        return (src_sen_emb,tgt_sen_emb)




class custom_eu_loss(nn.Module):
    def __init__(self,emb_dim):
        super(custom_eu_loss,self).__init__()
        self.emb_dim = emb_dim

    def forward(self,  src_sen_emb,tgt_sen_emb):


        ans = (src_sen_emb - tgt_sen_emb).view(-1, 1, self.emb_dim)
        dis = torch.bmm(ans, ans.transpose(1, 2)).view(-1)
        loss = 100 * dis/ dis.size()[0]
        loss = torch.sum(loss)
        return loss


class attention(nn.Module):

    def __init__(self,params):


        super(attention,self).__init__()
        self.emb_dim = params.emb_dim
        self.stage_u = params.stage_u
        reload_dir = params.reload_dir


        self.remove = remove(self.emb_dim)


        if self.stage_u==1:
            #reload all state_dict
            print("reload all state_dict")
            self.remove.src_mapping.load_state_dict(torch.load(reload_dir+"/srcmapping.dict"))
            self.remove.tgt_mapping.load_state_dict(torch.load(reload_dir+"/tgtmapping.dict"))
        else:
            print("reload nothing")


        self.custom_eu_loss = custom_eu_loss(self.emb_dim)



    def forward(self, data):

        src_sen_emb,tgt_sen_emb = data
        src_sen_emb = src_sen_emb.view(-1, 1, self.emb_dim)
        tgt_sen_emb = tgt_sen_emb.view(-1, 1, self.emb_dim)

        # src_sen_emb, tgt_sen_emb = self.remove(src_sen_emb, tgt_sen_emb)



        loss = self.custom_eu_loss(src_sen_emb,tgt_sen_emb)
        return src_sen_emb.squeeze(), tgt_sen_emb.squeeze(), loss



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

