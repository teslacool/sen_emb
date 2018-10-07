import argparse
import torch
import numpy as np
import sys
from DataLoader import  DataLoader
from util import  init_dic
from model import  attention
from util import new_cal_topk_csls
from util import select_pair
from util import cal_topk_csls

from util import get_score

import scipy
import scipy.linalg
import os
import subprocess
best_top = 0.

def train_epoch(model,train_data,optimizer):
    model.train()
    total_loss = 0

    for data in train_data:
        optimizer.zero_grad()
        _,_,loss = model(data)
        total_loss += loss.data[0]
        loss.backward()
        optimizer.step()

    return total_loss


def val_epoch(model,val_data,params):
    with torch.no_grad():
        model.eval()
        src_sen_emb = []
        tgt_sen_emb = []
        for data in val_data:
            src,tgt,_ = model(data)
            src_sen_emb.append(src.cpu().detach().numpy())
            tgt_sen_emb.append(tgt.cpu().detach().numpy())
        src = np.concatenate(src_sen_emb,0)
        # src = remove_pc(src)
        tgt = np.concatenate(tgt_sen_emb,0)
        # tgt = remove_pc(tgt)
        top1 = cal_topk_csls(torch.from_numpy(src),torch.from_numpy(tgt))
        global best_top

        if top1 > best_top:
            save_dir = params.save_dir
            best_top = top1
            if not os.path.exists(save_dir):
                subprocess.Popen("mkdir %s"%save_dir,shell=True).wait()
            print(save_dir)
            if params.stage_u==0:
                # train mapping
                torch.save(model.mapping.state_dict(), save_dir + '/mapping.dict')
                torch.save(model.mapping1.state_dict(), save_dir + '/mapping1.dict')
            elif params.stage_u ==1:
                # train u
                torch.save(model.remove.src_mapping.state_dict(),save_dir+'/srcmapping.dict')
                torch.save(model.remove.tgt_mapping.state_dict(),save_dir+'/tgtmapping.dict')


            print("save model for top1 %f"%best_top)

def val(model,val_data,train_data,params):
    with torch.no_grad():
        model.eval()
        src_sen_emb = []
        tgt_sen_emb = []
        for data in val_data:
            src,tgt,_ = model(data)
            src_sen_emb.append(src.cpu().detach().numpy())
            tgt_sen_emb.append(tgt.cpu().detach().numpy())
        src = np.concatenate(src_sen_emb,0)
        # src = remove_pc(src)
        tgt = np.concatenate(tgt_sen_emb,0)
        # tgt = remove_pc(tgt)
        # src_sen_emb_ = []
        # tgt_sen_emb_ = []
        # for i,data in enumerate(train_data):
        #     src_,tgt_,_ = model(data)
        #     src_sen_emb_.append(src_.cpu().detach().numpy())
        #     tgt_sen_emb_.append(tgt_.cpu().detach().numpy())
        #
        # src_ = np.concatenate(src_sen_emb_, 0)
        # # src_ = remove_pc(src_)
        # tgt_ = np.concatenate(tgt_sen_emb_, 0)
        # # tgt_ = remove_pc(tgt_)
        # mapping = torch.nn.Linear(300,300,bias=False)
        # M = np.dot(tgt_.transpose(),src_)
        # u,s,v_t = scipy.linalg.svd(M, full_matrices=True)
        # mapping.weight.data.copy_(torch.from_numpy(u.dot(v_t)).float())
        new_cal_topk_csls(torch.from_numpy(src),torch.from_numpy(tgt),params,2000,200000,params.direction)
        # cal_topk_csls(torch.from_numpy(src),torch.from_numpy(tgt))
        #
        # src = mapping(torch.from_numpy(src)).detach().numpy()
        # cal_topk_csls(torch.from_numpy(src), torch.from_numpy(tgt))

def score(model,val_data,train_data,params):
    with torch.no_grad():
        model.eval()
        src_sen_emb = []
        tgt_sen_emb = []
        for data in val_data:
            src,tgt,_ = model(data)
            src_sen_emb.append(src.cpu().detach().numpy())
            tgt_sen_emb.append(tgt.cpu().detach().numpy())
        src = np.concatenate(src_sen_emb,0)
        # src = remove_pc(src)
        tgt = np.concatenate(tgt_sen_emb,0)
        # tgt = remove_pc(tgt)
        # src_sen_emb_ = []
        # tgt_sen_emb_ = []
        # for i,data in enumerate(train_data):
        #     src_,tgt_,_ = model(data)
        #     src_sen_emb_.append(src_.cpu().detach().numpy())
        #     tgt_sen_emb_.append(tgt_.cpu().detach().numpy())
        #
        # src_ = np.concatenate(src_sen_emb_, 0)
        # # src_ = remove_pc(src_)
        # tgt_ = np.concatenate(tgt_sen_emb_, 0)
        # # tgt_ = remove_pc(tgt_)
        # mapping = torch.nn.Linear(300,300,bias=False)
        # M = np.dot(tgt_.transpose(),src_)
        # u,s,v_t = scipy.linalg.svd(M, full_matrices=True)
        # mapping.weight.data.copy_(torch.from_numpy(u.dot(v_t)).float())
        # # cal_topk_csls(torch.from_numpy(src),torch.from_numpy(tgt))
        #
        # src = mapping(torch.from_numpy(src)).detach().numpy()
        select_pair(torch.from_numpy(src), torch.from_numpy(tgt))


def train(model,train_data,val_data ,optimizer,epoch,params):
    val_epoch(model, val_data,params)
    for epoch_i in range(epoch):
        total_loss = train_epoch(model,train_data,optimizer)
        print("epoch--%d,   total loss  --  "%epoch_i,total_loss)
        if params.stage_u==0:
            interval = 10
        else:
            interval =1
        if epoch_i%interval == 0:
            val_epoch(model,val_data,params)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang" ,type=str, default="en", help="src language")
    parser.add_argument("--tgt_lang", type=str, default="de", help="tgt language")
    parser.add_argument("--emb_dim", type=int ,default=300, help="embedding dimension")
    parser.add_argument("--src_sen_path", type=str, default="data/train.de-en.en", help="where to find your src sentence")
    parser.add_argument("--tgt_sen_path", type=str, default="data/train.de-en.de", help="where to find your target sentence")
    parser.add_argument("--src_sen_path_test", type=str, default="data/test.en", help="where to find your src sentence for test")
    parser.add_argument("--tgt_sen_path_test", type=str, default="data/test.de", help="where to find your target sentence for test")
    parser.add_argument("--score_file", type=str, default="data/score", help="score file name")
    parser.add_argument("--src_emb", type=str, default="data/wiki.multi.en.vec", help="Reload source embeddings")
    parser.add_argument("--tgt_emb", type=str, default="data/wiki.multi.de.vec", help="Reload target embeddings")
    parser.add_argument("--epoch", type=int, default=100000, help="epoch number")
    parser.add_argument("--hid_dim", type=int, default=500, help="hidden dimension")
    parser.add_argument("--val", type=bool_flag, default=False, help="just validate our model")
    parser.add_argument("--score", type=bool_flag, default=False, help="score")
    parser.add_argument("--stage_u", type=int,default=0, help=" 0: train mapping 1: reload mapping train u 2: reload u")
    parser.add_argument("--reload_dir", type=str,default='')
    parser.add_argument("--save_dir", type=str, default='')
    parser.add_argument("--direction" , type=bool_flag,default=True)
    params = parser.parse_args()

    init_dic(params)
    train_data = DataLoader(params, params.src_sen_dico, params.tgt_sen_dico, batch_size= 512, shuffle=True)
    val_data = DataLoader(params,params.src_sen_dico_test,params.tgt_sen_dico_test,batch_size= 512,shuffle=False)

    model = attention(params,len(params.src_dico))
    if torch.cuda.is_available():
        model = model.cuda()

    score(model,val_data,train_data,params)



def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in ['off', 'false', '0']:
        return False
    if s.lower() in ['on', 'true', '1']:
        return True
    raise argparse.ArgumentTypeError("invalid value for a boolean flag (0 or 1)")


if __name__ == '__main__':
    main()



