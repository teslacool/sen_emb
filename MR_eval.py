import argparse
import torch
import numpy as np
import sys
from src_for_task.dataloader import  DataLoader
from src_for_task.util import init_dict
from src_for_task.model import attention
from src_for_task.binary import MREval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang" ,type=str, default="en", help="src language")
    parser.add_argument("--tgt_lang", type=str, default="de", help="tgt language")
    parser.add_argument("--emb_dim", type=int ,default=300, help="embedding dimension")
    parser.add_argument("--src_pos_sen_path", type=str, default="data/train.de-en.en")
    parser.add_argument("--src_neg_sen_path", type=str, default="data/train.de-en.en")
    parser.add_argument("--src_emb", type=str, default="data/wiki.multi.en.vec", help="Reload source embeddings")
    parser.add_argument("--tgt_emb", type=str, default="data/wiki.multi.de.vec", help="Reload target embeddings")
    params = parser.parse_args()

    with torch.no_grad():
        init_dict(params)
        train_pos = DataLoader(params.train_pos,params.src_dico,batch_size= 512, shuffle=False)
        train_neg = DataLoader(params.train_neg,params.src_dico,batch_size= 512, shuffle=False)
        valid_pos = DataLoader(params.valid_pos,params.tgt_dico,batch_size= 512, shuffle=False)
        valid_neg = DataLoader(params.valid_neg,params.tgt_dico,batch_size= 512, shuffle=False)


        src_model = attention(params,len(params.src_dico),True)
        tgt_model = attention(params,len(params.tgt_dico),False)

        if torch.cuda.is_available():
            src_model = src_model.cuda()
            tgt_model = tgt_model.cuda()

        train_pos_emb = []
        for data in train_pos:
            emb = src_model(data)
            train_pos_emb.append(emb.cpu().detach().numpy())
        train_pos_emb = np.concatenate(train_pos_emb,0)

        train_neg_emb = []
        for data in train_neg:
            emb = src_model(data)
            train_neg_emb.append(emb.cpu().detach().numpy())
        train_neg_emb = np.concatenate(train_neg_emb,0)

        valid_pos_emb = []
        for data in valid_pos:
            emb = tgt_model(data)
            valid_pos_emb.append(emb.cpu().detach().numpy())
        valid_pos_emb = np.concatenate(valid_pos_emb,0)

        valid_neg_emb = []
        for data in valid_neg:
            emb =tgt_model(data)
            valid_neg_emb.append(emb.cpu().detach().numpy())
        valid_neg_emb = np.concatenate(valid_neg_emb,0)

        assert train_neg_emb.shape[0] == valid_neg_emb.shape[0]
        assert train_pos_emb.shape[0] == valid_pos_emb.shape[0]
        label = np.array([1] * train_pos_emb.shape[0] + [0] * train_neg_emb.shape[0])



    eva = MREval(label.shape[0])
    eva.run((train_pos_emb,train_neg_emb,valid_pos_emb,valid_neg_emb,),label)










    print("over")



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



