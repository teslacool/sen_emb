import argparse
import torch
import numpy as np
import sys
from src_for_task.dataloader import  DataLoader
from src_for_task.util import init_dic
from src_for_task.model import attention
from src_for_task.sst import SSTEval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang" ,type=str, default="en", help="src language")
    parser.add_argument("--tgt_lang", type=str, default="de", help="tgt language")
    parser.add_argument("--emb_dim", type=int ,default=300, help="embedding dimension")
    parser.add_argument("--train_sen_path", type=str, default="data/train.de-en.en")
    parser.add_argument("--dev_sen_path", type=str, default="data/train.de-en.en")
    parser.add_argument("--test_sen_path", type=str, default="data/train.de-en.de")
    parser.add_argument("--src_emb", type=str, default="data/wiki.multi.en.vec", help="Reload source embeddings")
    parser.add_argument("--tgt_emb", type=str, default="data/wiki.multi.de.vec", help="Reload target embeddings")
    params = parser.parse_args()

    with torch.no_grad():
        init_dic(params)
        train_data = DataLoader(params.train_sen_dico, params.src_dico, batch_size= 512, shuffle=False)
        #TODO
        dev_data = DataLoader(params.dev_sen_dico ,params.tgt_dico ,batch_size= 512 ,shuffle=False)
        test_data = DataLoader(params.test_sen_dico,params.src_dico,batch_size= 512 ,shuffle=False)

        src_model = attention(params,len(params.src_dico),True)
        tgt_model = attention(params,len(params.tgt_dico),False)
        if torch.cuda.is_available():
            src_model = src_model.cuda()
            tgt_model = tgt_model.cuda()

        train_emb = []
        for data in train_data:
            emb = src_model(data)
            train_emb.append(emb.cpu().detach().numpy())
        train_emb = np.concatenate(train_emb,0)
        #TODO
        dev_emb = []
        for data in dev_data:
            emb = tgt_model(data)
            dev_emb.append(emb.cpu().detach().numpy())
        dev_emb = np.concatenate(dev_emb,0)

        test_emb = []
        for data in test_data:
            emb = src_model(data)
            test_emb.append(emb.cpu().detach().numpy())
        test_emb = np.concatenate(test_emb,0)

    eva = SSTEval()
    eva.run((train_emb,dev_emb,test_emb,params.train_y,params.dev_y,params.test_y))










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



