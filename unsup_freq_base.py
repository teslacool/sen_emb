# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time
import json
import argparse
from collections import OrderedDict
import numpy as np
import torch
from src_freq.custom import new_cal_topk_csls
from src_freq.utils import bool_flag, initialize_exp
from src_freq.models import build_model
from src_freq.custom import cal_topk_cos
from src_freq.custom import cal_topk_csls
from src_freq.custom import cal_topk_eu
from src_freq.custom import procrustes
from src_freq.custom import get_score

VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000'


# main
parser = argparse.ArgumentParser(description='Unsupervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
# data
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='de', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--sen_emb_dim", type=int, default=300, help="Sentence embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
# mapping
parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")



# dictionary creation parameters (for refinement)
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=15000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default="data/wiki.multi.en.vec", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default="data/wiki.multi.de.vec", help="Reload target embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
parser.add_argument("--src_sen_path", type=str, default="data/test.en", help="where to find your src sentence")
parser.add_argument("--tgt_sen_path", type=str, default="data/test.de", help="where to find your target sentence")

parser.add_argument("--src_word_freq", type=str, default="data/train.de-en.en2", help="where to update your src word freq")
parser.add_argument("--tgt_word_freq", type=str, default="data/train.de-en.de2", help="where to update your tgt word freq")
parser.add_argument("--reload_from_pkl", type=bool_flag, default=False, help='whether reload word dic from pkl file')
parser.add_argument("--score_file", type=str, default='data/score_iwslt_baseline')
parser.add_argument("--score",type=bool_flag, default=False)
# parse parameters
params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()

assert os.path.isfile(params.src_emb)
assert os.path.isfile(params.tgt_emb)


# build model / trainer / evaluator
logger = initialize_exp(params)
with torch.no_grad():
    src_emb, tgt_emb,src_emb_,tgt_emb_, mapping= build_model(params, True)
    # # procrustes(src_emb, tgt_emb,src_emb_,tgt_emb_, mapping)
    # if params.score:
    #     get_score(src_emb.weight,tgt_emb.weight,params.score_file)
    # else:
    #     cal_topk_csls(src_emb.weight,tgt_emb.weight,params)
    get_score(src_emb.weight, tgt_emb.weight, params.score_file)








