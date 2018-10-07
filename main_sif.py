# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import argparse

import torch

import io
from src_freq.models import build_model
from src_freq.models import get_score


def main():
    parser = argparse.ArgumentParser(description='Unsupervised training')


    parser.add_argument("--src_lang", type=str, default='en', help="Source language")
    parser.add_argument("--tgt_lang", type=str, default='es', help="Target language")
    parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
    parser.add_argument("--src_emb", type=str, default="/home/jinhua/code/news_data_select/data/vectors-en.txt", help="Reload source embeddings")
    parser.add_argument("--tgt_emb", type=str, default="/home/jinhua/code/news_data_select/data/vectors-de.txt", help="Reload target embeddings")
    parser.add_argument("--src_sen_path", type=str, default="", help="where to find your src sentence")
    parser.add_argument("--tgt_sen_path", type=str, default="", help="where to find your target sentence")
    parser.add_argument("--src_word_freq", type=str, default="", help="where to update your src word freq")
    parser.add_argument("--tgt_word_freq", type=str, default="", help="where to update your tgt word freq")
    parser.add_argument("--save_file", type=str, )

    # parse parameters
    params = parser.parse_args()

    print('src_sen_path',params.src_sen_path)
    print('tgt_sen_path',params.tgt_sen_path)


    params.src_word_freq = params.src_sen_path
    params.tgt_word_freq = params.tgt_sen_path
    # check parameters
    assert  torch.cuda.is_available()


    with torch.no_grad():
        src_emb, tgt_emb,src_emb_,tgt_emb_, mapping= build_model(params, True)
        get_score(src_emb.weight,tgt_emb.weight,params.save_file)
        print('success')

    print("over")

if __name__ == '__main__':
    main()







