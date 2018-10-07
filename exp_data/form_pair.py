import io
import torch
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--src_file",type=str)
parser.add_argument("--tgt_file",type=str)
parser.add_argument("--pair_file",type=str)

params = parser.parse_args()
src_file = params.src_file
tgt_file = params.tgt_file
pair_file = params.pair_file


src = src_file[:-1] + str(int(src_file[-1])+1)
tgt = tgt_file[:-1] + str(int(tgt_file[-1])+1)
a = torch.load(pair_file)
print(type(a))
print(a.shape)
try:
    f1 = io.open(src_file,'r', encoding='utf-8', newline='\n', errors='ignore')
    f2 = io.open(tgt_file,'r', encoding='utf-8', newline='\n', errors='ignore')
    f3 = io.open(src,'w', encoding='utf-8', newline='\n', errors='ignore')
    f4 = io.open(tgt,'w', encoding='utf-8', newline='\n', errors='ignore')
    f1_sen = f1.readlines()
    f2_sen = f2.readlines()
    for i in range(a.shape[0]):
        print(f1_sen[a[i,0]].strip(),file=f3)
        print(f2_sen[a[i,1]].strip(),file=f4)
except Exception as e:
    print(str(e))
print('success')
