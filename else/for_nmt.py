import numpy as np
import argparse
import os
import io
parser = argparse.ArgumentParser()

parser.add_argument("--src_f",type=str, default='')
parser.add_argument("--tgt_f",type=str, default='')
parser.add_argument("--sco_f",type=str, default='')

params = parser.parse_args()

src_f = params.src_f
tgt_f = params.tgt_f
sco_f = params.sco_f
gen_f = '../data/output_target'

print("save result in file: ", gen_f)



len1 = len(open(src_f , 'r').readlines())
len2 = len(open(tgt_f , 'r').readlines())
len3 = len(open(sco_f , 'r').readlines())

assert len1 == len2 and len2 == len3
with io.open(src_f,'r',encoding='utf8',errors='ignore',newline='\n') as src:
    with io.open(tgt_f,'r',encoding='utf8',errors='ignore',newline='\n') as tgt:
        with io.open(sco_f,'r',encoding='utf8',errors='ignore',newline='\n') as sco:
            with io.open(gen_f,'w',encoding='utf8',errors='ignore',newline='\n') as gen:
                src_sen = src.readline().rstrip()
                while True:
                    src_sens = []
                    tgt_sens = []
                    scores = []
                    src_sens.append(src_sen)
                    tgt_sens.append(tgt.readline().rstrip())
                    scores.append(float(sco.readline().rstrip()))
                    while True:
                        src_sen = src.readline().rstrip()
                        if not src_sen == src_sens[-1]:
                            break
                        if len(src_sens)==10:
                            break
                        src_sens.append(src_sen)
                        tgt_sens.append(tgt.readline().rstrip())
                        scores.append(float(sco.readline().rstrip()))
                    sco_np = np.array(scores)
                    indices = np.argsort(sco_np)
                    first_index = indices[-1]

                    # print(src_sens[first_index],file=gen)
                    print(tgt_sens[first_index],file=gen)
                    # print(scores[first_index],file=gen)


                    if src_sen=='':
                        if tgt.readline()=='' and sco.readline()=='':
                            print("successul!")
                        else:
                            print("one error!")
                        break






