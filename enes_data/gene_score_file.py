import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dir",type=str,default='')
parser.add_argument("--src_f",type=str, default='')
parser.add_argument("--tgt_f",type=str, default='')
parser.add_argument("--sco_f",type=str, default='')
parser.add_argument("--thre2",type=float, default=0)
params = parser.parse_args()
dir = params.dir
src_f = dir + params.src_f
tgt_f = dir + params.tgt_f
sco_f = dir + params.sco_f



threshold1 = 0.65
threshold2 = params.thre2
print("threshold2:%f"%threshold2)
suffix = str(int(threshold2*100))
gen_f = dir + 'out_' + sco_f[-2:] + '_' + suffix
print("output file",gen_f)
len1 = len(open(src_f , 'r').readlines())
len2 = len(open(tgt_f , 'r').readlines())
len3 = len(open(sco_f , 'r').readlines())
filter = 0
obtain = 0
assert len1 == len2 and len2 == len3
with open(src_f,'r') as src:
    with open(tgt_f,'r') as tgt:
        with open(sco_f,'r') as sco:
            with open(gen_f,'w') as gen:
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
                        src_sens.append(src_sen)
                        tgt_sens.append(tgt.readline().rstrip())
                        scores.append(float(sco.readline().rstrip()))
                    sco_np = np.array(scores)
                    indices = np.argsort(sco_np)
                    first_index = indices[-1]
                    if len(scores)>1:
                        second_index = indices[-2]
                    if   scores[first_index] >= threshold1:
                        if (len(scores)>1 and scores[first_index]-scores[second_index] >= threshold2) or len(scores) == 1:
                            print(src_sens[first_index],file=gen)
                            print(tgt_sens[first_index],file=gen)
                            print(scores[first_index],file=gen)
                            obtain += 1
                        else:
                            filter += 1
                    else:
                        filter += 1

                    if src_sen=='':
                        if tgt.readline()=='' and sco.readline()=='':
                            print("successul!")
                            print("drop/obtain {}/{}".format(filter,obtain))
                        else:
                            print("one error!")
                        break






