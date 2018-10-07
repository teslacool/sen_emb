import numpy
import os
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("threshold2",type=str,default='')
params = parser.parse_args()
threshold2 = params.threshold2
print('threshold2: ',threshold2)
fde_out = open('ende.de.'+threshold2, 'w', encoding='utf-8')
fen_out = open('ende.en.'+threshold2, 'w', encoding='utf-8')
fscore_out = open('ende.score.'+threshold2, 'w', encoding='utf-8')
for i in range(13):
	data = ('out_0' + str(i) +'_'+threshold2) if i<10 else ('out_' + str(i) +'_'+threshold2)
	fdata = open(data, 'r', encoding='utf-8')
	cnt = 0
	line = fdata.readline()
	while line:
		if cnt % 3 == 0:
			fde_out.write(line)
		elif cnt % 3 == 1:
			fen_out.write(line)
		else:
			fscore_out.write(line)
		cnt += 1
		line = fdata.readline()
	fdata.close()


fde_out.close()
fen_out.close()
fscore_out.close()

