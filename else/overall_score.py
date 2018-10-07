import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file",type=str, help='file to integrate')
parser.add_argument("--num", type=int, help='file num')
params = parser.parse_args()

outfile = params.file
num = params.num
i=0
f = open(outfile,'w')
while i < num:
    with open(outfile+'_'+str(i),'r') as infile:
        for line in infile:

            print(line.rstrip(),file=f)
    i += 1

f.close()
id = outfile[-2:]
ano_file = "../exp_data/en_wiki_align_"+id+".sen.permute.filt.tok"
len1 = len(open(outfile,'r').readlines())
len2 = len(open(ano_file, 'r').readlines())

assert len1 == len2
print("success!")