
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file",type=str, help='file to split')
params = parser.parse_args()

infile = params.file
f = open(infile, 'r')

line = f.readline()
cnt=0
while True:
    if line == '':
        break

    with open(infile+str(cnt),'w') as out:
        i = 0
        while i < 2000000:
            print(line.rstrip(),file=out)
            i += 1
            line = f.readline()
            if line == '':
                break
    cnt += 1

f.close()