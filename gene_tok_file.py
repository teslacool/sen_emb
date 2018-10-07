import os
import io
path = './enes_data'
files = os.listdir(path)
prefix = '../../mosesdecoder/scripts/tokenizer/tokenizer.perl -l '
middle = ' -threads 24 < '
last = ' > '
with io.open(os.path.join(path,'tok.sh'),'w',newline='\n',encoding='utf8') as tgt:
    for file in files:
        if file.endswith('filt'):
            lang = file[:2]
            tgt_file = file+'.tok'
            command = prefix + lang + middle + file + last + tgt_file
            tgt.write(command+'\n')

