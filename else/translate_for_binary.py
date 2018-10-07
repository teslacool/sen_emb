from googletrans import Translator
import io
import time
import argparse
translator = Translator()

parser = argparse.ArgumentParser()
parser.add_argument("--src_lang",type=str)
parser.add_argument("--tgt_lang",type=str)
parser.add_argument("--src_pos",type=str)
parser.add_argument("--src_neg",type=str)
# src_lang = 'en'
# tgt_lang = 'fr'
# src_pos = '/home/v-jinhzh/code/SentEval/data/downstream/MR/rt-polarity.pos'
# src_neg = '/home/v-jinhzh/code/SentEval/data/downstream/MR/rt-polarity.neg'

params = parser.parse_args()
src_lang = params.src_lang
print('src_lang: ',src_lang)
tgt_lang = params.tgt_lang
print('tgt_lang: ',tgt_lang)
src_pos = params.src_pos
src_neg = params.src_neg
try:
    for f in [src_pos,src_neg]:
        with io.open(f,'r',encoding='utf8',errors='ignore',newline='\n') as src:
            with io.open(f+'.'+tgt_lang,'w',encoding='utf8',errors='ignore',newline='\n') as tgt:
                for i,line in enumerate(src):
                    line = line.strip()
                    result = translator.translate(line,dest=tgt_lang,src=src_lang)
                    print(result.text, file=tgt)
                    time.sleep(0.4)
                    print("%d  in %s"%(i,f))
except:
    print("fail")

print('success')