from googletrans import Translator
import io
import time
src_file = '/home/v-jinhzh/code/SentEval/data/downstream/SST/binary/sentiment-dev'
tgt_file = src_file+'_de'

trans = Translator()



with io.open(tgt_file,'w',encoding='utf8',errors='ignore', newline='\n') as tgt:
    pass
i=0
with io.open(src_file,'r',encoding='utf8',errors='ignore', newline='\n') as src:
    with io.open(tgt_file, 'w', encoding='utf8', errors='ignore', newline='\n') as tgt:
        for line in src:
            line = line.strip()
            cla = line[-1]
            sen = line[:-1].strip()
            assert  cla.isdigit(),"not a digit number"
            result = trans.translate(sen,dest='de',src='en')
            print(result.text+' '+cla,file=tgt)
            i += 1
            print(i)
            time.sleep(1)
