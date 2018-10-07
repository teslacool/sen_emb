import io

src_path = '/home/v-jinhzh/code/sen_emb/data/STS2016-cross-lingual-test/STS.input.multisource.txt'
tgt_en = '/home/v-jinhzh/code/sen_emb/data/STS2016-cross-lingual-test/tgt_en'
tgt_es = '/home/v-jinhzh/code/sen_emb/data/STS2016-cross-lingual-test/tgt_es'

with io.open(src_path,'r',encoding='utf8',errors='ignore',newline='\n') as f0:
    with io.open(tgt_en, 'w', encoding='utf8', errors='ignore', newline='\n') as f_en:
        with io.open(tgt_es, 'w', encoding='utf8', errors='ignore', newline='\n') as f_es:
            for line in f0:
                a = line.strip().split('\t')
                assert len(a) == 4
                print(a[1].strip(),file=f_en)
                print(a[0].strip(),file=f_es)