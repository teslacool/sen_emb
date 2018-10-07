import io
import subprocess
import numpy as np
subprocess.Popen('export PYTHONPATH=/home/v-jinhzh/code/sen_emb:$PYTHONPATH',shell=True).wait()
from dictionary import Dictionary

def sen_is_lang(sen,dico_src:Dictionary,dico_tgt:Dictionary):
    sen = sen.strip()
    words = sen.split()
    src = 0
    tgt = 0
    for word in words:
        if word in dico_src:
            src += 1
        if word in dico_tgt:
            tgt += 1
    if src >= tgt:
        return 0
    else:
        return 1

def read_txt_embeddings(lang, emb_path, emb_dim):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []
    full_vocab = True

    # load pretrained embeddings


    _emb_dim_file = emb_dim
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
                assert _emb_dim_file == int(split[1])
            else:
                word, vect = line.rstrip().split(' ', 1)
                if not full_vocab:
                    word = word.lower()
                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                if word in word2id:
                    if full_vocab:
                        print("Word '%s' found twice in %s embedding file"
                                       % (word, lang))
                else:
                    if not vect.shape == (_emb_dim_file,):
                        print("Invalid dimension (%i) for %s word '%s' in line %i."
                                       % (vect.shape[0], lang, word, i))
                        continue
                    assert vect.shape == (_emb_dim_file,), i
                    word2id[word] = len(word2id)
                    vectors.append(vect[None])


    assert len(word2id) == len(vectors)
    print("Loaded %i pre-trained word embeddings." % len(vectors))

    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    word2vec = {id2word[k]:vectors[k] for k in range(len(id2word))}
    dico = Dictionary(id2word, word2id,word2vec ,lang)

    return dico

src_path = '/home/v-jinhzh/code/sen_emb/data/STS2016-cross-lingual-test/STS.input.news.txt'
tgt_en = '/home/v-jinhzh/code/sen_emb/data/STS2016-cross-lingual-test/tgt_en'
tgt_es = '/home/v-jinhzh/code/sen_emb/data/STS2016-cross-lingual-test/tgt_es'

dico_en = read_txt_embeddings('en','../data/wiki.multi.en.vec',300)
dico_es = read_txt_embeddings('es','../data/wiki.multi.es.vec',300)
with io.open(src_path,'r',encoding='utf8',errors='ignore',newline='\n') as f0:
    with io.open(tgt_en, 'w', encoding='utf8', errors='ignore', newline='\n') as f_en:
        with io.open(tgt_es, 'w', encoding='utf8', errors='ignore', newline='\n') as f_es:
            for line in f0:
                a = line.strip().split('\t')
                assert len(a) == 2
                index_left = sen_is_lang(a[0],dico_en,dico_es)
                index_right = sen_is_lang(a[1],dico_en,dico_es)
                if index_left == index_right:
                    print(a[0])
                print(a[index_left].strip(),file=f_en)
                print(a[index_right].strip(),file=f_es)