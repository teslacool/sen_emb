import argparse
import subprocess
subprocess.Popen('export PYTHONPATH=/home/v-jinhzh/code/sen_emb:$PYTHONPATH',shell=True).wait()
from dictionary import Dictionary
import io
import numpy as np

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

def sen_is_usable(line,dico:Dictionary):
    usable = False
    words = line.rstrip().split()
    for word in words:
        if word in dico:
            usable = True
    return usable

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang", type=str, default="en", help="src language")
    parser.add_argument("--tgt_lang", type=str, default="de", help="tgt language")
    parser.add_argument("--emb_dim", type=int, default=300, help="embedding dimension")
    parser.add_argument("--src_emb", type=str, default="../data/wiki.multi.en.vec", help="Reload source embeddings")
    parser.add_argument("--tgt_emb", type=str, default="../data/wiki.multi.de.vec", help="Reload target embeddings")
    parser.add_argument("--src_sen_file", type=str, default='../data/europarl-v7.de-en.en')
    parser.add_argument("--tgt_sen_file", type=str, default='../data/europarl-v7.de-en.de')
    parser.add_argument("--train_num",type=int, default=210000)
    parser.add_argument("--test_num", type=int, default=210000)
    parser.add_argument("--vali_num", type=int, default= 10000)

    params = parser.parse_args()
    src_dico = read_txt_embeddings(params.src_lang,params.src_emb,params.emb_dim)
    tgt_dico = read_txt_embeddings(params.tgt_lang,params.tgt_emb,params.emb_dim)
    i=0
    src_f = io.open(params.src_sen_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    tgt_f = io.open(params.tgt_sen_file, 'r', encoding='utf-8', newline='\n', errors='ignore')

    src_f_train = io.open(params.src_sen_file+'.train', 'w', encoding='utf-8', newline='\n', errors='ignore')
    tgt_f_train = io.open(params.tgt_sen_file+'.train', 'w', encoding='utf-8', newline='\n', errors='ignore')
    src_f_test = io.open(params.src_sen_file+'.test', 'w', encoding='utf-8', newline='\n', errors='ignore')
    tgt_f_test = io.open(params.tgt_sen_file+'.test', 'w', encoding='utf-8', newline='\n', errors='ignore')
    src_f_vali = io.open(params.src_sen_file+'.valid', 'w', encoding='utf-8', newline='\n', errors='ignore')
    tgt_f_vali = io.open(params.tgt_sen_file+'.valid', 'w', encoding='utf-8', newline='\n', errors='ignore')

    try:
        for senx,seny in zip(src_f,tgt_f):
            senx = senx.strip()
            seny = seny.strip()
            if i == params.train_num + params.test_num + params.vali_num:
                break
            if (not sen_is_usable(senx,src_dico)) or (not sen_is_usable(seny,tgt_dico)):
                continue
            if i < params.train_num:
                print(senx,file=src_f_train)
                print(seny,file=tgt_f_train)
            elif i< params.train_num + params.test_num:
                print(senx,file=src_f_test)
                print(seny,file=tgt_f_test)
            else:
                print(senx,file=src_f_vali)
                print(seny,file=tgt_f_vali)
            i += 1
    finally:
        src_f.close()
        tgt_f.close()
        src_f_train.close()
        src_f_test.close()
        src_f_vali.close()
        tgt_f_test.close()
        tgt_f_train.close()
        tgt_f_vali.close()












if __name__ == '__main__':
    main()