

from __future__ import absolute_import, division, unicode_literals
import io
import torch
from dictionary import Dictionary
from dictionary import SenDictionary
import sys
import numpy as np
import re
import inspect
from torch import optim

def init_dic(params):
    src_dico = load_embeddings(params, source=True)
    params.src_dico = src_dico
    tgt_dico = load_embeddings(params, source=False)
    params.tgt_dico = tgt_dico

    train_sen_dico,train_y = read_sentence_embeddings(params,params.src_dico, params.train_sen_path,source=True)
    params.train_sen_dico = train_sen_dico
    #TODO
    dev_sen_dico,  dev_y = read_sentence_embeddings(params,params.tgt_dico, params.dev_sen_path,source=False)
    params.dev_sen_dico = dev_sen_dico

    test_sen_dico, test_y = read_sentence_embeddings(params, params.src_dico, params.test_sen_path,source=True)
    params.test_sen_dico = test_sen_dico

    params.train_y = train_y
    params.dev_y = dev_y
    params.test_y = test_y


def init_dict(params):
    src_dico = load_embeddings(params, source=True)
    params.src_dico = src_dico
    tgt_dico = load_embeddings(params, source=False)
    params.tgt_dico = tgt_dico

    train_pos = read_sentence_embeddings2(params, params.src_dico,params.src_pos_sen_path, source=True)
    params.train_pos = train_pos
    train_neg = read_sentence_embeddings2(params, params.src_dico,params.src_neg_sen_path, source=True)
    params.train_neg = train_neg
    valid_pos = read_sentence_embeddings2(params, params.tgt_dico,params.src_pos_sen_path+'.'+params.tgt_lang + '.tok', source=False)
    params.valid_pos = valid_pos
    valid_neg = read_sentence_embeddings2(params, params.tgt_dico,params.src_neg_sen_path+'.'+params.tgt_lang + '.tok', source=False)
    params.valid_neg = valid_neg



def load_embeddings(params, source, full_vocab=True):

    assert type(source) is bool and type(full_vocab) is bool

    return read_txt_embeddings(params, source, full_vocab)



def read_txt_embeddings(params, source, full_vocab):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []

    # load pretrained embeddings
    lang = params.src_lang if source else params.tgt_lang
    emb_path = params.src_emb if source else params.tgt_emb
    _emb_dim_file = params.emb_dim
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
                                       % (word, 'source' if source else 'target'))
                else:
                    if not vect.shape == (_emb_dim_file,):
                        print("Invalid dimension (%i) for %s word '%s' in line %i."
                                       % (vect.shape[0], 'source' if source else 'target', word, i))
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

def read_sentence_embeddings(params, dico,sen_path,source=True):

    id2sen = {}
    class_y = []


    lang = params.src_lang if source else params.tgt_lang
    i = 0

    with io.open(sen_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for line in f:
            sen = line.strip()
            cla = sen[-1]
            assert cla.isdigit(),'must be a digit'
            sen = sen[:-1]
            sen =  sen.lower()
            sen2words = []
            words = sen.strip().split()
            for word in words:
                if word not in dico:
                    continue
                sen2words.append(dico.word2id[word])
            if len(sen2words)==0:
                print("the %dth sentence is null in %s file"%(i+1,sen_path))
                continue
            id2sen[i] = sen2words
            class_y.append([int(cla)])
            i += 1

    print("Loaded %i sentence." % len(id2sen))


    dico = SenDictionary(id2sen,lang,dico)
    class_y = np.concatenate(class_y,0)
    return dico,class_y


def read_sentence_embeddings2(params, dico,sen_path,source=True):

    id2sen = {}

    # load pretrained embeddings
    lang = params.src_lang if source else params.tgt_lang
    with io.open(sen_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            sen = line.rstrip()
            sen =  sen.lower()
            sen2words = []
            words = sen.strip().split()
            for word in words:
                if word not in dico:
                    continue
                sen2words.append(dico.word2id[word])
            if len(sen2words)==0:
                print("the %dth sentence is null in %s file"%(i+1,sen_path))
                sen2words.append(dico.word2id['.'])
            id2sen[i] = sen2words

    print("Loaded %i sentence." % len(id2sen))


    dico = SenDictionary(id2sen,lang,params.src_dico if source else params.tgt_dico)
    return dico



def create_dictionary(sentences):
    words = {}
    for s in sentences:
        for word in s:
            if word in words:
                words[word] += 1
            else:
                words[word] = 1
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2
    # words['<UNK>'] = 1e9 + 1
    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params

