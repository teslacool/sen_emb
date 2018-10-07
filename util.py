import numpy as np
import io
import torch
from dictionary import Dictionary
from dictionary import SenDictionary
import sys
def init_dic(params):
    src_dico = load_embeddings(params, source=True)
    params.src_dico = src_dico
    tgt_dico = load_embeddings(params, source=False)
    params.tgt_dico = tgt_dico
    # src_sen_dico = read_sentence_embeddings(params, source=True, test=False)
    # params.src_sen_dico = src_sen_dico
    # tgt_sen_dico = read_sentence_embeddings(params, source=False, test=False)
    # params.tgt_sen_dico = tgt_sen_dico
    src_sen_dico_test = read_sentence_embeddings(params, source=True, test=True)
    params.src_sen_dico_test = src_sen_dico_test
    tgt_sen_dico_test = read_sentence_embeddings(params, source=False, test=True)
    params.tgt_sen_dico_test = tgt_sen_dico_test




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

def read_sentence_embeddings(params, source,test=False):

    id2sen = {}

    # load pretrained embeddings
    lang = params.src_lang if source else params.tgt_lang
    if not test:
        sen_path = params.src_sen_path if source else params.tgt_sen_path
    else:
        sen_path = params.src_sen_path_test if source else params.tgt_sen_path_test
    dico = params.src_dico if source else params.tgt_dico
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
                sen2words.append(0)
                print('sen2words.append(0)')
            id2sen[i] = sen2words

    print("Loaded %i sentence." % len(id2sen))


    dico = SenDictionary(id2sen,lang,params.src_dico if source else params.tgt_dico)
    return dico






from sklearn.decomposition import TruncatedSVD
def remove_pc(X, npc=1):
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    pc = svd.components_
    if npc == 1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX

def remove_all_pc(x,y,npc):
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    X=np.concatenate((x,y),0)
    svd.fit(X)
    pc = svd.components_
    if npc==1:
        x = x-x.dot(pc.transpose())*pc
        y = y-y.dot(pc.transpose())*pc
    else:
        x = x - x.dot(pc.transpose()).dot(pc)
        y = y - y.dot(pc.transpose()).dot(pc)
    return x,y

def cal_topk_csls(src_emb, tgt_emb):

    src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
    tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

    bs = 128
    n_src = src_emb.size(0)
    top1_acc = 0.
    for knn in [1,5,10]:


        average_dist1 = torch.from_numpy(get_nn_avg_dist(tgt_emb, src_emb, 10))
        average_dist2 = torch.from_numpy(get_nn_avg_dist(src_emb, tgt_emb, 10))
        average_dist1 = average_dist1.type_as(src_emb)
        average_dist2 = average_dist2.type_as(tgt_emb)

        all_targets = []
        for i in range(0, n_src, bs):

            # compute target words scores
            scores = tgt_emb.mm(src_emb[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
            scores.mul_(2)
            scores.sub_(average_dist1[i:min(n_src, i + bs)][:, None] + average_dist2[None, :])
            _, best_targets = scores.topk(knn, dim=1, largest=True, sorted=True)



            # update scores / potential targets
            all_targets.append(best_targets.cpu())

        all_targets = torch.cat(all_targets, 0)
        equ_out  = torch.eq(all_targets,torch.from_numpy(np.array(range(n_src)))[:,None])
        topk = torch.sum(equ_out)
        out = topk.numpy().reshape((1))[0]
        print(out/float(n_src))
        if knn ==1:
            top1_acc = out/float(n_src)
    return top1_acc

def new_cal_topk_csls(src_emb, tgt_emb, params,src_num,tgt_num,direction):


    if not direction:
        temp = src_emb
        src_emb = tgt_emb
        tgt_emb = temp

    src_sen_file = params.src_sen_path_test
    tgt_sen_file = params.tgt_sen_path_test
    assert len(open(src_sen_file,'r', encoding='utf-8').readlines()) == src_emb.size(0)
    assert len(open(src_sen_file,'r', encoding='utf-8').readlines()) == len(open(tgt_sen_file,'r', encoding='utf-8').readlines())
    data = {'src':[],'tgt':[]}
    fname = src_sen_file
    with io.open(fname, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= tgt_num*1.5:
                break
            line = line.lower()
            data['src'].append(line.rstrip().split())
    fname = tgt_sen_file
    with io.open(fname, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= tgt_num * 1.5:
                break
            line = line.lower()
            data['tgt'].append(line.rstrip().split())

    assert  len(data['src'])==len(data['tgt'])
    data['src'] = np.array(data['src'])
    data['tgt'] = np.array(data['tgt'])
    data['src'], indices = np.unique(data['src'],return_index=True)
    src_emb = src_emb[indices]
    tgt_emb = tgt_emb[indices]
    data['tgt'] = data['tgt'][indices]
    data['tgt'] , indices = np.unique(data['tgt'],return_index=True)
    data['src'] = data['src'][indices]
    src_emb = src_emb[indices]
    tgt_emb = tgt_emb[indices]

    # assert tgt_emb.size(0) >= tgt_num
    tgt_num = tgt_num if tgt_emb.size(0) >= tgt_num else tgt_emb.size(0)
    src_num = src_num if src_emb.size(0) >= src_num else src_emb.size(0)
    assert src_num <= tgt_num
    print("src_num: %d   tgt_num: %d"%(src_num,tgt_num))

    #shuffle
    #now the same size
    rng = np.random.RandomState(1234)
    perm = rng.permutation(len(data['src']))
    data['src'] = data['src'][perm]
    data['tgt'] = data['tgt'][perm]
    #different size
    tgt_emb = tgt_emb[:tgt_num]
    rng = np.random.RandomState(1234)
    idx_query = rng.choice(range(tgt_num), size=src_num, replace=False)
    src_emb =src_emb[idx_query]

    src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
    tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
    bs = 128
    n_src = src_emb.size(0)
    top1_acc = 0.

    average_dist1 = torch.from_numpy(get_nn_avg_dist(tgt_emb, src_emb, 10))
    average_dist2 = torch.from_numpy(get_nn_avg_dist(src_emb, tgt_emb, 10))
    average_dist1 = average_dist1.type_as(src_emb)
    average_dist2 = average_dist2.type_as(tgt_emb)

    all_targets = []
    for i in range(0, n_src, bs):

        # compute target words scores
        scores = tgt_emb.mm(src_emb[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
        scores.mul_(2)
        scores.sub_(average_dist1[i:min(n_src, i + bs)][:, None] + average_dist2[None, :])
        _, best_targets = scores.topk(10, dim=1, largest=True, sorted=True)



        # update scores / potential targets
        all_targets.append(best_targets.cpu())

    all_targets = torch.cat(all_targets, 0)
    for knn in [1, 5, 10]:
        equ_out  = torch.eq(all_targets[:,:knn],torch.from_numpy(idx_query)[:,None])
        topk = torch.sum(equ_out)
        out = topk.numpy().reshape((1))[0]
        print(out/float(idx_query.shape[0]))
        if knn ==1:
            top1_acc = out/float(n_src)
    return top1_acc



def select_pair(src_emb, tgt_emb):

    src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
    tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
    bs = 128
    n_src = src_emb.size(0)
    top1_acc = 0.

    average_dist1 = torch.from_numpy(get_nn_avg_dist(tgt_emb, src_emb, 10))
    average_dist2 = torch.from_numpy(get_nn_avg_dist(src_emb, tgt_emb, 10))
    average_dist1 = average_dist1.type_as(src_emb).cuda()
    average_dist2 = average_dist2.type_as(tgt_emb).cuda()
    src_emb = src_emb.cuda()
    tgt_emb = tgt_emb.cuda()


    all_targets = []
    all_scores = []
    for i in range(0, n_src, bs):

        # compute target words scores
        scores = tgt_emb.mm(src_emb[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
        scores.mul_(2)
        scores.sub_(average_dist1[i:min(n_src, i + bs)][:, None] + average_dist2[None, :])
        best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)



        # update scores / potential targets
        all_targets.append(best_targets.cpu())
        all_scores.append(best_scores.cpu())

    all_targets = torch.cat(all_targets, 0)
    all_scores = torch.cat(all_scores,0)

    all_pairs = torch.cat([
        torch.arange(0, all_targets.size(0)).long().unsqueeze(1),
        all_targets[:,0].unsqueeze(1)
    ], 1 )

    diff = all_scores[:, 0] - all_scores[:, 1]
    reordered = diff.sort(0, descending=True)[1]
    all_scores = all_scores[reordered]
    all_pairs = all_pairs[reordered]

    threshold = 0.01 # for fre_select.de.0
    diff = all_scores[:, 0] - all_scores[:, 1]
    mask = diff > threshold
    print("Selected %i / %i pairs above the confidence threshold." % (mask.sum(), diff.size(0)))
    mask = mask.unsqueeze(1).expand_as(all_pairs).clone()
    all_pairs = all_pairs.masked_select(mask).view(-1, 2)
    all_scores = all_scores.masked_select(mask).view(-1,2)
    mask  = all_scores[:,0] > 0.
    mask = mask.unsqueeze(1).expand_as(all_pairs).clone()
    all_pairs = all_pairs.masked_select(mask).view(-1, 2)
    torch.save(all_pairs,'exp_data/pair_from_')



def get_score(src_emb, tgt_emb,score_file):

    src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
    tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
    assert src_emb.size()[0] == tgt_emb.size()[0]
    # average_dist1 = torch.from_numpy(get_nn_avg_dist(tgt_emb, src_emb, 10))
    # average_dist2 = torch.from_numpy(get_nn_avg_dist(src_emb, tgt_emb, 10))
    # average_dist1 = average_dist1.type_as(src_emb)
    # average_dist2 = average_dist2.type_as(tgt_emb)
    n_src = src_emb.size(0)
    src_emb = src_emb.view(n_src, 1, -1)
    tgt_emb = tgt_emb.view(n_src, 1, -1)
    score = torch.bmm(src_emb, tgt_emb.transpose(1, 2)).squeeze()
    # score = 2 * score-average_dist1-average_dist2
    score = score.detach().cpu().numpy().tolist()
    with open(score_file, 'w') as f:
        for s in score:
            print(s, file=f)




def cal_topk_eu(src_emb, tgt_emb):
    # src_emb = src_emb.weight.data
    # tgt_emb = tgt_emb.weight.data
    src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
    tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
    n_src = src_emb.size(0)
    src_norm = src_emb.norm(2,1).expand(n_src,n_src)
    tgt_norm = tgt_emb.norm(2,1).expand(n_src,n_src)
    mm = src_emb.mm(tgt_emb.transpose(0,1))
    dis= src_norm*src_norm+(tgt_norm*tgt_norm).transpose(0,1)-2*mm



    for knn in [1,5,10]:

        _, best_targets = dis.topk(knn, dim=1, largest=False, sorted=True)

        equ_out  = torch.eq(best_targets.cpu(),torch.from_numpy(np.array(range(n_src)))[:,None])
        topk = torch.sum(equ_out)
        out = topk.numpy().reshape((1))[0]
        print(out/float(n_src))




try:
    import faiss
    FAISS_AVAILABLE = True
    if not hasattr(faiss, 'StandardGpuResources'):
        sys.stderr.write("Impossible to import Faiss-GPU. "
                         "Switching to FAISS-CPU, "
                         "this will be slower.\n\n")

except ImportError:
    sys.stderr.write("Impossible to import Faiss library!! "
                     "Switching to standard nearest neighbors search implementation, "
                     "this will be significantly slower.\n\n")
    FAISS_AVAILABLE = False

def get_nn_avg_dist(emb, query, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    if FAISS_AVAILABLE:
        emb = emb.cpu().numpy()
        query = query.cpu().numpy()
        # if hasattr(faiss, 'StandardGpuResources'):
        if False:
            # gpu mode
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)
        else:
            # cpu mode
            index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        distances, _ = index.search(query, knn)
        return distances.mean(1)
    # if FAISS_AVAILABLE:
    #     emb_all = emb.cpu().numpy()
    #     query_all = query.cpu().numpy()
    #     batch = 2000
    #
    #     mean_distance = []
    #     for i in range(0,emb_all.shape[0],batch):
    #         query = query_all[i:min(i + batch, emb_all.shape[0])]
    #         if i + batch < emb_all.shape[0]:
    #             emb = emb_all[i:i+batch]
    #         else:
    #             emb = emb_all[-batch:]
    #
    #
    #     # if hasattr(faiss, 'StandardGpuResources'):
    #         if False:
    #             # gpu mode
    #             res = faiss.StandardGpuResources()
    #             config = faiss.GpuIndexFlatConfig()
    #             config.device = 0
    #             index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)
    #         else:
    #             # cpu mode
    #             index = faiss.IndexFlatIP(emb.shape[1])
    #         index.add(emb)
    #         distances, _ = index.search(query, knn)
    #         mean_distance.append(distances.mean(1))
    #     return np.concatenate(mean_distance,0)
    else:
        bs = 1024
        all_distances = []
        emb = emb.transpose(0, 1).contiguous()
        for i in range(0, query.shape[0], bs):
            distances = query[i:i + bs].mm(emb)
            best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
            all_distances.append(best_distances.mean(1).cpu())
        all_distances = torch.cat(all_distances)
        return all_distances.numpy()




def procrustes(src_emb, tgt_emb,src_emb_,tgt_emb_, mapping):
    """
    Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
    https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    """
    A = src_emb_.weight.data
    B = tgt_emb_.weight.data
    W = mapping.weight.data
    M = B.transpose(0, 1).mm(A).cpu().numpy()
    U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
    W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))
    src_emb.weight.data.copy_(mapping(src_emb.weight).data)

