import os
from logging import getLogger
import scipy
import scipy.linalg
import torch
import sys
import numpy as np

# load Faiss if available (dramatically accelerates the nearest neighbor search)
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

def cal_topk_csls(src_emb, tgt_emb, params):

    src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
    tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

    bs = 128
    n_src = src_emb.size(0)
    top1 = 0
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
        if knn==1:
            top1 = out/float(n_src)
    return top1


def get_score(src_emb, tgt_emb, score_file):
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
    # score = 2 * score - average_dist1 - average_dist2
    score = score.detach().cpu().numpy().tolist()
    with open(score_file, 'w') as f:
        for s in score:
            print(s, file=f)

def cal_topk_eu(src_emb, tgt_emb, params):
    src_emb = src_emb.weight.data
    tgt_emb = tgt_emb.weight.data
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


def cal_topk_cos(src_emb, tgt_emb, params):
    src_emb = src_emb.weight.data
    tgt_emb = tgt_emb.weight.data
    src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
    tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

    bs = 128
    n_src = src_emb.size(0)
    for knn in [1,5,10]:




        all_targets = []
        for i in range(0, n_src, bs):

            # compute target words scores
            scores = tgt_emb.mm(src_emb[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
            all_targets.append(scores.cpu())

        all_targets = torch.cat(all_targets, 0)
        score = all_targets.numpy()
        arg = np.argsort(score,1)
        argtop=arg[:,-knn:]
        equ_out = torch.eq(torch.from_numpy(argtop), torch.from_numpy(np.array(range(n_src)))[:, None])
        topk = torch.sum(equ_out)
        out = topk.numpy().reshape((1))[0]
        print(out/float(n_src))

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
def new_cal_topk_csls(src_emb, tgt_emb, src_num,tgt_num,direction):


    if not direction:
        temp = src_emb
        src_emb = tgt_emb
        tgt_emb = temp

    # src_sen_file = params.src_sen_path_test
    # tgt_sen_file = params.tgt_sen_path_test
    # assert len(open(src_sen_file,'r', encoding='utf-8').readlines()) == src_emb.size(0)
    # assert len(open(src_sen_file,'r', encoding='utf-8').readlines()) == len(open(tgt_sen_file,'r', encoding='utf-8').readlines())
    # data = {'src':[],'tgt':[]}
    # fname = src_sen_file
    # with io.open(fname, 'r', encoding='utf-8') as f:
    #     for i, line in enumerate(f):
    #         if i >= tgt_num*1.5:
    #             break
    #         line = line.lower()
    #         data['src'].append(line.rstrip().split())
    # fname = tgt_sen_file
    # with io.open(fname, 'r', encoding='utf-8') as f:
    #     for i, line in enumerate(f):
    #         if i >= tgt_num * 1.5:
    #             break
    #         line = line.lower()
    #         data['tgt'].append(line.rstrip().split())
    #
    # assert  len(data['src'])==len(data['tgt'])
    # data['src'] = np.array(data['src'])
    # data['tgt'] = np.array(data['tgt'])
    # data['src'], indices = np.unique(data['src'],return_index=True)
    # src_emb = src_emb[indices]
    # tgt_emb = tgt_emb[indices]
    # data['tgt'] = data['tgt'][indices]
    # data['tgt'] , indices = np.unique(data['tgt'],return_index=True)
    # data['src'] = data['src'][indices]
    # src_emb = src_emb[indices]
    # tgt_emb = tgt_emb[indices]
    #
    # # assert tgt_emb.size(0) >= tgt_num
    # tgt_num = tgt_num if tgt_emb.size(0) >= tgt_num else tgt_emb.size(0)
    # src_num = src_num if src_emb.size(0) >= src_num else src_emb.size(0)
    assert src_num <= tgt_num
    print("src_num: %d   tgt_num: %d"%(src_num,tgt_num))

    #shuffle
    #now the same size
    # rng = np.random.RandomState(1234)
    # perm = rng.permutation(len(data['src']))
    # data['src'] = data['src'][perm]
    # data['tgt'] = data['tgt'][perm]
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
