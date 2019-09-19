# encoding: utf-8
import torch
import numpy as np
from .re_ranking import re_ranking
from utils import check_jupyter_run
if check_jupyter_run():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

def evaluation(all_feats,all_pids,all_camids,num_query,rr=False, max_rank=50):
    all_feats = torch.cat(all_feats, dim=0)
    # query
    qf = all_feats[:num_query]
    q_pids = np.asarray(all_pids[:num_query])
    q_camids = np.asarray(all_camids[:num_query])
    # gallery
    gf = all_feats[num_query:]
    g_pids = np.asarray(all_pids[num_query:])
    g_camids = np.asarray(all_camids[num_query:])
    
    if not rr:    
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
    else:
        print('Calculating Distance')
        q_g_dist = np.dot(qf.data.cpu(), np.transpose(gf.data.cpu()))
        q_q_dist = np.dot(qf.data.cpu(), np.transpose(qf.data.cpu()))
        g_g_dist = np.dot(gf.data.cpu(), np.transpose(gf.data.cpu()))
        print('Re-ranking:')
        distmat= re_ranking(q_g_dist, q_q_dist, g_g_dist)
        
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in tqdm(range(num_q), desc='Metric Computing', leave=False):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc_base = orig_cmc.cumsum()
        cmc = cmc_base

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        # tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        # tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        tmp_cmc = tmp_cmc / (np.arange(tmp_cmc.size) + 1.0)
        tmp_cmc = tmp_cmc * orig_cmc

        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP
