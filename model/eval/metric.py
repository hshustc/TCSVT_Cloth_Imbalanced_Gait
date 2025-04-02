
import numpy as np

def compute_CMC_mAP(distmat, q_pids, g_pids, max_rank):
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx]
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        
        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        if num_rel > 0:
            num_valid_q += 1.
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_q
    assert(len(all_AP) == num_valid_q)
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def compute_PR(distmat, q_pids, g_pids, sim_thres=-1):
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_P = []
    all_R = []
    for q_idx in range(num_q):
        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx]

        num_rel = orig_cmc.sum()
        if num_rel > 0:
            num_thres_qualified = len(np.where((1 - distmat[q_idx]) >= sim_thres)[0])
            if num_thres_qualified > 0:
                orig_cmc = orig_cmc[:num_thres_qualified]
                all_P.append(orig_cmc.sum()*1.0/len(orig_cmc))
                all_R.append(orig_cmc.sum()*1.0/num_rel)
            else:
                all_R.append(0)

    if len(all_P) > 0:
        return np.mean(all_P), np.mean(all_R)
    else:
        return -1, np.mean(all_R)