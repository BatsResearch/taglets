import logging
import math

import torch


log = logging.getLogger(__name__)


def rank_biased_overlap(seen_distribution, unseen_distribution):
    """ Takes two matrices of prob distribution over the seen classes.
    
    seen_distribution: matrix with probability computed using seen prompt
    unseen_distribution: matrix with probability computed using unseen prompt
    """

    # log.info(f"Size of distributions: {seen_distribution.size()} and {unseen_distribution.size()}")
    # should change to https://pytorch.org/docs/master/generated/torch.vmap.html#torch.vmap
    
    sort_seen = torch.argsort(seen_distribution, descending=True)
    # log.info(f"seen: {sort_seen}")
    sort_unseen = torch.argsort(unseen_distribution, descending=True)
    # log.info(f"seen: {sort_unseen}")
    #log.info(f"seen: {sort_unseen==sort_seen}")
    log.info(f"loss - Features seen: {sort_seen.requires_grad}")
    log.info(f"loss - Features unseen: {sort_unseen.requires_grad}")

    rank_list = []
    for i in range(sort_seen.size()[0]):
        l_S = sort_seen[i]
        l_U = sort_unseen[i]
        rank_list.append(1 - i_rank_biased_overlap(l_S, l_U, p=0.9))

    return torch.sum(torch.tensor(rank_list))

def spearmanr(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()



def i_rank_biased_overlap(S, T, p=0.9):
    """ Takes two lists S and T of any lengths and gives out the RBO Score
    Parameters
    ----------
    S, T : Lists (str, integers)
    p : Weight parameter, giving the influence of the first d
        elements on the final score. p<0<1. Default 0.9 give the top 10 
        elements 86% of the contribution in the final score.
    
    Returns
    -------
    Float of RBO score
    """
    
    # Fixed Terms
    k = torch.max(torch.tensor([S.size(), T.size()]))
    #print(f"k: {k}")
    x_k = (S[(S.view(1, -1) == T.view(-1, 1)).any(dim=0)]).size()[0]
    #x_k = len(set(S).intersection(set(T)))
    # log.info(f"x_k: {x_k}")
    
    summation_term = 0

    # Loop for summation
    # k+1 for the loop to reach the last element (at k) in the bigger list    
    for d in range (1, k+1): 
        # Create sets from the lists
        set1 = S[:d] if d < S.size()[0] else S
        set2 = T[:d] if d < T.size()[0] else T

        # Intersection at depth d
        #x_d = len(set1.intersection(set2))
        x_d = (set1[(set1.view(1, -1) == set2.view(-1, 1)).any(dim=0)]).size()[0]
        # log.info(f"x_d: {x_d}")
        # Agreement at depth d
        a_d = x_d/d  
        # log.info(f"a_d: {a_d}")

        # Summation
        summation_term = summation_term + math.pow(p, d) * a_d

    # Rank Biased Overlap - extrapolated
    rbo_ext = (x_k/k) * math.pow(p, k) + ((1-p)/p * summation_term)

    return rbo_ext