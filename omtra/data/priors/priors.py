from torch.distributions import Exponential
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn.functional import softmax, one_hot
import dgl

from omtra.data.priors.align import align_prior

def gaussian(x1: torch.Tensor, std: float = 1.0, ot=False):
    """
    Generate a prior feature by sampling from a Gaussian distribution.
    """

    n, d = x1.shape

    x0 = torch.randn(n, d) * std
    
    if ot:
        # move x0 to the same COM as x1
        x0_mean = x0.mean(dim=0, keepdim=True)
        x1_mean = x1.mean(dim=0, keepdim=True)
        x0 += x1_mean - x0_mean

        # align x0 to x1
        x0 = align_prior(x0, x1, rigid_body=True, permutation=True)

    return x0


def centered_normal_prior(x1: torch.Tensor, std: float = 1.0):
    """
    Generate a prior feature by sampling from a centered normal distribution.
    """
    prior_feat = torch.randn(n, d) * std
    prior_feat = prior_feat - prior_feat.mean(dim=0, keepdim=True)
    return prior_feat

def centered_normal_prior_batched_graph(g: dgl.DGLGraph, node_batch_idx: torch.Tensor, std: float = 1.0):
    raise NotImplementedError
    # TODO: implement this for a heterogeneous graph
    n = g.num_nodes()
    prior_sample = torch.randn(n, 3, device=g.device)
    with g.local_scope():
        g.ndata['prior_sample'] = prior_sample
        prior_sample = prior_sample - dgl.readout_nodes(g, feat='prior_sample', op='mean')[node_batch_idx]

    return prior_sample
    

def ctmc_masked_prior(n: int, d: int):
    """
    Sample from a CTMC masked prior. All samples are assigned the mask token at t=0.
    """
    p = torch.full((n,), fill_value=d)
    p = one_hot(p, num_classes=d+1).float()
    return p


