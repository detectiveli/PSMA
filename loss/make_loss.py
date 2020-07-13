# encoding: utf-8


import torch.nn.functional as F

from .triplet_loss import TripletLoss

def make_loss(cfg):
    sampler = cfg.DATALOADER.SAMPLER
    triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss

    if sampler == 'softmax':
        def loss_func(score, feat, target, index):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target, index):
            return triplet(feat[index], target[index])[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, index=None):
            return F.cross_entropy(score, target) + triplet(feat, target)[0]#
    elif cfg.DATALOADER.SAMPLER == 'softmax_Htriplet':
        def loss_func(score, feat, target, index=None):
            return F.cross_entropy(score, target) + triplet(feat, target, HSoft=True)[0]#

    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func
