# encoding: utf-8


import torch.nn.functional as F

from .triplet_loss import TripletLoss
from .OIM_loss import OIM_Module

def make_loss(cfg):
    sampler = cfg.DATALOADER.SAMPLER
    triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    OIM_loss = OIM_Module(751)

    if sampler == 'softmax':
        def loss_func(score, feat, target, index):
            # scores = OIM_loss(feat, target)
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target, index):
            return triplet(feat[index], target[index])[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, index=None):
            return F.cross_entropy(score, target) + triplet(feat, target)[0]#
            # score_OIM = OIM_loss(feat, target)
            # return F.cross_entropy(score_OIM, target) + triplet(feat, target)[0],  score_OIM#
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func
