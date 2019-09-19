import torch
from torch import nn
from torchvision import models
from .BasicModule import BasicModule
import scipy.sparse as sp
import torch.nn.functional as F

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
        
class ResNet50(BasicModule):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.model_name = 'ResNet50'
        self.base = models.resnet50(pretrained=True)
        self.base.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # self.base.avgpool = nn.AdaptiveMaxPool2d((1,1))
        self.num_classes = num_classes
        self.num_ftrs = self.base.fc.in_features
        remove_block = []
        remove_block = nn.Sequential(*remove_block)
        self.base.fc = remove_block
        
        self.bottleneck = nn.BatchNorm1d(self.num_ftrs)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.feat_fc = nn.Linear(self.num_ftrs, 2048, bias=False)
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.feat_fc.apply(weights_init_classifier)

        self.A = None
        self.A_hat = None

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, indices=None, get_global=False):
        global_feat = self.base(x)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        if get_global: return global_feat
        # if self.training:
        feat = self.bottleneck(global_feat)  # normalize for angular softmax
        # feat_out = self.feat_fc(feat)
        # else:
        #     feat = global_feat
        if self.training:
            # new_normed_adj = self.A_hat[indices, :][:, indices]

            # feat = torch.spmm(new_normed_adj, feat)
            feat = self.dropout(feat)
            cls_score = self.classifier(feat)

            # cls_score = torch.spmm(new_normed_adj, cls_score)
            # feat = torch.spmm(new_normed_adj, feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            return feat

    def update_A(self, A):
        self.A = A - torch.eye(A.shape[0])
        self.A_hat = normalize(A) # + torch.eye(A.shape[0])

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = mx.sum(1)
    r_inv = torch.pow(rowsum, -1).flatten()
    return mx * r_inv