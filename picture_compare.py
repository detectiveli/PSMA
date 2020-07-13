import fire
import os
import time
import torch
import numpy as np
import models
from config import cfg
from data_loader import data_loader
from data_loader.data_loader import read_image
from utils import check_jupyter_run

from data_loader.transforms import transforms
def load_one_image(img_path):
    val_transforms = transforms(cfg, is_train=False)
    img = read_image(img_path)
    img = val_transforms(img)
    return img.unsqueeze(0)

def test(config_file, **kwargs):
    np.set_printoptions(suppress=True)
    cfg.merge_from_file(config_file)
    if kwargs:
        opts = []
        for k,v in kwargs.items():
            opts.append(k)
            opts.append(v)
        cfg.merge_from_list(opts)
    cfg.freeze()
    
    device = torch.device(cfg.DEVICE)
    _, val_loader, num_query, num_classes = data_loader(cfg, cfg.DATASETS.NAMES)
    model = getattr(models, cfg.MODEL.NAME)(num_classes)
    model.load(cfg.OUTPUT_DIR,80)
    if device:
        model.to(device) 
    model = model.eval()
    
    feats = []
    
    since = time.time()
    files = os.listdir('./test_picture')
    files.sort()
    print(files)
    for file in files:  # files
    # for data in tqdm(val_loader, desc='Feature Extraction', leave=False):
        with torch.no_grad():
            images = load_one_image('./test_picture/' + file)
            if device:
                model.to(device)
                images = images.to(device)
            feat = model(images)

        feats.append(feat)
    all_feats = torch.cat(feats, dim=0)
    n = all_feats.shape[0]
    distmat = torch.pow(all_feats, 2).sum(dim=1, keepdim=True).expand(n, n) + \
              torch.pow(all_feats, 2).sum(dim=1, keepdim=True).expand(n, n).t()
    distmat.addmm_(1, -2, all_feats, all_feats.t())
    print(distmat.cpu().numpy())
    # cmc, mAP = evaluation(all_feats,all_pids,all_camids,num_query,re_ranking)
    
if __name__=='__main__':
    fire.Fire(test)