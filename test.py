import fire
import os
import time
import torch
import numpy as np
from torch.autograd import Variable
import models
from config import cfg
from data_loader import data_loader
from logger import make_logger
from evaluation import evaluation
from datasets import PersonReID_Dataset_Downloader
from utils import check_jupyter_run
if check_jupyter_run():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

    
def test(config_file, **kwargs):
    cfg.merge_from_file(config_file)
    if kwargs:
        opts = []
        for k,v in kwargs.items():
            opts.append(k)
            opts.append(v)
        cfg.merge_from_list(opts)
    cfg.freeze()
    
    re_ranking=cfg.RE_RANKING
    
    PersonReID_Dataset_Downloader('./datasets',cfg.DATASETS.NAMES)
    if not re_ranking:
        logger = make_logger("Reid_Baseline", cfg.OUTPUT_DIR,'result')
        logger.info("Test Results:")
    else:
        logger = make_logger("Reid_Baseline", cfg.OUTPUT_DIR,'result_re-ranking')
        logger.info("Re-Ranking Test Results:") 
    
    device = torch.device(cfg.DEVICE)
    
    _, val_loader, num_query, num_classes = data_loader(cfg,cfg.DATASETS.NAMES)
    
    model = getattr(models, cfg.MODEL.NAME)(num_classes)
    model.load(cfg.OUTPUT_DIR,cfg.TEST.LOAD_EPOCH)
    if device:
        model.to(device) 
    model = model.eval()
    
    all_feats = []
    all_pids = []
    all_camids = []
    
    since = time.time()
    for data in tqdm(val_loader, desc='Feature Extraction', leave=False):
        with torch.no_grad():
            images, pids, camids = data
            if device:
                model.to(device) 
                images = images.to(device)
            
            feats = model(images)

        all_feats.append(feats)
        all_pids.extend(np.asarray(pids))
        all_camids.extend(np.asarray(camids))

    cmc, mAP = evaluation(all_feats,all_pids,all_camids,num_query,re_ranking)

    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
       
    test_time = time.time() - since
    logger.info('Testing complete in {:.0f}m {:.0f}s'.format(test_time // 60, test_time % 60))
    
if __name__=='__main__':
    fire.Fire(test)