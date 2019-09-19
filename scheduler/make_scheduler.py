import torch
from torch.optim import lr_scheduler


def make_scheduler(cfg,optimizer):
    scheduler = lr_scheduler.StepLR(optimizer, cfg.SOLVER.STEP, cfg.SOLVER.GAMMA)
    return scheduler
