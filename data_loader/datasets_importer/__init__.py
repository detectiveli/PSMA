# encoding: utf-8
from .market1501 import Market1501, Market1501_origin
from .dukemtmc import DukeMTMC, DukeMTMC_origin
from .ImageDataset import ImageDataset

__factory = {
    'Market1501_origin': Market1501_origin,
    'Market1501': Market1501,
    'DukeMTMC_origin': DukeMTMC_origin,
    'DukeMTMC': DukeMTMC,
}


def get_names():
    return __factory.keys()


def init_dataset(cfg,dataset_name, *args, **kwargs):
    if cfg.DATASETS.NAMES not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(dataset_name))
    return __factory[dataset_name](cfg,*args, **kwargs)
