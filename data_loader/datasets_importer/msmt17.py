# encoding: utf-8
import os
import glob
import re
from .BaseDataset import BaseImageDataset


class MSMT17(BaseImageDataset):
    """
    MSMT17

    Dataset statistics:
    # appearance: 805
    # images: 20,411 (train) + 3,443 (query) + 21,542 (gallery)
    """
    dataset_dir = 'MSMT17'

    def __init__(self, cfg, verbose=True, **kwargs):
        super(MSMT17, self).__init__()
        self.dataset_dir = os.path.join(cfg.DATASETS.STORE_DIR, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'mask_train_v2')
        self.test_dir = os.path.join(self.dataset_dir, 'mask_test_v2')
        
        self.train_list = os.path.join(self.dataset_dir, 'list_train.txt')
        self.query_list = os.path.join(self.dataset_dir, 'list_query.txt')
        self.gallery_list = os.path.join(self.dataset_dir, 'list_gallery.txt')

        self._check_before_run()

        train = self._process_dir(self.train_dir, self.train_list, relabel=True)
        query = self._process_dir(self.test_dir, self.query_list, relabel=False)
        gallery = self._process_dir(self.test_dir, self.gallery_list, relabel=False)

        if verbose:
            print("=> MSMT17 Loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not os.path.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))
        if not os.path.exists(self.train_list):
            raise RuntimeError("'{}' is not available".format(self.train_list))
        if not os.path.exists(self.query_list):
            raise RuntimeError("'{}' is not available".format(self.query_list))
        if not os.path.exists(self.gallery_list):
            raise RuntimeError("'{}' is not available".format(self.gallery_list))

    def _process_dir(self, dir_path, txt_list, relabel=False):
        text_file = open(txt_list, "r")
        img_paths = text_file.read().split('\n')
        img_paths = list(filter(None, img_paths))

        pid_container = set()
        for img_path in img_paths:
            pid = img_path.split(' ')[1]
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        dataset = []
        for img_path in img_paths:
            pid = img_path.split(' ')[1]
            camid = str(int(img_path.split(' ')[0].split('_')[2]))

            if relabel: pid = pid2label[pid]
            dataset.append((os.path.join(dir_path,img_path.split(' ')[0]), pid, camid))

        return dataset
