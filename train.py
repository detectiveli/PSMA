import time
import torch
import numpy as np
import models
from config import cfg
from data_loader import data_loader
from loss import make_loss
from optimizer import make_optimizer
from scheduler import make_scheduler
from logger import make_logger
from evaluation import evaluation
from utils import check_jupyter_run
from tqdm import tqdm
from data_loader.datasets_importer import init_dataset
from data_loader.transforms import transforms
from PIL import Image
import os
from torch.utils.data import DataLoader
from data_loader.data_loader import train_collate_fn, ImageDataset
from tensorboardX import SummaryWriter
writer = SummaryWriter('log')
train_transforms = transforms(cfg, is_train=True)
val_transforms = transforms(cfg, is_train=False)

def train(config_file, **kwargs):
    # 1. config
    cfg.merge_from_file(config_file)
    if kwargs:
        opts = []
        for k,v in kwargs.items():
            opts.append(k)
            opts.append(v)
        cfg.merge_from_list(opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    logger = make_logger("Reid_Baseline", output_dir,'log')
    logger.info("Using {} GPUS".format(1))
    logger.info("Loaded configuration file {}".format(config_file))
    logger.info("Running with config:\n{}".format(cfg))
    
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    device = torch.device(cfg.DEVICE)
    epochs = cfg.SOLVER.MAX_EPOCHS

    # 2. datasets
    # Load the original dataset
    dataset_reference = init_dataset(cfg, cfg.DATASETS.NAMES + '_origin') #'Market1501_origin'
    train_set_reference = ImageDataset(dataset_reference.train, train_transforms)
    train_loader_reference = DataLoader(
        train_set_reference, batch_size=128, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=train_collate_fn
    )

    # Load the one-shot dataset
    train_loader, val_loader, num_query, num_classes = data_loader(cfg, cfg.DATASETS.NAMES)

    # 3. load the model and optimizer
    model = getattr(models, cfg.MODEL.NAME)(num_classes)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)
    loss_fn = make_loss(cfg)
    logger.info("Start training")
    since = time.time()

    top = 0 # the choose of the nearest sample
    top_update = 0 # the first iteration train 80 steps and the following train 40

    # 4. Train and test
    for epoch in range(epochs):
        running_loss = 0.0
        running_acc = 0
        count = 1

        # get nearest samples and reset the model
        if top_update < 80:
            train_step = 80
        else:
            train_step = 40
        if top_update % train_step == 0:
            print("top: ", top)
            A, path_labeled = PSP(model, train_loader_reference, train_loader, top, cfg)
            top += cfg.DATALOADER.NUM_JUMP
            model = getattr(models, cfg.MODEL.NAME)(num_classes)
            optimizer = make_optimizer(cfg, model)
            scheduler = make_scheduler(cfg, optimizer)
            A_store = A.clone()
        top_update += 1


        for data in tqdm(train_loader, desc='Iteration', leave=False):
            model.train()
            images, labels_batch, img_path = data
            index, index_labeled = find_index_by_path(img_path, dataset_reference.train, path_labeled)
            images_relevant, GCN_index, choose_from_nodes, labels = load_relevant(cfg, dataset_reference.train, index, A_store, labels_batch, index_labeled)
            # if device:
            model.to(device)
            images = images_relevant.to(device)

            scores, feat = model(images)
            del images
            loss = loss_fn(scores, feat, labels.to(device), choose_from_nodes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count = count + 1
            running_loss += loss.item()
            running_acc += (scores[choose_from_nodes].max(1)[1].cpu() == labels_batch).float().mean().item()

        scheduler.step()


        # for model save if you need
        # if (epoch+1) % checkpoint_period == 0:
        #     model.cpu()
        #     model.save(output_dir,epoch+1)

        # Validation
        if (epoch+1) % eval_period == 0:
            all_feats = []
            all_pids = []
            all_camids = []
            for data in tqdm(val_loader, desc='Feature Extraction', leave=False):
                model.eval()
                with torch.no_grad():
                    images, pids, camids = data

                    model.to(device)
                    images = images.to(device)

                    feats = model(images)
                    del images
                all_feats.append(feats.cpu())
                all_pids.extend(np.asarray(pids))
                all_camids.extend(np.asarray(camids))

            cmc, mAP = evaluation(all_feats,all_pids,all_camids,num_query)
            logger.info("Validation Results - Epoch: {}".format(epoch+1))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10, 20]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))


    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('-' * 10)
'''
author: lihui

This function help to choose the nearest neighbors

if top is 0 means only choose the original one-shot sample

'''
def PSP(model, train_loader, train_loader_orig, top, cfg):
    vis = len(train_loader_orig.dataset)
    A_base = torch.zeros(vis, len(train_loader.dataset)) # the one-shot example
    A_map = torch.zeros(vis, len(train_loader.dataset))

    if top == 0: # no PSP choose
        img_paths = []
        for data in tqdm(train_loader):
            images, label, img_path = data
            img_paths += img_path
    else:
        device = torch.device(cfg.DEVICE)
        model.eval().to(device)
        feats = []
        labels = []
        # 1 get all features and distance
        img_paths = []
        with torch.no_grad():
            for data in tqdm(train_loader):
                images, label, img_path = data
                images = images.to(device)
                feat = model(images)
                feats.append(feat.cpu())
                labels.append(label)
                img_paths += img_path
        labels = torch.cat(labels, dim=0)
        feats = torch.cat(feats, dim=0)

    pathes_labeded = []
    all_labels = []
    # only use for accuracy estimate
    for unlabed_data in train_loader_orig:
        images, label, img_path = unlabed_data
        pathes_labeded += img_path
        all_labels.append(label)

    index = {}
    index_list = []
    for unlabeled_one_shot_index, img_path in enumerate(pathes_labeded):
        for index_origin, path_of_origin in enumerate(img_paths):
            if cfg.DATALOADER.METHOD != 'GAN':
                if img_path.split("/")[-1] == path_of_origin.split("/")[-1]:
                    index[index_origin] = unlabeled_one_shot_index
                    index_list.append(index_origin)
                    A_base[unlabeled_one_shot_index][index_origin] = 1
                    break
            else:
                if img_path.split("/")[-1] == path_of_origin.split("/")[-1]:
                    index[index_origin] = unlabeled_one_shot_index
                    index_list.append(index_origin)
                    A_base[unlabeled_one_shot_index][index_origin] = 1

                if img_path.split("/")[-1].split("_")[0:3] == path_of_origin.split("/")[-1].split("_")[0:3]:
                    A_base[unlabeled_one_shot_index][index_origin] = 1
    if top == 0:
        return A_base, pathes_labeded
    else:
        A_gt = torch.zeros(vis, len(labels))
        for count, label_each in enumerate(labels[index_list]):
            A_gt[count, labels == label_each] = 1

        # calculate distance
        if cfg.DATALOADER.METHOD != 'GAN':
            dis_feats = get_euclidean_dist(feats, feats[index_list])
        else:
            # find the GAN same picture
            feats_new = feats[index_list]
            for i, path_want in enumerate(index_list):
                counter = 1
                for j, path_all in enumerate(img_paths):
                    if img_paths[path_want].split("/")[-1].split("_")[0:3] == path_all.split("/")[-1].split("_")[0:3]:
                        feats_new[i] += feats[j]
                        counter += 1
                        # A_map[i][j] = 1
                feats_new[i] /= counter
            dis_feats = get_euclidean_dist(feats, feats_new)

        dis_feats = -dis_feats + dis_feats.max()
        A = dis_feats

        no_eye_A = A - A_base * A

        test_top = top
        sorted_A = no_eye_A.to(device).sort(descending=True)[1][:, 0:test_top]
        for index_labeled, one_labeled in enumerate(sorted_A):
            for chosen_index, choose_one in enumerate(one_labeled):
                exist_index_top_e = False
                choose_from_top = no_eye_A[:, choose_one].sort(descending=True)[1][:1]
                for i in choose_from_top:
                    if i == index_labeled:
                        exist_index_top_e = True
                        break
                if (choose_one not in index.keys()) & exist_index_top_e:
                    A_map[index_labeled][choose_one] = 1
                    # A_map[choose_one][index_labeled] = 1

        # for test
        acc = (A_gt - A_base)[A_map > 0]
        print(acc.sum() / (A_map > 0).sum(),' ', (A_map > 0).sum())
        A_map = A_map + A_base
        return A_map, pathes_labeded

def get_euclidean_dist(gf, qf):
    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    return distmat

def load_relevant(cfg, data_train, index_batch_withid, A_map, label_labeled, index_labeled=None):
    indices = get_indice_graph(A_map, index_batch_withid, 96, index_labeled)
    indices_to_index = {}
    images = []
    for counter, indice in enumerate(indices):
        img_path = data_train[indice][0]
        img_orig = Image.open(img_path).convert('RGB')
        img = train_transforms(img_orig)
        images.append(img)
        indices_to_index[indice] = counter
    images = torch.stack(images)

    choose_from_nodes = []
    for id in index_batch_withid:
        choose_from_nodes.append(indices_to_index[id])

    if index_labeled is None: return images, indices, choose_from_nodes, None
    labels = []
    for indice in indices:
        for id, each_labeled in zip(index_labeled, label_labeled):
            if (A_map[id][indice] > 0):
                labels.append(each_labeled)
                break
    labels = torch.stack(labels)

    return images, indices, choose_from_nodes, labels

def get_indice_graph(adj, mask, size, index_labeled):
    indices = mask
    pre_indices = set()
    indices = set(indices)
    choosen = indices if index_labeled is None else set(index_labeled)

    # pre_indices = indices.copy()
    candidates = get_candidates(adj, choosen) - indices
    if len(candidates) > size - len(indices):
        candidates = set(np.random.choice(list(candidates), size-len(indices), False))
    indices.update(candidates)
    # print('indices size:-------------->', len(indices))
    return sorted(indices)

def get_candidates(adj, new_add):
    same = adj[sorted(new_add)].sum(dim=0).nonzero().squeeze().numpy()
    return set(tuple(same))

def find_index_by_path(path, data_origin, path_labeled=None):
    index = []
    index_labeled = []
    for img_path in path:
        max_index = img_path.split("/")[-1]
        for index_origin, path_of_origin in enumerate(data_origin):
            id_from_path = path_of_origin[0].split("/")[-1]
            if max_index == id_from_path:
                index.append(index_origin)
                break
        if path_labeled is None: continue
        for index_labeded, path_temp in enumerate(path_labeled):
            if max_index == path_temp.split("/")[-1]:
                index_labeled.append(index_labeded)
                break
    return index, index_labeled

if __name__=='__main__':
    import fire
    fire.Fire(train)
