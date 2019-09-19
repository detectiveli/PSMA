import fire
import fire
import os
import time
import torch
import numpy as np
import shutil
from torch.autograd import Variable
import models
from config import cfg
from data_loader import data_loader
from loss import make_loss
from optimizer import make_optimizer
from scheduler import make_scheduler
from logger import make_logger
from evaluation import evaluation
from datasets import PersonReID_Dataset_Downloader
from utils import check_jupyter_run
if check_jupyter_run():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
from data_loader.datasets_importer import init_dataset
from data_loader.transforms import transforms
from PIL import Image
import os
from torch.utils.data import DataLoader
from data_loader.data_loader import train_collate_fn, ImageDataset
from evaluation.re_ranking import re_ranking
import scipy.io as scio

from tensorboardX import SummaryWriter
writer = SummaryWriter('log')
train_transforms = transforms(cfg, is_train=True)
val_transforms = transforms(cfg, is_train=False)
# size_pic = 0 #12936 338 77616
vis_flag = False
def train(config_file, **kwargs):
    if vis_flag:
        os.mkdir('./vis/')
    cfg.merge_from_file(config_file)
    if kwargs:
        opts = []
        for k,v in kwargs.items():
            opts.append(k)
            opts.append(v)
        cfg.merge_from_list(opts)
    cfg.freeze()
    
    # PersonReID_Dataset_Downloader('./datasets',cfg.DATASETS.NAMES)
    
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    logger = make_logger("Reid_Baseline", output_dir,'log')
    logger.info("Using {} GPUS".format(1))
    logger.info("Loaded configuration file {}".format(config_file))
    logger.info("Running with config:\n{}".format(cfg))
    
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = torch.device(cfg.DEVICE)
    epochs = cfg.SOLVER.MAX_EPOCHS

    dataset_reference = init_dataset(cfg, cfg.DATASETS.NAMES + '_origin') #'Market1501_origin'
    train_set_reference = ImageDataset(dataset_reference.train, train_transforms)
    train_loader_reference = DataLoader(
        train_set_reference, batch_size=128, shuffle=False, num_workers=0,
        collate_fn=train_collate_fn
    )

    train_loader, val_loader, num_query, num_classes = data_loader(cfg,cfg.DATASETS.NAMES)
    # A = GCN_A(dataset_reference)
    # A_store = A.clone()

    model = getattr(models, cfg.MODEL.NAME)(num_classes)
    # model.load(cfg.OUTPUT_DIR, 1280)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)
    loss_fn = make_loss(cfg)

    logger.info("Start training")
    since = time.time()

    # feats_origin = feat_pre_generate(model, train_loader)
    count_log = 0
    top = 0
    top_update = 0
    update = 1
    top_counter = 0
    # model.update_A(A_store)
    # model.A_hat = model.A_hat.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        running_acc = 0
        count = 1

        if epoch % update == 0:
            if top_update < 80:
                train_step = 120
            else:
                train_step = 120
            if top_update % train_step  == 0:
                print("top: ", top)
                # A = GCN_A_iter(model, train_loader_reference, train_loader, top, cfg)
                A, path_labeled = GCN_A_iter_small(model, train_loader_reference, train_loader, top, cfg)
                top += 1
                top_counter += 1
                # top = top if top < 20 else 20
                model = getattr(models, cfg.MODEL.NAME)(num_classes)
                optimizer = make_optimizer(cfg, model)
                scheduler = make_scheduler(cfg, optimizer)
                A_store = A.clone()
                # model.update_A(A_store)
                # model.A_hat = model.A_hat.to(device)
            top_update += 1


        for data in tqdm(train_loader, desc='Iteration', leave=False):
            model.train()
            images, labels_batch, img_path = data
            index, index_labeled = find_index_by_path(img_path, dataset_reference.train, path_labeled)
            images_relevant, GCN_index, choose_from_nodes, labels = load_relevant(cfg, dataset_reference.train, index, A_store, labels_batch, index_labeled)
            # index, _ = find_index_by_path(img_path, dataset_reference.train)
            # images_relevant, GCN_index, choose_from_nodes, labels = load_relevant(cfg, dataset_reference.train, index, A_store, labels_batch)
            # if device:
            model.to(device)
            images = images_relevant.to(device)

            scores, feat = model(images, GCN_index)
            del images
            loss = loss_fn(scores, feat, labels.to(device), choose_from_nodes)
            # loss = loss_fn(scores[choose_from_nodes], feat[choose_from_nodes], labels_batch.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count = count + 1
            running_loss += loss.item()
            running_acc += (scores[choose_from_nodes].max(1)[1].cpu() == labels_batch).float().mean().item()

            writer.add_scalar('Train/Loss', loss.item(), count_log)
            count_log += 1
            
        # logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
        #                            .format(epoch+1, count, len(train_loader),
        #                            running_loss/count, running_acc/count,
        #                            scheduler.get_lr()[0]))
        scheduler.step()

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

def GCN_A(train_loader):
    # return torch.eye(size_pic)
    labels = []
    images_sources = []
    for data in train_loader.train:
        images_source, label, _ = data
        labels.append(label)
        images_sources.append(images_source)
    labels = torch.IntTensor(labels)
    # labels = torch.cat(labels, dim=0)
    A = torch.zeros(len(labels), len(labels))
    for count, label_each in enumerate(labels):
        A[count, labels == label_each] = 1

    # return A
    A_x = A
    # A_x = torch.Tensor(scio.loadmat('./similar_LOMO.mat')['matrix'])
    # A_x = -A_x + A_x.max()

    no_eye_A = A_x - torch.eye(A.shape[0]) * A_x
    sorted_A = no_eye_A.sort(descending=True)[1]
    #
    t = 1
    A_map = torch.zeros(A.shape[0], A.shape[0])
    for index, similar_x in enumerate(sorted_A[:, 0:t]):
        A_map[index, similar_x] = 1
        A_map[similar_x, index] = 1
    acc = (A - torch.eye(A.shape[0]))[A_map > 0]
    print(acc.sum() / (A_map > 0).sum(),' ', (A_map > 0).sum())

    A_map = A_map + torch.eye(A_map.shape[0])
    # A_map = A_map * A_x * A
    return A_map

def GCN_A_iter(model, train_loader, train_loader_orig, top, cfg):
    if top == 0:
        return torch.eye(size_pic)
    device = torch.device(cfg.DEVICE)
    model.eval().to(device)
    feats = []
    labels = []
    # get all features and distance
    debug = False
    img_paths = []
    with torch.no_grad():
        for data in tqdm(train_loader):
            images, label, img_path = data
            images = images.to(device)
            feat = model(images)
            feats.append(feat)
            labels.append(label)
            img_paths += img_path
    # scio.savemat('./pth.mat',{'path':img_paths})
    # labels = torch.IntTensor(labels)
    labels = torch.cat(labels, dim=0)
    feats = torch.cat(feats, dim=0)
    A_gt = torch.zeros(len(labels), len(labels))
    for count, label_each in enumerate(labels):
        A_gt[count, labels == label_each] = 1

    # get oneshot labels with GCN index
    pathes_labeded = []
    for unlabed_data in train_loader_orig:
        images, label, img_path = unlabed_data
        pathes_labeded += img_path
    index = {}
    index_list = []
    for unlabeled_one_shot_index, img_path in enumerate(pathes_labeded):
        max_index = img_path.split("/")[-1]
        for index_origin, path_of_origin in enumerate(img_paths):
            id_from_path = path_of_origin.split("/")[-1]
            if max_index == id_from_path:
                index[index_origin] = unlabeled_one_shot_index
                index_list.append(index_origin)
                break

    dis_feats = get_euclidean_dist(feats, index_list)
    dis_feats = -dis_feats + dis_feats.max()
    A = dis_feats
    A_map = torch.zeros(A.shape[0], A.shape[0])
    no_eye_A = A - torch.eye(A.shape[0]) * A

    test_top = top
    sorted_A = no_eye_A.to(device).sort(descending=True)[1][:, 0:test_top]
    for one_labeled_index in index:
        for chosen_index, choose_one in enumerate(sorted_A[one_labeled_index]):
            exist_index_top_e = False
            choose_from_top = no_eye_A[choose_one][index_list].sort(descending=True)[1][:1]
            for i in choose_from_top:
                if index_list[i] == one_labeled_index:
                    exist_index_top_e = True
                    break

            if (choose_one not in index.keys()) & exist_index_top_e:
                A_map[one_labeled_index][choose_one] = 1
                A_map[choose_one][one_labeled_index] = 1

    # for test
    acc = (A_gt - torch.eye(A.shape[0]))[A_map > 0]
    print(acc.sum() / (A_map > 0).sum(),' ', (A_map > 0).sum())
    A_map = A_map + torch.eye(A_map.shape[0])
    return A_map

def GCN_A_iter_small(model, train_loader, train_loader_orig, top, cfg):
    vis = len(train_loader_orig.dataset)
    A_base = torch.zeros(vis, len(train_loader.dataset))
    # img_paths = scio.loadmat('./path_all.mat')['path']
    if top == 0:
        img_paths = []
        for data in tqdm(train_loader):
            images, label, img_path = data
            img_paths += img_path
    # scio.savemat('./path_all.mat', {'path': img_paths})
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
    for unlabed_data in train_loader_orig:
        images, label, img_path = unlabed_data
        pathes_labeded += img_path
        all_labels.append(label)
    all_labels = torch.cat(all_labels, dim=0)
    index = {}
    index_list = []
    for unlabeled_one_shot_index, img_path in enumerate(pathes_labeded):
        for index_origin, path_of_origin in enumerate(img_paths):
            if img_path.split("/")[-1] == path_of_origin.split("/")[-1]:
                index[index_origin] = unlabeled_one_shot_index
                index_list.append(index_origin)
                A_base[unlabeled_one_shot_index][index_origin] = 1
                break
    if top == 0:
        if vis_flag:
            path = './vis/orig'
            os.mkdir(path)
            f = open(path + '/temp.txt', 'w')
            f.write('\n'.join(pathes_labeded))
            f.close()
            for i in range(vis):
                path_temp = path + '/' + str(i)
                os.mkdir(path_temp)
                image_origin = pathes_labeded[i]
                shutil.copy(image_origin, path_temp)
        return A_base, pathes_labeded
    else:
        A_gt = torch.zeros(vis, len(labels))
        for count, label_each in enumerate(labels[index_list]):
            A_gt[count, labels == label_each] = 1

        # 2 calculate distance
        dis_feats = get_euclidean_dist(feats, index_list)
        dis_feats = -dis_feats + dis_feats.max()
        A = dis_feats
        A_map = torch.zeros(vis, A.shape[1])
        no_eye_A = A - A_base * A

        test_top = top
        sorted_A = no_eye_A.to(device).sort(descending=True)[1][:, 0:test_top]
        for index_labeled, one_labeled in enumerate(sorted_A):
            for chosen_index, choose_one in enumerate(one_labeled):
                exist_index_top_e = False
                choose_from_top = no_eye_A[:, choose_one].sort(descending=True)[1][:6]
                for i in choose_from_top:
                    if i == index_labeled:
                        exist_index_top_e = True
                        break
                if (choose_one not in index.keys()) & exist_index_top_e:
                    A_map[index_labeled][choose_one] = 1
                    # A_map[choose_one][index_labeled] = 1
        # for vis
        if vis_flag:
            path = './vis/top' + str(top)
            os.mkdir(path)
            f = open(path + '/temp.txt', 'w')
            f.write('\n'.join(pathes_labeded))
            f.close()
            for i in range(vis):
                path_temp = path + '/' + str(i)
                os.mkdir(path_temp)
                image_origin = pathes_labeded[i]
                shutil.copy(image_origin, path_temp)
                for target in A_map[i].nonzero():
                    shutil.copy(img_paths[target], path_temp)

        # if vis > 800:
        #     for i, label in enumerate(all_labels):
        #         for j, temp_label in enumerate(all_labels):
        #             if temp_label == label:
        #                 A_map[i][index_list[j]] = 1
        # for test
        acc = (A_gt - A_base)[A_map > 0]
        print(acc.sum() / (A_map > 0).sum(),' ', (A_map > 0).sum())
        A_map = A_map + A_base
        return A_map, pathes_labeded


def get_euclidean_dist(feat_list, index_list):
    all_feats = feat_list#torch.cat(feat_list, dim=0)
    qf = all_feats[index_list]
    gf = all_feats
    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())

    # all_feats = feat_list#torch.cat(feat_list, dim=0)
    # n = all_feats.shape[0]
    # distmat = torch.pow(all_feats, 2).sum(dim=1, keepdim=True).expand(n, n) + \
    #           torch.pow(all_feats, 2).sum(dim=1, keepdim=True).expand(n, n).t()
    # distmat.addmm_(1, -2, all_feats, all_feats.t())

    # re ranking
    # distmat = np.zeros((all_feats.shape[0], all_feats.shape[0]))
    # qf = all_feats[index_list]
    # gf = all_feats
    # q_g_dist = np.dot(qf.data.cpu(), np.transpose(gf.data.cpu()))
    # q_q_dist = np.dot(qf.data.cpu(), np.transpose(qf.data.cpu()))
    # g_g_dist = np.dot(gf.data.cpu(), np.transpose(gf.data.cpu()))
    # print('Re-ranking:')
    # distmat_reranking = re_ranking(q_g_dist, q_q_dist, g_g_dist)
    # distmat = torch.Tensor(distmat_reranking)
    return distmat

def load_relevant(cfg, data_train, index_batch_withid, A_map, label_labeled, index_labeled=None):
    indices = get_indice_graph(A_map, index_batch_withid, 96, index_labeled)
    indices_to_index = {}
    # data_choose = data_train[indices]
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
