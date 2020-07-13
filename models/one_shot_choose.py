import os
import glob
import re
import shutil
import numpy as np
import random

#One-shot example generator
Method = 'EUG'
path_data = '~/datasets/Market1501'
path_store =  'bounding_box_train_query_one_' + Method
target_path = os.path.join(path_data, path_store)
if os.path.exists(target_path):
    os.remove(target_path)
os.makedirs(target_path)
path = path_data + '/bounding_box_train/'
img_paths = glob.glob(path + '*.jpg')
pattern = re.compile(r'([-\d]+)_c(\d)')
pid_container = set()

# PSMA one-shot choose
if Method == 'PSMA':
    for img_path in img_paths:
        pid, value = map(int, pattern.search(img_path).groups())
        if pid == -1: continue  # junk images are just ignored
        if pid not in pid_container:
            print(img_path)
            shutil.copy(img_path, path_store + '/' + img_path)
            pid_container.add(pid)

else: # EUG one-shot choose
    pid_to_index = {}
    for index, img_path in enumerate(img_paths):
        pid, value = map(int, pattern.search(img_path).groups())
        if pid == -1: continue  # junk images are just ignored
        pid_to_index[pid] = index

    np.random.seed(0)
    random.seed(0)
    dataset_in_pid_cam = [[[] for _ in range(6)] for _ in range(751) ]
    for img_path in img_paths:
        pid, value = map(int, pattern.search(img_path).groups())
        dataset_in_pid_cam[pid_to_index[pid]][value].append([img_path])

    # generate the labeled dataset by randomly selecting a tracklet from the first camera for each identity
    for pid, cams_data  in enumerate(dataset_in_pid_cam):
        for camid, videos in enumerate(cams_data):
            if len(videos) != 0:
                selected_video = random.choice(videos)
                break
        shutil.copy(selected_video, path_store + '/' + selected_video)
