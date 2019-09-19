import os
import glob
import re
import shutil

#path_data = '~/datasets/Market1501'
#os.makedirs(os.path.join(path_data, 'bounding_box_train_query_one'))

path = '~/datasets/Market1501/bounding_box_train'
img_paths = glob.glob('*.jpg')
pattern = re.compile(r'([-\d]+)_c(\d)')
pid_container = set()
for img_path in img_paths:
    pid, value = map(int, pattern.search(img_path).groups())
    if pid == -1: continue  # junk images are just ignored
    if pid not in pid_container:
    	if value == 1:
    		print(img_path)
	    	shutil.copy(img_path, '../bounding_box_train_query_one/'+img_path)
	    	pid_container.add(pid)
# pid2label = {pid: label for label, pid in enumerate(pid_container)}
