# Code for PSMA

## envs
This code is mmodified from https://github.com/manutdzou/Person_ReID_Baseline with a same environment.

## dataset
We tested our framework on two main datasets: Market1501 and Duke

To achieve the same performance as I did, I recommand to download the dataset I use and put it under datasets:

(BAIDU) link: https://pan.baidu.com/s/1_XJygK4TTlG1l_VlUY4K7g pin: 4ak9

The PATH for dataset is in ./config/default.py 
(for example: _C.DATASETS.STORE_DIR = ('/home/lihui/datasets/PSM_DATA'))

## train

python train.py ./config/market_softmax_Htriplet_GAN.yaml

### The performance log for checking
The log I print is under the /log_answer directory. This can be used as a reference for your training process.

