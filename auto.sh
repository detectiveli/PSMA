#!/bin/sh

python test.py ./config/cuhk_softmax.yaml True
python test.py ./config/cuhk_softmax_triplet.yaml True

python test.py ./config/market_softmax.yaml True
python test.py ./config/market_softmax_triplet.yaml True

python test.py ./config/duke_softmax.yaml True
python test.py ./config/duke_softmax_triplet.yaml True

python test.py ./config/ntu_softmax.yaml True
python test.py ./config/ntu_softmax_triplet.yaml True

# python test.py ./config/msmt_softmax.yaml True
# python test.py ./config/msmt_softmax_triplet.yaml
