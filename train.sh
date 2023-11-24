#!/bin/bash

python3 train.py --model unet --batch_size 128
python3 train.py --model vit-nano --batch_size 256
python3 train.py --model vit-nano --batch_size 256 --teacher unet
python3 train.py --model vit-micro --batch_size 128
python3 train.py --model vit-tiny --batch_size 64

python3 evaluate.py --batch_size 128