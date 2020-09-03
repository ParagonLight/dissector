#!/bin/bash
python3 anatomy.py --net vgg16 --dataset cifar100 --gpu-id 1 --workers 8 -b 1 --layer-info layer_info --root /data/xjw/anatomy