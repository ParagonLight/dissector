#!/bin/bash
python anatomy.py --net vgg16 --dataset imagenet --gpu-id 13 --workers 8 -b 1 --layer-info layer_info --root /data/xujw/anatomy
# python anatomy.py --net resnet50 --dataset imagenet --gpu-id 13 --workers 8 -b 1 --layer-info layer_info --root /data/xujw/anatomy


