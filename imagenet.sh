#!/bin/bash
python anatomy.py --net resnet101 --dataset imagenet --gpu-id 1 --workers 8 -b 1 --layer-info layer_info --root /data/xjw/anatomy
