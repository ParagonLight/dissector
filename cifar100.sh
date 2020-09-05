#!/bin/bash
#python anatomy.py --net dense --dataset cifar10 --gpu-id 1 --workers 8 -b 1 --layer-info layer_info --root /data/xujw/anatomy
#python anatomy.py --net wrn --dataset cifar10 --gpu-id 1 --workers 8 -b 1 --layer-info layer_info --root /data/xujw/anatomy
#python anatomy.py --net vgg16 --dataset cifar10 --gpu-id 1 --workers 8 -b 1 --layer-info layer_info --root /data/xujw/anatomy
#python anatomy.py --net dense --dataset cifar100 --gpu-id 1 --workers 8 -b 1 --layer-info layer_info --root /data/xujw/anatomy
python anatomy.py --net vgg16 --dataset cifar100 --gpu-id 1 --workers 8 -b 1 --layer-info layer_info --root /data/xujw/anatomy
#python anatomy.py --net resnext --dataset cifar100 --gpu-id 1 --workers 8 -b 1 --layer-info layer_info --root /data/xujw/anatomy
