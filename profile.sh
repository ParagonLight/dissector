#!/bin/bash
python merge_raw_layer_outputs.py --net resnet101 --dataset imagenet --dataset_size 50000 --dataset_label 1000 --gpu-id 13 --layer_info layer_info --root /data/xujw/anatomy/ --weightformula exp --para_alpha 0 --para_beta 1

