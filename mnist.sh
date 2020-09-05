#!/bin/bash
python anatomy.py --net dnn2 --dataset mnist --gpu-id 1 --workers 8 -b 1 --layer-info layer_info --root /data/xujw/anatomy
