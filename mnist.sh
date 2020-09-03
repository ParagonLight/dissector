#!/bin/bash
python3 anatomy.py --net lenet5 --dataset mnist --gpu-id 1 --workers 8 -b 1 --layer-info layer_info --root /data/xjw/anatomy