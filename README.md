# dissector

Understanding behaviors of convolutional neural networks on image classification

# Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- *Pretrained model and dataset

# How to use

Suppose we have ImageNet dataset and pretrained ResNet101 model and corresponding pretrained 6 submodels.

1. Create several folders in `data` folder.

- `data/imagenet`: root folder of imagenet dataset

- `data/imagenet/data`: root folder of imagenet dataset files

- `data/imagenet/models/resnet101`: root folder of ResNet101 sub models for imagenet dataset

- `data/imagenet/tensor_pub`: root folder for anatomy outputs

2. Put pretrained submodels model in `data/imagenet/models/resnet101`.

3. Create file `layer_info` to write layers' info, which the layers are used for anatomy.

    For each row, write `layer_name,layer's output size`

4. Run anatomy to produce results from each submodel for all instances.

    `python3 anatomy.py` or `sh imagenet.sh`

5. Run merge_raw_layer_outputs.py to merge results from all layers.

    `python3 merge_raw_layer_outputs.py`

Use `--help` to see arguments.

# Dissector Example for ResNet101 on ImageNet

## 1. Download example folder for ImageNet dataset 

https://1drv.ms/u/s!Anr26WqGCJOLsSICmSnSpZgvJM0K

`ILSVRC-val.lmdb` is ImageNet validation set.

`imagenet_pub` is the root folder of the target. Pretrained submodels and layer info are all in `imagenet_pub/models/resnet101`

`tensor_pub` is the root folder for outputs of dissector.
