# Dissector

Understanding behaviors of convolutional neural networks on image classification.

**This repo covers the implementation of the following ICSE 2020 paper:**

"DISSECTOR: Input Validation for Deep Learning Applications by Crossing-layer Dissection
" (DISSECTOR).


# Supported Models and Datasets
  ResNet101 + ImageNet
  ResNet50 + ImageNet
  VGG16 + ImageNet
  
  ResNeXt + CIFAR-100
  VGG16 + CIFAR-100
  DenseNet + CIFAR-100

  WRN + CIFAR-10
  VGG16 + CIFAR-10
  DenseNet + CIFAR-10

  LeNet4 + MNIST
  LeNet5 + MNIST
  DNN2 + MNIST
  
 - ToDo:
    Support for submodel training.

# Installation

- Install PyTorch and TorchVision ([pytorch.org](http://pytorch.org))
  ```
  pip install torch torchvision
  ```
- *Pretrained model and dataset
- Install requirements

    ```
    pip install lmdb msgpack progressbar pillow sklearn
    ```

# Dissector Example for ResNet101 on ImageNet

## Fetch the example data folder for ImageNet dataset 

https://1drv.ms/u/s!Anr26WqGCJOLsSICmSnSpZgvJM0K

`ILSVRC-val.lmdb` is ImageNet validation set. You should change the dataset path in `utils.py`.

    imagenet_val_path = YOURPATH

`imagenet_pub` is the root folder of the target. Pretrained submodels and layer info are all in `imagenet_pub/models/resnet101`.

`tensor_pub` is the root folder for outputs of dissector.

 ## How to use

Suppose we have ImageNet dataset and pretrained ResNet101 model and corresponding pretrained 6 submodels.

1. Create the root folder, such as `YOURPROOT`.

2. Create several folders in `YOURROOT` folder.

- `YOURROOT/imagenet`: root folder of imagenet dataset.

- `YOURROOT/imagenet/data`: root folder of imagenet dataset files.

- `YOURROOT/imagenet/models/resnet101`: root folder of ResNet101 sub models for imagenet dataset.

- `YOURROOT/imagenet/tensor_pub`: root folder for anatomy outputs.

    - `YOURROOT/imagenet/tensor_pub/res_layer1`: folder for output of submodel res_layer1.
    - `YOURROOT/imagenet/tensor_pub/res_layer2`: folder for output of submodel res_layer2.
    - `YOURROOT/imagenet/tensor_pub/res_block8`: folder for output of submodel res_block8.
    - `YOURROOT/imagenet/tensor_pub/res_block16`: folder for output of submodel res_block16.
    - `YOURROOT/imagenet/tensor_pub/res_layer3`: folder for output of submodel res_layer3.
    - `YOURROOT/imagenet/tensor_pub/res_layer4`: folder for output of submodel res_layer4.
    - `YOURROOT/imagenet/tensor_pub/out`: folder for output of ResNet101.

2. Put pretrained submodels model in `data/imagenet/models/resnet101`.

3. Create file `layer_info` to write layers' info, which the layers are used for anatomy.

    For each row, write `layer_name,layer's output size`

4. Run anatomy to produce results from each submodel for all instances.

    ```
    sh imagenet.sh
    ```
5. Run merge_raw_layer_outputs.py to merge results from all layers.

    ```
    sh profile.sh
    ```
    this is for running imagenet using Dissector-linear as an example.
Use `--help` to see arguments.

## What to expect
   For our example Imagenet+resnet101, AUC results are as follows:
    
| Env |    Dissector-linear    |    Dissector-log    |    Dissector-exp   | 
|:---------------:|:-----------------:|:-----------------:|:-----------------:|
| **[Python 2.7.15, Pytorch 0.4.1] (our ICSE'20 paper setting)** | **0.8250** | **0.8223** | **0.8562** |
| [Python 3.6.9, Pytorch 1.4.0] | 0.8212 | 0.8237 | 0.8547 |

# Citation

If you find this repo useful for your research, please consider citing the paper.

```
@inproceedings{Wang2019Dissector,
  title={Dissector: Input Validation for Deep Learning Applications by Crossing-layer Dissection},
  author={Huiyan Wang and Jingwei Xu and Chang Xu and Xiaoxing Ma and Jian Lu},
  booktitle={The 42th International Conference on Software Engineering},
  year={2020}
}
```

For any questions, please contact Huiyan Wang (cocowhy1013@gmail.com) and Jingwei Xu (jingweix@nju.edu.cn).
