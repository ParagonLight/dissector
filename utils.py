# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torchvision
from torchvision import datasets, transforms
from models.pretrained.dnn import NN
from models.pretrained.lenet5 import LeNet5
from models.pretrained.lenet4 import LeNet4
from models.pretrained.vgg16 import VGG16
from models.pretrained import densenet
from models.pretrained import wrn
from models.pretrained import resnext
from models.embedding.fc import FC
from models.embedding import resnet_layer4
from models.embedding import resnet_layer3
from models.embedding import resnet_layer3_block8
from models.embedding import resnet_layer3_block16
from models.embedding import resnet_layer2
from models.embedding import resnet_layer1
from models.pretrained import resnet
import numpy as np
from PIL import Image
from lmdbdataset import lmdbDataset
import math

imagenet_train_path = '/data/share/ImageNet/ILSVRC-train.lmdb'
imagenet_val_path = '/data/share/ImageNet/ILSVRC-val.lmdb'


def get_layer_info(root, dataset, model, name):
    model_root = get_model_root(root, dataset, model)
    layers = []
    cols = []
    with open('{}/{}'.format(model_root, name)) as f:
        data = f.readlines()
        for x in data:
            x = x.strip().split(',')
            layers.append(x[0])
            cols.append(int(x[1]))

    print('sub models:', layers)
    return layers, cols

def softmax(tensor, dim=0):
    return torch.nn.functional.log_softmax(tensor,dim=dim)
    #return torch.nn.Softmax()(tensor)

def log_softmax_to_softmax(tensor):
    e = math.e
    return e**tensor

def get_root(root, dataset, elem, suffix=None):
    if suffix == None:
        return '{}/{}/{}'.format(root, dataset, elem)
    else:
        return '{}/{}/{}/{}'.format(root, dataset + '_' + suffix, elem, suffix)

def get_model_root(root, dataset, model):
    return get_root(root, dataset, 'models', model)

def get_pretrained_model(root, dataset, model):
    model_root = get_model_root(root, dataset, model)
    return '{}/{}_{}.pth'.format(model_root, dataset, model)

def get_weight_model(root, dataset, model):
    model_root = get_model_root(root, dataset, model)
    return '{}/weight.pth'.format(model_root)
def get_embd_prefix(root, dataset, model):
    return '{}/{}_{}.embd'.format(get_model_root(root, dataset, model), dataset, model)

def save_tensor(x, f):
    torch.save(x,f)

def save_tensors_in_dict(d, root, index, ttype):
    for key, value in d.items():
        save_tensor(value, root + '/' + index + '_' + key + '_' + ttype + '.pt')


def save_mnist_image(tensor, path):
    torchvision.utils.save_image(tensor, path)

def save_image_from_array(array, path):
    img = Image.fromarray(array*255)
    img = img.convert('RGB')
    img.save(path)


def save_imagenet_image(tensor, path):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = tensor
    x[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
    torchvision.utils.save_image(x, path)
    del x, tensor

def save_cifar10_image(tensor, path):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    x = tensor
    x[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
    #x = np.clip(x, 0, 1)
    torchvision.utils.save_image(x, path)
    #x = x.transpose((2,0,1))
    #x = x.numpy().transpose((1, 2, 0))
    #img = Image.fromarray(x, 'RGB')
    #img.save(path)

def load_imagenet_test(batch_size, workers):
    test_loader = torch.utils.data.DataLoader(
        lmdbDataset(imagenet_val_path, False),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True
    )
    return test_loader


def load_imagenet(batch_size, workers):
    train_loader = torch.utils.data.DataLoader(
        lmdbDataset(imagenet_train_path, True),
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        lmdbDataset(imagenet_val_path, False),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True
    )
    return train_loader, test_loader

def load_cifar100(root):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])),
            batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root, train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])),
            batch_size=1, shuffle=False)
    return train_loader, test_loader

def load_cifar10(root):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])),
            batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root, train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])),
            batch_size=1, shuffle=False)
    return train_loader, test_loader

def load_fashionMNIST(root):
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(root, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(root, train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=1, shuffle=False)
    return train_loader, test_loader

def load_resnext_model(pretrained_model, cardinality, num_classes, depth, widen_factor, dropRate):
    model = resnext.resnext(num_classes=num_classes, depth=depth, cardinality=cardinality, widen_factor=widen_factor, dropRate=dropRate)#.cuda()
    model = torch.nn.DataParallel(model, [0]).cuda()
    checkpoint = torch.load(pretrained_model)
    # print(checkpoint['state_dict'].keys())
    model.load_state_dict(checkpoint['state_dict'])
    # print(model.state_dict().keys())
    model.eval()
    return model


def load_dense_model(pretrained_model, num_classes, depth, growthRate, compressionRate, dropRate):
    model = densenet.densenet(num_classes=num_classes, depth=depth, growthRate=growthRate, compressionRate=compressionRate, dropRate=dropRate)#.cuda()
    model = torch.nn.DataParallel(model, [0]).cuda()
    checkpoint = torch.load(pretrained_model)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    return model


def load_wrn_model(pretrained_model, num_classes, depth, widen_factor, dropRate):
    model = wrn.wrn(num_classes=num_classes, depth=depth, widen_factor=widen_factor, dropRate=dropRate)#.cuda()

    model = torch.nn.DataParallel(model, [0]).cuda()
    checkpoint = torch.load(pretrained_model)
    model.load_state_dict(checkpoint['state_dict'])

    # print(model.state_dict().keys())
    model.eval()
    return model



def load_mnist(root):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root, train=True, download=True,
        transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root, train=False, download=True,
        transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=1, shuffle=False)
    return train_loader, test_loader


def load_resnet_sub_models(root, layers, net):
    models = []
    for index, layer in enumerate(layers):
        print('load sub model:', layer)
        if net == 'resnet50':
            model = load_resnet50_sub_model(root + '/' + layer + '.pth.tar', layer)
        else:
            model = load_resnet_sub_model(root + '/' + layer + '.pth.tar', layer)
        models.append(model)
    return models

def load_resnet_sub_model(pretrained_model, layer):
    if layer == 'res_layer4':
        model = resnet_layer4.resnet101()
        model = torch.nn.DataParallel(model, [0]).cuda()
    elif layer == 'res_layer3':
        model = resnet_layer3.resnet101()
        model = torch.nn.DataParallel(model, [0]).cuda()
    elif layer == 'res_layer2':
        model = resnet_layer2.resnet101()
        model = torch.nn.DataParallel(model, [0]).cuda()
    elif layer == 'res_layer1':
        model = resnet_layer1.resnet101()
        model = torch.nn.DataParallel(model, [0]).cuda()
    elif layer == 'res_block8':
        model = resnet_layer3_block8.resnet101()
        model = torch.nn.DataParallel(model, [0]).cuda()
    elif layer == 'res_block16':
        model = resnet_layer3_block16.resnet101()
        model = torch.nn.DataParallel(model, [0]).cuda()
    checkpoint = torch.load(pretrained_model)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def load_resnet50_sub_model(pretrained_model, layer):
    if layer == 'res_layer4':
        model = resnet_layer4.resnet50()
        model = torch.nn.DataParallel(model, [0]).cuda()
    elif layer == 'res_layer3':
        model = resnet_layer3.resnet50()
        model = torch.nn.DataParallel(model, [0]).cuda()
    elif layer == 'res_layer2':
        model = resnet_layer2.resnet50()
        model = torch.nn.DataParallel(model, [0]).cuda()
    elif layer == 'res_layer1':
        model = resnet_layer1.resnet50()
        model = torch.nn.DataParallel(model, [0]).cuda()
    checkpoint = torch.load(pretrained_model)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def load_resnet50_model(pretrained=True):
    model = resnet.resnet50(pretrained)
    model = torch.nn.DataParallel(model, [0]).cuda()
    model.eval()
    return model

def load_resnet_model(pretrained=True):
    model = resnet.resnet101(pretrained)
    model = torch.nn.DataParallel(model, [0]).cuda()
    model.eval()
    return model

def load_weight_models(net, device, root, layers, cols, nclass):
    #cols = [1176, 400, 120, 84]

    models = []
    for index, layer in enumerate(layers):
        model = load_weight_model(layer, device,
        root + '/' + layer + '.pth', cols[index], nclass)
        models.append(model)

    return models


def remove_module_in_state_dict(filepath):
    state_dict = torch.load(filepath)['state_dict']
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
                # load params
    return new_state_dict


def load_weight_model(layer, device, pretrained_model, model_col, nclass):
    model = FC(model_col, nclass).to(device)
    print('load ' + layer)

        # Load the pretrained model
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()
    return model

def load_model(net, device, pretrained_model, dataset):
    if net == 'lenet5':
        model = LeNet5().to(device)
    elif net == 'lenet4':
        model = LeNet4().to(device)
    elif net == 'dnn2':
        model = NN().to(device)
    elif net == 'lenet5_weight':
        model = FC().to(device)
    elif net == 'vgg16' and dataset == 'cifar10':
        model = VGG16(10).to(device)
    elif net == 'vgg16' and dataset == 'cifar100':
        model = VGG16(100).to(device)
    elif net == 'wrn' and dataset == 'cifar10':
        model = load_wrn_model(pretrained_model, 10, 28, 10, 0.3)
    elif net == 'dense' and dataset == 'cifar10':
        model = load_dense_model(pretrained_model, 10, 100, 12, 2, 0)
    elif net == 'dense' and dataset == 'cifar100':
        model = load_dense_model(pretrained_model, 100, 100, 12, 2, 0)
    elif net == 'resnext' and dataset == 'cifar10':
        model = load_resnext_model(pretrained_model, 8, 10, 29, 4, 0)
    elif net == 'resnext' and dataset == 'cifar100':
        model = load_resnext_model(pretrained_model, 8, 100, 29, 4, 0)
    print(net)

        # Load the pretrained model
    if net == 'vgg16':
        #print(torch.load(pretrained_model, map_location='cpu')['state_dict'])
        model.load_state_dict(remove_module_in_state_dict(pretrained_model))
    elif net == 'dense' or net == 'resnext' or net == 'wrn':
        # model.load_state_dict(remove_module_in_state_dict(pretrained_model))
        pass
    else:
        model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()
    return model

