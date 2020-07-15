# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn.functional as F
import torchvision

import os
import argparse
import numpy as np
import utils

def compute_distances_layer(index, x, model, layer, image_type, dataset):
    if layer == 'res_layer1': # do dimension reduction for such very large layer
        x = model.module.avgpool(x)
    x = x.flatten()
    dists = model.module.fc(x)
    softmax = utils.softmax(dists)
    return softmax

def compute_distances(img_index, layers, sub_models,
                    embeddings, image_type, tensor_root, dataset):
    for index, layer in enumerate(layers):
        embd = embeddings[layer]
        sub_model = sub_models[index]
        embd_vector = embd.data
        layer_softmax = compute_distances_layer(img_index, embd_vector,
                                              sub_model, layer, image_type, dataset)
        utils.save_tensor(layer_softmax, tensor_root + '/' + layer + '/' + str(img_index)
                          + '_' + image_type + '_' + layer + '_softmax.pt')
        del layer_softmax

def anatomy(model, sub_models, test_loader, root, dataset, tensor_folder, layers):
    dataset_root = root + '/' + dataset
    img_root = dataset_root + '/img'
    tensor_root = dataset_root + '/' + tensor_folder
    index = -1

    results = []

    for data_origin, target_origin in test_loader:
        index += 1
        print('Image index:', index)
        # Send the data and label to the device
        target_origin = target_origin.cuda(async=True)
        data = torch.autograd.Variable(data_origin).cuda()
        target = torch.autograd.Variable(target_origin).cuda()

        # Forward pass the data through the model
        output, embeddings = model(data)

        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct = 'correct'
        if init_pred.item() != target.item():
            correct = 'incorrect'

        # extract log softmax for each sub model 
        compute_distances(index, layers, sub_models,
                                  embeddings, 'clean', tensor_root, dataset)
        out_values = embeddings['out']
        out_softmax = utils.softmax(out_values, dim=1)
        utils.save_tensor(out_softmax, tensor_root + '/out/' + str(index)
                          + '_clean_out_softmax.pt')
        del embeddings, out_softmax
        # print('Clean pred:', init_pred.item(), 'Label:', target.item(), 'Result:', correct)
        line = [str(index),str(init_pred.item()), str(target.item()), correct]
        results.append(line[1:])
        line = ','.join(line)
        torch.cuda.empty_cache()

    utils.save_tensor(results, tensor_root + '/results.pt')

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Embedding extraction module')
    parser.add_argument('--net', default='lenet5',
                        help='DNN name (default=lenet5)')
    parser.add_argument('--root', default='data',
                        help='rootpath (default=data)')
    parser.add_argument('--dataset', default='imagenet',
                        help='dataset (default=imagenet)')
    parser.add_argument('--tensor_folder', default='tensor_pub',
                        help='tensor_folder (default=tensor_pub)')
    parser.add_argument('--layer-info', default='layer_info',
                        help='layer-info (default=layer_info)')
    parser.add_argument('--gpu-id', default='1', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N',
                        help='should be 1')
    args = parser.parse_args()
    use_cuda=True
    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())

    root = args.root
    dataset = args.dataset
    net = args.net
    tensor_folder = args.tensor_folder
    layers, cols = utils.get_layer_info(root, dataset, net, args.layer_info)
    print(dataset)
    print(root, dataset, net)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    model = utils.load_resnet_model(pretrained=True)
    sub_models = utils.load_resnet_sub_models(utils.get_model_root(root,
    dataset, net), layers, net)
    test_loader = utils.load_imagenet_test(args.batch_size, args.workers)
    anatomy(model, sub_models, test_loader, root,
        dataset, tensor_folder, layers)
if __name__ == '__main__':
    main()
