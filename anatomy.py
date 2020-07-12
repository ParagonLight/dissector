# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn.functional as F
import torchvision

import os
import argparse
import numpy as np
import utils
import attacks.fgsm as fgsm



def visualize_weight(model, layer, root, dataset):
    if dataset.startswith('imagenet'):
        w = model.module.fc.weight.detach()
    else:
        w = model.fc.weight.detach()
    utils.save_tensor(w, '{}/{}_w.pt'.format(root, layer))
    n_cols = 8
    if layer == 'conv1':
        w = w.view(60, 1, 14, 14)
        n_cols = 6
    elif layer == 'conv2':
        w = w.view(160, 1, 5, 5)
        n_cols = 8
    else:
        n_cols = 8
    # torchvision.utils.save_image(w,  '{}/{}_w.png'.format(root, layer),
    #                          n_cols, normalize=True, range=(0,1), padding=4)
    del w


def compute_distances_layer(index, x, model, layer, image_type, root, dataset):
    if dataset.startswith('imagenet'):
        if layer == 'res_layer1':
            x = model.module.avgpool(x)
        x = x.flatten()
        dists = model.module.fc(x)
        w = model.module.fc.weight.detach()
        bias = model.module.fc.bias.view(-1,1)
    softmax = utils.softmax(dists)
    softmax = torch.nn.Softmax()(softmax)
    init_pred = softmax.max(0, keepdim=True)[1] # get the index of the max log-probability
    norms = torch.norm(w, 2, 1)
    dists = torch.div(dists, norms)
    del w
    return dists, softmax

def point_to_plane(x, w, bias):
    return np.linalg.norm((np.cross(x, w) + bias) / np.linalg.norm(w))


def compute_distances(img_index, layers, weight_models,
                    embeddings, image_type, root, dataset):
    for index, layer in enumerate(layers):
        embd = embeddings[layer]
        weight_model = weight_models[index]
        embd_vector = embd.data
        layer_dists, layer_softmax = compute_distances_layer(img_index, embd_vector,
                                              weight_model, layer, image_type, root + '/img', dataset)
        utils.save_tensor(layer_dists, root + '/tensor_pub/' + layer + '/' + str(img_index)
                          + '_' + image_type + '_' + layer + '_dists.pt')
        utils.save_tensor(layer_softmax, root + '/tensor_pub/' + layer + '/' + str(img_index)
                          + '_' + image_type + '_' + layer + '_softmax.pt')
        del layer_dists, layer_softmax
    return None, None
    # return dists, softmaxs

def anatomy(model, weight_models, test_loader, device, root, dataset, layers):
    dataset_root = root + '/' + dataset
    img_root = dataset_root + '/img'
    tensor_root = dataset_root + '/tensor'
    index = -1
    incorrect_count = 0

    results = []

    softmax_conf = None
    preds = None
    gt = None
    corrects = None

    w = model.module.fc.weight.detach()
    bias = model.module.fc.bias.view(-1,1)
    norms = torch.norm(w, 2, 1)
    for i, weight_model in enumerate(weight_models):
        visualize_weight(weight_model, layers[i], dataset_root + '/weight', dataset)


    for data_origin, target_origin in test_loader:
        index += 1
        # Send the data and label to the device

        if dataset.startswith('imagenet'):

            target_origin = target_origin.cuda(async=True)
            data = torch.autograd.Variable(data_origin).cuda()
            target = torch.autograd.Variable(target_origin).cuda()
        else:
            data, target = data_origin.to(device), target_origin.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output, embeddings = model(data)

        conf, indexes = torch.max(embeddings['out'].data, 1)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        #if init_pred.item() == target.item():# and index > thres:
            #break
        #    continue
        #if index > thres:
        #    break

        correct = 'correct'
        if init_pred.item() != target.item():
            correct = 'incorrect'
            incorrect_count += 1


        if softmax_conf is None:
            softmax_conf = conf
        else:
            softmax_conf = torch.cat((softmax_conf, conf))

        if preds is None:
            preds = indexes
        else:
            preds = torch.cat((preds, indexes))
        if gt is None:
            gt = target
        else:
            gt = torch.cat((gt, target))
        if corrects is None:
            corrects = indexes.eq(target)
        else:
            corrects = torch.cat((corrects, indexes.eq(target)))

        # print(softmax_conf)
        # print(preds)
        # print(gt)
        # print(corrects)

        print('Image index:', index)



        #utils.save_tensors_in_dict(embeddings, tensor_root,
        #                           str(index) + '_clean', 'embd')
        dists,softmaxs = compute_distances(index, layers, weight_models,
                                  embeddings, 'clean', dataset_root, dataset)
        #utils.save_tensor(dists, tensor_root + '/' + str(index)
        #                  + '_clean_dists.pt')
        #utils.save_tensor(softmaxs, tensor_root + '/' + str(index)
        #                  + '_clean_softmax.pt')
        #print(dists)
        out_values = embeddings['out']
        # print(out_values)
        out_softmax = utils.softmax(out_values, dim=1)
        # print(out_softmax)
        out_dists = torch.div(out_values, norms)
        # print(out_dists)
        utils.save_tensor(out_dists, tensor_root + '_pub/out/' + str(index)
                          + '_clean_out_dists.pt')
        utils.save_tensor(out_softmax, tensor_root + '_pub/out/' + str(index)
                          + '_clean_out_softmax.pt')
        '''
        if dataset == 'mnist':
            print('output before log softmax\n', embeddings['fc3'])
        else:
            print('output before log softmax\n', embeddings['out'])
        '''
        del embeddings, out_softmax, out_dists
        print('Clean pred:', init_pred.item(), 'Label:', target.item(), 'Result:', correct)
        line = [str(index),str(init_pred.item()), str(target.item()), correct]
        results.append(line[1:])
        line = ','.join(line)
        with open(tensor_root + '_pub/result.csv', "a") as myfile:
            myfile.write(line + '\n')
            myfile.flush()
            myfile.close()
        # utils.save_imagenet_image(data, img_root + '_pub/' + str(index) + '_clean.png' )
        torch.cuda.empty_cache()
    torch.save(softmax_conf, dataset + '_softmax_conf_pub.pt')
    torch.save(preds, dataset + '_preds_pub.pt')
    torch.save(gt, dataset + '_gt_pub.pt')
    torch.save(corrects, dataset + '_corrects_pub.pt')

    print(len(softmax_conf))
    print(len(preds))
    print(len(gt))
    print(len(corrects))
    print(incorrect_count)
    utils.save_tensor(results, tensor_root + '_pub/results.pt')

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Embedding extraction module')
    parser.add_argument('--net', default='lenet5',
                        help='DNN name (default=lenet5)')
    parser.add_argument('--root', default='data',
                        help='rootpath (default=data)')
    parser.add_argument('--dataset', default='imagenet',
                        help='dataset (default=imagenet)')
    parser.add_argument('--layer-info', default='layer_info',
                        help='layer-info (default=layer_info)')
    parser.add_argument('--embeddings', default='lenet5_mnist.embd',
                        help='embeddings (default=lenet5_mnist.embd)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--gpu-id', default='1', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N',
                        help='should be 1')
    args = parser.parse_args()
    use_cuda=True
    # Define what device we are using
    #print("CUDA Available: ",torch.cuda.is_available())

    #device = torch.device('cuda',int(args.gpu_id))
    root = args.root
    dataset = args.dataset
    net = args.net
    layers, cols = utils.get_layer_info(root, dataset, net, args.layer_info)
    print(dataset)
    print(root, dataset, net)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    nclass= 1000
    model = utils.load_resnet_model(True)
    weight_models = utils.load_resnet_weight_models(utils.get_model_root(root,
    dataset, net), layers, net)
    test_loader = utils.load_imagenet_test(args.batch_size, args.workers)
    anatomy(model, weight_models, test_loader, None, root,
        dataset, layers)
if __name__ == '__main__':
    main()
