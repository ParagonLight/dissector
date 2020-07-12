import os
import argparse
import numpy as np


import torch
import utils

def compute_sum(ds_root, layers, labels, type, type2, layer_weights, dataset):
    final = [] #torch.zeros((1,3))
    start = 0#30000
    end = 10000#40000
    softmax = None
    out_softmax = None
    sums = None
    softmax1 = None
    import math
    base = None
    sum_layer = []
    sum_layer_max = []
    sum_layer_max_value = []
    our_conf = 0.0
    sum = 0
    for index in range(50000):
        result = results[index]
        print(result)
        sums = torch.zeros(1, labels)
        out_softmax = torch.load(ds_root+ 'out/' + str(index) + '_' + type2 + '_out_softmax.pt')
        out_softmax = utils.log_softmax_to_softmax(out_softmax.detach()).cpu()
        pred_ind = out_softmax.max(1, keepdim=True)[1].item()
        count = 0
        our_conf = 0.0
        flag = 0
        ss = 0
        bonus = 1
        for layer in layers:
            softmax = torch.load(ds_root+ layer + '/' + str(index) + '_' + type2 + '_'+layer+'_' + type + '.pt')
            softmax = utils.log_softmax_to_softmax(softmax.detach()).cpu()

            softmax11 = softmax.view(-1,1)
            layer_ind = softmax11.max(0, keepdim=True)[1].item()
            sums += softmax
            softmax = softmax.view(-1,1)
            layer_ind = softmax.max(0, keepdim=True)[1].item()
            layer_value_max = softmax.max().item()
            layer_value_pred_ind = softmax[int(result[0])].item()
            sum_layer.append(layer_value_pred_ind)
            sum_layer_max.append(layer_ind)
            sum_layer_max_value.append(layer_value_max)
            # del softmax1
            #our_conf += layer_value_pred_ind / layer_value_max

            count += 1
        # sorted, index = torch.sort(sums, descending=True)
        # print(result[2], sums)
        # out_softmax = torch.load(ds_root+ folder + str(i) + '_' + type2 + '_out_softmax.pt')
        # print(out_softmax.max())
        # out_softmax = utils.log_softmax_to_softmax(out_softmax.detach()).cpu()
        # print(out_softmax.max())
        # print(index)
        # print(result)
        base = sums[0][int(result[0])]
        if type == 'dists':

            sums = torch.nn.functional.log_softmax(sums)
            sums = math.e**sums
            base = sums[0][int(result[0])]
        sorted, index = torch.sort(sums, descending=True)
        sum = sums.sum()
        sum = base / sum

        flag = 1
        if result[2] == 'incorrect':
            flag = 0
        v = [index, sum.item(), out_softmax.max().item(), int(result[0]), int(result[1]), flag]

        v.extend(sum_layer)
        v.extend(sum_layer_max)
        v.extend(sum_layer_max_value)
        final.append(v)
        del result, sum, base, v
        del sum_layer[:]
        del sum_layer_max[:]
        del sum_layer_max_value[:]
    torch.cuda.empty_cache()
    utils.save_tensor(final, ds_root + 'out_' + type  + '_' + type2 + '_{}_{}.pt'.format('_'.join(layers), str(start)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Embedding extraction module')
    parser.add_argument('--dataset', default='imagenet',
                        help='dataset (default=mnist)')
    parser.add_argument('--type', default='softmax',
                        help='value type (default=softmax)')
    parser.add_argument('--gpu-id', default='1', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    dataset = args.dataset
    values = 10000
    type = args.type
    root = '/data/xjw/anatomy/'
    type2 = 'clean'
    ds_root = root + dataset + '/tensor_pub/'
    results = torch.load(ds_root + 'results.pt')
    labels = 1000
    values = 50000
    layers = ['res_layer1', 'res_layer2', 'res_block8', 'res_block16', 'res_layer3', 'res_layer4', 'out']
    layer_weights = [0.05,0.1,0.15,0.2,0.2,0.1,1]
    # folders = ['10000/','20000/', '30000/', '40000/', '50000/']
    print(layer_weights)
    compute_sum(ds_root, layers, labels, type, type2, layer_weights, dataset)
    # show_results(ds_root, values, 'dists', 1)
    # show_results(ds_root, values, type, 1, type2, layers, layer_weights)

    # show_results(ds_root, values, type, 2, type2, layers, layer_weights)
    # generate_raw_data(root, dataset, values, type)
