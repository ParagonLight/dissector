import os
import argparse
import numpy as np

import torch
import utils
import math
from progress.bar import Bar as Bar
from sklearn.metrics import roc_auc_score

def compute_sum(dataset_size, ds_root, layers, dataset_label, output_type, img_type, layer_weights, dataset):
    final = [] 
    softmax = None
    out_softmax = None
    sums = None
    
    labels = int(dataset_label)
    sum_layer = []
    sum_layer_max = []
    sum_layer_max_value = []
    our_conf = 0.0
    auc_y = []
    auc_score = []
    sum = 0
    dataset_size
    bar = Bar('Processing', max=dataset_size)
    for index in range(dataset_size):
        result = results[index]

        sums = torch.zeros(1, labels)
        out_softmax = torch.load(ds_root+ 'out/' + str(index) + '_' + img_type + '_out_softmax.pt')
        out_softmax = utils.log_softmax_to_softmax(out_softmax.detach()).cpu()


        pred_ind = out_softmax.max(1, keepdim=True)[1].item()
        count = 0
        our_conf = 0.0
        flag = 0
        validity_layer = []
        our_conf_score = 0.0
        for layer in layers:
            softmax = torch.load(ds_root+ layer + '/' + str(index) + '_' + img_type + '_'+layer+'_' + output_type + '.pt')
            softmax = utils.log_softmax_to_softmax(softmax.detach()).cpu()

            sums += softmax
            softmax = softmax.view(-1,1)
            layer_ind = softmax.max(0, keepdim=True)[1].item()
            layer_value_max = softmax.max().item()
            layer_value_pred_ind = softmax[int(result[0])].item()
            sum_layer.append(layer_value_pred_ind)
            sum_layer_max.append(layer_ind)
            sum_layer_max_value.append(layer_value_max)

            count += 1

            predict_softmax = layer_value_pred_ind
            max_softmax = layer_value_max # the maximum softmax score for this layer
            max_label = layer_ind # the corresponding label for the maximum softmax score
            score = 0.0
            if pred_ind == max_label:
                secondHighest = torch.sort(softmax, dim = 0,descending=True)[0][1].item()
                score = predict_softmax/(predict_softmax + secondHighest + 1e-100)
            else:
                score = 1.0 - max_softmax / (max_softmax + predict_softmax + 1e-100)
            validity_layer.append(score)

        our_conf_score = calculate_conf(validity_layer, layer_weights)

        flag = 1
        if result[2] == 'incorrect':
            flag = 0
        auc_y.append(flag)
        auc_score.append(our_conf_score)
        v = [index, our_conf_score, out_softmax.max().item(), flag]

        v = [index, our_conf_score, out_softmax.max().item(), int(result[0]), int(result[1]), flag]

        final.append(v)
        del result, v
        del sum_layer[:]
        del sum_layer_max[:]
        del sum_layer_max_value[:]

        bar.suffix  = '({index}/{size}) | Total: {total:} | ETA: {eta:}'.format(
                    index=index + 1,
                    size=dataset_size,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    )
        bar.next()
    bar.finish()
    torch.cuda.empty_cache()
    print("AUC score being: ", roc_auc_score(auc_y,auc_score))


def calculate_conf(validity_layer, layer_weights):
    result = 0.0
    for i in range(len(validity_layer)):
        result += validity_layer[i]*layer_weights[i]
    return result/sum(layer_weights)

def output_weight(weight_x, weightformula, para_alpha, para_beta):
    if weightformula == 'linear':
        return output_weight_linear(weight_x,  para_alpha, para_beta)
    if weightformula == 'log' or weightformula == 'logarithmic':
        return output_weight_log(weight_x, para_alpha, para_beta)
    if weightformula == 'exp' or weightformula == 'exponential':
        return output_weight_exp(weight_x, para_alpha, para_beta)

def output_weight_linear(weight_x,  para_alpha, para_k):
    layer_weight = weight_x
    if para_alpha == 0 and para_k == 0: # y = ax+k
        for i in range(len(weight_x)):
            layer_weight[i] = weight_x[i]
    if para_alpha != 0: #y = x
        for i in range(len(weight_x)):
            layer_weight[i] = para_alpha * weight_x[i] + 1
    return layer_weight

def output_weight_log(weight_x, para_alpha, para_beta):
    layer_weight = weight_x;
    if para_alpha == 0 and para_beta == 0: # y = lnx
        for i in range(len(weight_x)):
            layer_weight[i] = math.log(weight_x[i], math.e)
    if para_alpha != 0 and para_beta == 0: # y = alnx + 1
        for i in range(len(weight_x)):
            layer_weight[i] = math.log(weight_x[i], math.e) * para_alpha + 1
    if para_alpha != 0 and para_beta != 0: # y = aln(bx+1)+1
        for i in range(len(weight_x)):
            layer_weight[i] = math.log(weight_x[i]*para_beta+1)*para_alpha + 1
    return layer_weight

def output_weight_exp(weight_x, para_alpha, para_beta):
    layer_weight = weight_x
    if para_alpha == 0: # y = e ^ (bx)
        for i in range(len(weight_x)):
            layer_weight[i] = math.exp(para_beta * weight_x[i]) # y e^(bx)
    if para_alpha != 0: # y = a e * (bx) + 1
        #for i in range(len(weight_x)):
        for i in range(len(weight_x)):
            layer_weight[i] = math.exp(para_beta * weight_x[i] ) * para_alpha + 1
    return layer_weight


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dissector profile generation module')
    parser.add_argument('--dataset', default='imagenet',
                        help='dataset (default=imagenet)')
    parser.add_argument('--dataset_size', default = '50000',type = int,
                        help='dataset_size (default= 50000)')
    parser.add_argument('--dataset_label',default = '1000', type = int,
                        help='dataset_label (default=1000)')
    parser.add_argument('--net',default = 'resnet101',
                        help = 'DNN name (default=resnet101)')
    parser.add_argument('--root', default = 'data',
                        help='rootpath (default = data)')
    parser.add_argument('--tensor_folder', default = 'tensor_pub',
                            help='tensor_folder(default=tensor_pub)')
    parser.add_argument('--layer_info', default = 'layer_info',
                        help='layer-info (default = layer_info)')
    parser.add_argument('--output_type', default='softmax',
                        help='value type (default=softmax)')
    parser.add_argument('--gpu-id', default='1', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--weightformula', default = 'linear',
                        help='weightformula (default=linear)')
    parser.add_argument('--para_alpha', default = '0', type = str,
                        help = 'para_alpha (default = 0)')
    parser.add_argument('--para_beta', default = '0', type = str,
                        help = 'para_beta (default = 0)')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


    dataset = args.dataset
    dataset_size = args.dataset_size
    dataset_label = args.dataset_label
    net = args.net
    output_type = args.output_type
    weightformula = args.weightformula
    para_alpha = args.para_alpha
    para_beta = args.para_beta

    root = args.root #'/data/xujw/anatomy/'
    img_type = 'clean'
    ds_root = root + dataset + '/'+args.tensor_folder+'/'
    results = torch.load(ds_root + 'results.pt')
    layers, cols = utils.get_layer_info(root,dataset,net,args.layer_info)
    layers.append('out')
    weight_x = list(range(1,len(layers)+1))
    print(weight_x)

    layer_weights = output_weight(weight_x, weightformula, float(para_alpha), float(para_beta))
    compute_sum(dataset_size, ds_root, layers, dataset_label, output_type, img_type, layer_weights, dataset)
