import os
import argparse
import numpy as np

import torch
import utils
import math
from sklearn.metrics import roc_auc_score

def compute_sum(ds_root, layers, labels, type, type2, layer_weights, dataset):
    final = [] #torch.zeros((1,3))
    start = 0#30000
    end = 50000#40000
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
    auc_y = []
    auc_score = []
    sum = 0
    for index in range(50000):
        result = results[index]
        #print(result)
        if(index%100 == 0):
            print(index)
        sums = torch.zeros(1, labels)
        out_softmax = torch.load(ds_root+ 'out/' + str(index) + '_' + type2 + '_out_softmax.pt')
        out_softmax = utils.log_softmax_to_softmax(out_softmax.detach()).cpu()
        if(index<10):
            print(out_softmax.max().item())

        pred_ind = out_softmax.max(1, keepdim=True)[1].item()
        count = 0
        our_conf = 0.0
        flag = 0
        ss = 0
        bonus = 1
        validity_layer = []
        our_conf_score = 0.0
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
        # sorted, index = torch.sort(sums, descending=True)
        # print(result[2], sums)
        # out_softmax = torch.load(ds_root+ folder + str(i) + '_' + type2 + '_out_softmax.pt')
        # print(out_softmax.max())
        # out_softmax = utils.log_softmax_to_softmax(out_softmax.detach()).cpu()
        # print(out_softmax.max())
        # print(index)
        # print(result)
        our_conf_score = calculate_conf(validity_layer, layer_weights)

        #base = sums[0][int(result[0])]
        #if type == 'dists':

        #    sums = torch.nn.functional.log_softmax(sums)
        #    sums = math.e**sums
        #    base = sums[0][int(result[0])]
        #sorted, index = torch.sort(sums, descending=True)
        #sum = sums.sum()
        #sum = base / sum

        #sum = our_conf_score

        flag = 1
        if result[2] == 'incorrect':
            flag = 0
        auc_y.append(flag)
        auc_score.append(our_conf_score)
        v = [index, our_conf_score, out_softmax.max().item(), flag]

        v = [index, our_conf_score, out_softmax.max().item(), int(result[0]), int(result[1]), flag]

        #v.extend(sum_layer)
        #v.extend(sum_layer_max)
        #v.extend(sum_layer_max_value)
        final.append(v)
        del result,  v
        del sum_layer[:]
        del sum_layer_max[:]
        del sum_layer_max_value[:]
    torch.cuda.empty_cache()
    utils.save_tensor(final, ds_root + 'out_' + type  + '_' + type2 + '_{}_{}.pt'.format('_'.join(layers), str(start)))
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
        for i in range(len(weight_x)):
            layer_weight[i] = math.exp(para_beta * weight_x[i] ) * para_alpha + 1
    return layer_weight


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Embedding extraction module')
    parser.add_argument('--dataset', default='imagenet',
                        help='dataset (default=imagenet)')
    parser.add_argument('--type', default='softmax',
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
    values = 10000
    type = args.type
    weightformula = args.weightformula
    para_alpha = args.para_alpha
    para_beta = args.para_beta

    root = '/data/xujw/anatomy/'
    type2 = 'clean'
    ds_root = root + dataset + '/tensor_pub/'
    results = torch.load(ds_root + 'results.pt')
    labels = 1000
    values = 50000
    layers = ['res_layer1', 'res_layer2', 'res_block8', 'res_block16', 'res_layer3', 'res_layer4', 'out']
    #layer_weights = [0.05,0.1,0.15,0.2,0.2,0.1,1]
    # folders = ['10000/','20000/', '30000/', '40000/', '50000/']
    #print(layer_weights)
    weight_x = [1,2,3,4,5,6,7]
    layer_weights = output_weight(weight_x, weightformula, float(para_alpha), float(para_beta))
    compute_sum(ds_root, layers, labels, type, type2, layer_weights, dataset)
    # show_results(ds_root, values, 'dists', 1)
    # show_results(ds_root, values, type, 1, type2, layers, layer_weights)

    # show_results(ds_root, values, type, 2, type2, layers, layer_weights)
    # generate_raw_data(root, dataset, values, type)
