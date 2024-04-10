from model import *
import numpy as np
import os
import torch
import torch.nn as nn
import copy

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def init_nets(args):
    client_num = args.client_num
    nets = {idx: None for idx in range(client_num)}
    
    if args.model == 'cnn':
        if args.dataset == 'cifar10':
            n_classes = 10
            global_model = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
        elif args.dataset == 'mnist':
            n_classes = 10
            global_model = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
        if args.dataset == 'svhn':
            n_classes = 10
            global_model = SimpleCquiNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)    
        elif args.dataset == 'cifar100':
            global_model = SimpleCNNCIFAR100(input_dim=(32 * 5 * 5), hidden_dims=[128, 100], output_dim=100)
    elif args.model == 'resnet':
        if args.dataset == 'cifar10':
            n_classes = 10
            global_model = ResNet18(n_classes)
        elif args.dataset == 'mnist':
            n_classes = 10
            global_model = ResNet18(n_classes)
    
    for idx in range(client_num):
        nets[idx] = copy.deepcopy(global_model)
    
    return nets, global_model

def compute_accuracy(args, model, dataloader, device):
    was_training = False
    if model.training:
        model.eval()
        was_training = True
    
    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                model.to(device)
                x = x.float()
                target = target.long()
                #修改通道数
                if args.dataset == 'mnist':
                    x = x.unsqueeze(1)
                elif args.dataset == 'cifar10':
                    x = x.permute(0, 3, 1, 2)
                elif args.dataset == 'cifar100':
                    x = x.permute(0, 3, 1, 2) 
                out = model(x)
                if len(out.data.shape)==1:
                    out=out.unsqueeze(0)
                _, pred_label = torch.max(out.data, 1)

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

    if was_training:
        model.train()   

    return correct/float(total)

def count_class_distribution(dataset_label, indices):
    labels = dataset_label[indices]

    class_number_distribution = {}
    unq , unq_cnt = np.unique(labels, return_counts=True)
    for i in range(len(unq)):
        class_number_distribution[unq[i]] = unq_cnt[i] 

    return class_number_distribution