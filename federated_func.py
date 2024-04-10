import torch
import torch.nn as nn
from utils import compute_accuracy
import torch.optim as optim
import numpy as np
from BTD import *
import copy

def local_test_net_btd(args, nets, client_num, client_test_dl_list, device, logger):
    acc = []
    for idx in range(client_num):
        test_acc = compute_accuracy(args, nets[idx], client_test_dl_list[idx], args.device)
        acc.append(test_acc)
        logger.info('>> Personalized Model in Client {idx}\'s test accuracy: {number}' .format(idx=idx,number=test_acc))
    logger.info('>> average test accuracy: {number}' .format(number=sum(acc)/len(acc)))

def local_test_net(args, global_model, client_num, client_test_dl_list, device, logger):
    acc = []
    for idx in range(client_num):
        test_acc = compute_accuracy(args, global_model, client_test_dl_list[idx], args.device)
        acc.append(test_acc)
        logger.info('>> Global Model in Client {idx}\'s test accuracy: {number}' .format(idx=idx,number=test_acc))
    logger.info('>> average test accuracy: {number}' .format(number=sum(acc)/len(acc)))

def local_train_net(args, nets, client_num, client_train_dl_list, device, logger, round):
    if round == 0:
        epochs = 10
    else:
        epochs = 10
    for net_id, net in nets.items():
        net.to(device)
        trainacc = train_net(args, net_id, net, client_train_dl_list[net_id], epochs, args.lr, args.optimizer, device, logger)

def local_train_net_fedprox(args, nets, global_net, client_num, client_train_dl_list, device, logger, round):
    if round == 0:
        epochs = 10
    else:
        epochs = 10
    for net_id, net in nets.items():
        net.to(device)
        trainacc = train_net_fedprox(args, net_id, net, global_net, client_train_dl_list[net_id], epochs, args.lr, args.optimizer, device, logger)

def local_train_net_scaffold(args, nets, global_net, client_num, client_train_dl_list, device, logger, round, c_nets, c_global):
    c_global.to(device)
    global_net.to(device)
    total_delta = copy.deepcopy(global_net.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    if round == 0:
        epochs = 10
    else:
        epochs = 10
    for net_id, net in nets.items():
        net.to(device)
        c_nets[net_id].to(device)
        trainacc, c_delta_para = train_net_scaffold(args, net_id, net, global_net, client_train_dl_list[net_id], epochs, args.lr, args.optimizer, device, logger, c_nets[net_id], c_global)
        for key in total_delta:
            total_delta[key] += c_delta_para[key]

    for key in total_delta:
        total_delta[key] /= client_num
    c_global_para = c_global.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)

def train_net(args, net_id, net, train_dataloader, epochs, lr, optimizer, device, logger):
    logger.info('Training network %s' % str(net_id))
    if optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)
                optimizer.zero_grad()
                # x.requires_grad = True
                # target.requires_grad = False
                x = x.float()
                target = target.long()
                if args.dataset == 'mnist':
                    x = x.unsqueeze(1)
                elif args.dataset == 'cifar10':
                    x = x.permute(0,3,1,2)
                elif args.dataset == 'cifar100':
                    x = x.permute(0,3,1,2)
                out = net(x)
                if len(out.data.shape)==1:
                    out=out.unsqueeze(0)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()

                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    train_acc = compute_accuracy(args, net, train_dataloader, device=device)
    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info(' ** Training complete **')
    return train_acc

def train_net_fedprox(args, net_id, net, global_net, train_dataloader, epochs, lr, optimizer, device, logger):
    logger.info('Training network %s' % str(net_id))
    if optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]
    
    mu = 0.01
    global_weight_collector = list(global_net.to(device).parameters())

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                # x.requires_grad = True
                # target.requires_grad = False
                x = x.float()
                target = target.long()
                if args.dataset == 'mnist':
                    x = x.unsqueeze(1)
                elif args.dataset == 'cifar10':
                    x = x.permute(0,3,1,2)
                elif args.dataset == 'cifar100':
                    x = x.permute(0,3,1,2)
                out = net(x)
                if len(out.data.shape)==1:
                    out=out.unsqueeze(0)
                loss = criterion(out, target)
                #for fedprox
                fed_prox_reg = 0.0
                for param_index, param in enumerate(net.parameters()):
                    fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
                loss += fed_prox_reg

                loss.backward()
                optimizer.step()

                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    train_acc = compute_accuracy(args, net, train_dataloader, device=device)
    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info(' ** Training complete **')
    return train_acc

def train_net_scaffold(args, net_id, net, global_net, train_dataloader, epochs, lr, optimizer, device, logger, c_local, c_global):
    logger.info('Training network %s' % str(net_id))
    if optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]
    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()
    cnt = 0
    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                # x.requires_grad = True
                # target.requires_grad = False
                x = x.float()
                target = target.long()
                if args.dataset == 'mnist':
                    x = x.unsqueeze(1)
                elif args.dataset == 'cifar10':
                    x = x.permute(0,3,1,2)
                elif args.dataset == 'cifar100':
                    x = x.permute(0,3,1,2)
                out = net(x)
                if len(out.data.shape)==1:
                    out=out.unsqueeze(0)
                loss = criterion(out, target)
                #for scaffold
                loss.backward()
                optimizer.step()
                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)
                cnt += 1

                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
    #scaffold
    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_net.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)

    train_acc = compute_accuracy(args, net, train_dataloader, device=device)
    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info(' ** Training complete **')
    return train_acc, c_delta_para