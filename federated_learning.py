import torch
import torch.nn as nn
from utils import compute_accuracy
import torch.optim as optim
import numpy as np
from BTD import *
import copy
from federated_func import *
def federated_learning(args, nets, global_model, logger, client_train_dl_list, client_test_dl_list):
    client_num = args.client_num
    if args.alg == 'fedavg':
        for round in range(args.comm_round):
            logger.info("in comm round " + str(round))
            
            global_para = global_model.state_dict()
            for idx in range(client_num):
                nets[idx].load_state_dict(global_para)

            local_train_net(args, nets, client_num, client_train_dl_list, args.device, logger, round)
            #更新全局模型
            total_data_length = sum([len(client_train_dl_list[idx].dataset) for idx in range(client_num)])
            fed_avg_freqs = [len(client_train_dl_list[idx].dataset) / total_data_length for idx in range(client_num)]

            for idx in range(client_num):
                net_para = nets[idx].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            
            global_model.load_state_dict(global_para)
            global_model.to(args.device)
            #测试全局模型在各个客户端测试集的准确率
            local_test_net(args, global_model, client_num, client_test_dl_list, args.device, logger)
    elif args.alg == 'fedprox':
        logger.info("FedProx")
        for round in range(args.comm_round):
            logger.info("in comm round " + str(round))
            
            global_para = global_model.state_dict()
            for idx in range(client_num):
                nets[idx].load_state_dict(global_para)

            local_train_net_fedprox(args, nets, global_model, client_num, client_train_dl_list, args.device, logger, round)
            #更新全局模型
            total_data_length = sum([len(client_train_dl_list[idx].dataset) for idx in range(client_num)])
            fed_avg_freqs = [len(client_train_dl_list[idx].dataset) / total_data_length for idx in range(client_num)]

            for idx in range(client_num):
                net_para = nets[idx].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)
            global_model.to(args.device)
            #测试全局模型在各个客户端测试集的准确率
            local_test_net(args, global_model, client_num, client_test_dl_list, args.device, logger)
    elif args.alg == 'scaffold':
        logger.info("scaffold")
        c_global = copy.deepcopy(global_model)
        
        c_nets = {net_i: copy.deepcopy(c_global) for net_i in range(client_num)}

        for round in range(args.comm_round):
            logger.info("in comm round " + str(round))
            
            global_para = global_model.state_dict()
            for idx in range(client_num):
                nets[idx].load_state_dict(global_para)

            local_train_net_scaffold(args, nets, global_model, client_num, client_train_dl_list, args.device, logger, round , c_nets, c_global)
            #更新全局模型
            total_data_length = sum([len(client_train_dl_list[idx].dataset) for idx in range(client_num)])
            fed_avg_freqs = [len(client_train_dl_list[idx].dataset) / total_data_length for idx in range(client_num)]

            for idx in range(client_num):
                net_para = nets[idx].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)
            global_model.to(args.device)
            #测试全局模型在各个客户端测试集的准确率
            local_test_net(args, global_model, client_num, client_test_dl_list, args.device, logger)
    #其他联邦学习算法
    elif args.alg == 'fedbtd':
        #定义btd分解的优化网络和C
        conv_list, C = btd_parameters(args.group_number, client_num, args.device)
        for round in range(args.comm_round):
            logger.info("in comm round " + str(round))
            local_train_net(args, nets, client_num, client_train_dl_list, args.device, logger, round)
            #将客户端的模型参数叠成张量
            conv1_clients_weight_tensor, conv2_clients_weight_tensor, fc1_clients_weight_tensor, fc2_clients_weight_tensor, fc3_clients_weight_tensor = cat_to_tensor(nets, client_num)
            #获取分组张量C
            group_c, C, conv_list = btd([fc1_clients_weight_tensor, fc2_clients_weight_tensor, fc3_clients_weight_tensor],[conv1_clients_weight_tensor, conv2_clients_weight_tensor], args.group_number, args.device, C, conv_list, round)
            logger.info('>> C: %s', group_c)
            #记录分组情况
            group_idx = np.array(torch.argmax(group_c.cpu(),dim=1))
            cur_group = []
            for r in range(args.group_number):
                a = np.where(group_idx == r)[0]
                if len(a) != 0:
                    cur_group.append(a)
            cur_group = sorted(cur_group, key = (lambda x:x[0]))
            group = {}
            for i, cur_idx in enumerate(cur_group):
                group[i] = cur_idx
                logger.info('>> client {idx} is in group {number}'.format(idx=cur_idx,number=i))
            #固定C为one-hot矩阵，重新训练E的conv层
            max_idx = torch.argmax(group_c.cpu(), dim=1)
            C = torch.zeros_like(group_c).to(args.device)
            C[range(len(group_c)), max_idx] = 1
            C.requires_grad = False
            logger.info('>> C: %s', C)
            #得到各个mlp层的E
            conv_list, E_list = learn_E([fc1_clients_weight_tensor, fc2_clients_weight_tensor, fc3_clients_weight_tensor], args.group_number, args.device, C, conv_list, round)
            group = {}
            for r in range(args.group_number):
                a = np.where(group_idx == r)[0]
                if len(a) != 0:
                    group[r]=a
            for r, client_idx in group.items():
                #更新组内conv层的模型
                total_data_length = sum([len(client_train_dl_list[idx].dataset) for idx in client_idx])
                fed_avg_freqs = [len(client_train_dl_list[idx].dataset) / total_data_length for idx in client_idx]
                global_para = global_model.state_dict()
                time = 0
                for idx in client_idx:
                    net_para = nets[idx].cpu().state_dict()
                    if time == 0:
                        for key in ['conv1.weight','conv2.weight']:
                            global_para[key] = net_para[key] * fed_avg_freqs[time]
                    else:
                        for key in ['conv1.weight','conv2.weight']:
                            global_para[key] += net_para[key] * fed_avg_freqs[time]
                    time += 1
                #更新组内mlp层
                i = 0
                for key in ['fc1.weight','fc2.weight','fc3.weight']:
                    global_para[key] = E_list[i][r]
                    i+=1
                for idx in client_idx:
                    nets[idx].load_state_dict(global_para)
            #测试更新后的本地模型在各个客户端测试集的准确率
            # #本地微调
            # local_train_net(args, nets, client_num, client_train_dl_list, args.device, logger, round)
            local_test_net_btd(args, nets, client_num, client_test_dl_list, args.device, logger)
    elif args.alg == 'fedbtd_test':
        #定义btd分解的优化网络和C
        conv_list, C = btd_parameters(args.group_number, client_num, args.device)
        for round in range(args.comm_round):
            logger.info("in comm round " + str(round))
            local_train_net(args, nets, client_num, client_train_dl_list, args.device, logger, round)
            #将客户端的模型参数叠成张量
            conv1_clients_weight_tensor, conv2_clients_weight_tensor, fc1_clients_weight_tensor, fc2_clients_weight_tensor, fc3_clients_weight_tensor = cat_to_tensor(nets, client_num)
            #获取分组张量C
            group_c, C, conv_list = btd([fc1_clients_weight_tensor, fc2_clients_weight_tensor, fc3_clients_weight_tensor],[conv1_clients_weight_tensor, conv2_clients_weight_tensor], args.group_number, args.device, C, conv_list, round)
            logger.info('>> C: %s', group_c)
            #记录分组情况
            group_idx = np.array(torch.argmax(group_c.cpu(),dim=1))
            cur_group = []
            for r in range(args.group_number):
                a = np.where(group_idx == r)[0]
                if len(a) != 0:
                    cur_group.append(a)
            cur_group = sorted(cur_group, key = (lambda x:x[0]))
            group = {}
            for i, cur_idx in enumerate(cur_group):
                group[i] = cur_idx
                logger.info('>> client {idx} is in group {number}'.format(idx=cur_idx,number=i))
            #固定C为one-hot矩阵，重新训练E的conv层
            max_idx = torch.argmax(group_c.cpu(), dim=1)
            C = torch.zeros_like(group_c).to(args.device)
            C[range(len(group_c)), max_idx] = 1
            C.requires_grad = False
            logger.info('>> C: %s', C)
            #得到各个mlp层的E
            # conv_list, E_list = learn_E([fc1_clients_weight_tensor, fc2_clients_weight_tensor, fc3_clients_weight_tensor], args.group_number, args.device, C, conv_list, round)
            group = {}
            for r in range(args.group_number):
                a = np.where(group_idx == r)[0]
                if len(a) != 0:
                    group[r]=a
            for r, client_idx in group.items():
                #更新组内模型
                total_data_length = sum([len(client_train_dl_list[idx].dataset) for idx in client_idx])
                fed_avg_freqs = [len(client_train_dl_list[idx].dataset) / total_data_length for idx in client_idx]
                global_para = global_model.state_dict()
                time = 0
                for idx in client_idx:
                    net_para = nets[idx].cpu().state_dict()
                    if time == 0:
                        for key in net_para:
                            global_para[key] = net_para[key] * fed_avg_freqs[time]
                    else:
                        for key in net_para:
                            global_para[key] += net_para[key] * fed_avg_freqs[time]
                    time += 1
                for idx in client_idx:
                    nets[idx].load_state_dict(global_para)
            #测试更新后的本地模型在各个客户端测试集的准确率
            # #本地微调
            # local_train_net(args, nets, client_num, client_train_dl_list, args.device, logger, round)
            local_test_net_btd(args, nets, client_num, client_test_dl_list, args.device, logger)
    elif args.alg == 'local_train':
        for round in range(args.comm_round):
            logger.info("in comm round " + str(round))
            local_train_net(args, nets, client_num, client_train_dl_list, args.device, logger, round)
            local_test_net_btd(args, nets, client_num, client_test_dl_list, args.device, logger)

    elif args.alg == 'fedavg_new':
        for round in range(args.comm_round):
            logger.info("in comm round " + str(round))
            
            global_para = global_model.state_dict()
            if round == 0:
                for idx in range(client_num):
                    nets[idx].load_state_dict(global_para)

            local_train_net(args, nets, client_num, client_train_dl_list, args.device, logger, round)
            #更新全局模型
            total_data_length = sum([len(client_train_dl_list[idx].dataset) for idx in range(client_num)])
            fed_avg_freqs = [len(client_train_dl_list[idx].dataset) / total_data_length for idx in range(client_num)]

            for idx in range(client_num):
                net_para = nets[idx].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
                        
            for idx in range(client_num):
                net_para = nets[idx].cpu().state_dict()
                for key in ['conv1.weight','conv2.weight','fc1.weight','fc2.weight']:
                    net_para[key] = global_para[key]
                nets[idx].load_state_dict(net_para)
            # global_model.load_state_dict(global_para)
            # global_model.to(args.device)
            #测试全局模型在各个客户端测试集的准确率
            local_test_net_btd(args, nets, client_num, client_test_dl_list, args.device, logger)