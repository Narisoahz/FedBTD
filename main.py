import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from download_data import *
from data_process import *
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import argparse
import copy
import pickle
import json
import logging
import os
import datetime
from utils import *
from federated_learning import federated_learning

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--load_data', type=bool, default=False, help="True:load data,False:download data and precoss")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/cifar10", help='Log directory path')
    parser.add_argument('--device', type=str, default='cuda:1', help='The device to run the program')

    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    parser.add_argument('--model', type=str, default='cnn', help='model')
    parser.add_argument('--train_batch_size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=32, help='input batch size for testing ')
    #划分方式
    parser.add_argument('--partition', type=str, default='dir', help='pathlogical or practical')
    parser.add_argument('--noniid', type=float, default=0.1, help='non-iid')
    parser.add_argument('--class_per_client', type=int, default=2, help='pathlogical partition')
    #联邦学习设置
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication roun')
    parser.add_argument('--client_num', type=int, default=10, help="client number")
    parser.add_argument('--alg', type=str, default='fedbtd',
                            help='fl algorithms: fedavg/fedprox/scaffold/fednova/moon')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.01)')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    #BTD设置
    parser.add_argument('--group_number', type=int, default=10, help="group number")

    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = get_args()
    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    #修改日志路径
    if args.partition == 'pat' and args.dataset == 'cifar10':
        args.logdir = './logs/cifar10/10client_class2_5group'
    elif args.partition == 'pat' and args.dataset == 'cifar100':
        args.logdir = './logs/cifar100/20client_class60_2group'
    elif args.partition == 'pat' and args.dataset == 'mnist':
        args.logdir = './logs/mnist'
    elif args.partition == 'pat' and args.dataset == 'svhn':
        args.logdir = './logs/svhn/10client_class4_3group' 

    if args.partition == 'dir' and args.dataset == 'cifar10':
        args.logdir = './logs/cifar10/10client_beta1'
    elif args.partition == 'dir' and args.dataset == 'cifar100':
        args.logdir = './logs/cifar100/10client_beta1'
    elif args.partition == 'dir' and args.dataset == 'mnist':
        args.logdir = './logs/mnist'
    elif args.partition == 'dir' and args.dataset == 'svhn':
        args.logdir = './logs/svhn/10client_beta1'    
    #日志设置
    mkdirs(args.logdir)
    argument_path='experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    #通过load_data确定是加载数据还是下载数据
    if not args.load_data:
        #下载数据集
        train_ds, test_ds = download_data(args.dataset, '/home/zsr/data/')
        #合并训练集和测试集 
        dataset_image = []
        dataset_label = []
        # if args.dataset == 'cifar10' or 'cifar100':
        dataset_image.extend(train_ds.data.cpu().detach().numpy())
        dataset_image.extend(test_ds.data.cpu().detach().numpy())
        dataset_label.extend(train_ds.targets.cpu().detach().numpy())
        dataset_label.extend(test_ds.targets.cpu().detach().numpy())
        # elif args.dataset == 'svhn':
        # dataset_image.extend(train_ds.data)
        # dataset_image.extend(test_ds.data)
        # dataset_label.extend(train_ds.labels)
        # dataset_label.extend(test_ds.labels)

        dataset_image = np.array(dataset_image)
        dataset_label = np.array(dataset_label)
        #对客户端数据进行划分，得到客户端数据索引列表
        if args.partition == 'dir':
            client_ds_indices_list = dirichlet_split_data((dataset_image, dataset_label),beta=args.noniid, num_clients=args.client_num)
        elif args.partition == 'pat':
            client_ds_indices_list = pathlogical_split_data((dataset_image, dataset_label),class_per_client=args.class_per_client, num_clients=args.client_num)
        #对客户端数据索引列表进行划分，得到客户端的训练集和测试集
        client_trainds_indices_list, client_testds_indices_list = split_data(client_ds_indices_list, train_size = 0.85)
        
        ##记录数据划分和处理日志
        for idx in range(args.client_num):
            logger.info("#" * 100)
            logger.info('>> Client {idx}'.format(idx=idx))

            logger.info('>> Client {idx}\'s train data number:{number}'.format(idx=idx,number=len(client_trainds_indices_list[idx])))
            logger.info('>> Client {idx}\'s test data number:{number}'.format(idx=idx,number=len(client_testds_indices_list[idx])))          

            class_number_distribution = count_class_distribution(dataset_label,client_trainds_indices_list[idx])
            logger.info("#" * 100)
            logger.info('>> Client {idx}\'s train data class'.format(idx=idx)) 
            for label,number in class_number_distribution.items():
                logger.info('>> class {i}\'s number :{number} '.format(i=label,number=number))
            class_number_distribution = count_class_distribution(dataset_label,client_testds_indices_list[idx])
            logger.info("#" * 100)
            logger.info('>> Client {idx}\'s test data class'.format(idx=idx)) 
            for label,number in class_number_distribution.items():
                logger.info('>> class {i}\'s number :{number} '.format(i=label,number=number))
        #预处理完，将客户端数据集封装成dataloader
        client_train_dl_list = {}
        client_test_dl_list = {}
        for idx in range(args.client_num):
            train_indice = client_trainds_indices_list[idx]
            test_indice = client_testds_indices_list[idx]
            train_dl = DataLoader(dataset=MyDataset(dataset_image[train_indice], dataset_label[train_indice]), batch_size=args.train_batch_size, shuffle=True, drop_last=False, num_workers=4)
            test_dl = DataLoader(dataset=MyDataset(dataset_image[test_indice], dataset_label[test_indice]), batch_size=args.test_batch_size, shuffle=True, drop_last=False, num_workers=4)
            client_train_dl_list[idx] = train_dl
            client_test_dl_list[idx] = test_dl
            #保存到本地
            # with open('./process_data/'+args.dataset+'_client'+str(idx)+'_train.pkl','wb') as f:
            #     pickle.dump(train_dl,f)
            # with open('./process_data/'+args.dataset+'_client'+str(idx)+'_test.pkl','wb') as f:
            #     pickle.dump(test_dl,f)
   
    else:
        #加载数据集
        client_train_dl_list={}
        client_test_dl_list = {}
        for idx in range(args.client_num):
            #保存到本地
            with open('./process_data/'+args.dataset+'_client'+str(idx)+'_train.pkl', 'rb') as f:
                client_train_dl_list[idx] = pickle.load(f)
            with open('./process_data/'+args.dataset+'_client'+str(idx)+'_test.pkl', 'rb') as f:
                client_test_dl_list[idx] = pickle.load(f)
    
    logger.info("#" * 100)
    #开始联邦学习训练
    nets, global_model = init_nets(args)
    # for name,parameters in global_model.named_parameters():
    #     print(name,':',parameters.size())
    federated_learning(args, nets, global_model, logger, client_train_dl_list, client_test_dl_list)
