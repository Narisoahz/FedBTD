import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from download_data import *
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

#迪利克雷划分数据集
def dirichlet_split_data(dataset, beta=0.3, num_clients=5):
    min_size = 0
    min_require_size = 10
    x, y = dataset
    K = len(np.unique(y))
    N = len(y)
    #每个客户端的数据量必须大于min_require_size
    while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            #遍历每个类
            for k in range(K):
                idx_k = np.where(y == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, num_clients))
                #控制每个客户端数量不超过 N/num_clients
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
    client_ds_indices_list=[]
    for i in range(num_clients):
        client_ds_indices_list.append(idx_batch[i])
    return client_ds_indices_list

#pathlogical划分数据集
def pathlogical_split_data(dataset, class_per_client=2, num_clients=5):
    min_require_size = 10
    dataidx_map = {}
    
    x, y = dataset
    num_classes = len(np.unique(y))
    idxs = np.array(range(len(y)))
    
    idx_for_each_class = []
    for i in range(num_classes):
        idx_for_each_class.append(idxs[y == i])
    
    class_num_per_client = [class_per_client for _ in range(num_clients)]
    for i in range(num_classes):
        selected_clients = []
        for client in range(num_clients):
            if class_num_per_client[client] > 0:
                selected_clients.append(client)
            selected_clients = selected_clients[:int(num_clients/num_classes*class_per_client)]

        num_all_samples = len(idx_for_each_class[i])
        num_selected_clients = len(selected_clients)
        num_per = num_all_samples / num_selected_clients
        num_samples = np.random.randint(max(num_per/10, min_require_size/num_classes), num_per, num_selected_clients-1).tolist()
        num_samples.append(num_all_samples-sum(num_samples))

        idx = 0
        for client, num_sample in zip(selected_clients, num_samples):
            if client not in dataidx_map.keys():
                dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
            else:
                dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
            idx += num_sample
            class_num_per_client[client] -= 1 

    client_ds_indices_list=[]
    for i in range(num_clients):
        client_ds_indices_list.append(dataidx_map[i])
    #  将客户端3的数据分一部分到其他客户端
    # random.shuffle(client_ds_indices_list[3])
    # split_idx = int(0.9*len(client_ds_indices_list[3]))
    # new_indices = client_ds_indices_list[3][split_idx:]
    # client_ds_indices_list[3] = client_ds_indices_list[3][:split_idx]

    # list_1 = random.sample(range(0, 8), 2)
    # list_2 = random.sample(range(8, 16), 2)
    # # list_3 = random.sample(range(12, 16), 2)
    # list_4 = random.sample(range(16, 20), 2)
    # list_total = list_1 + list_2 + list_3 + list_4
    # each_size = int(len(new_indices) / 8)
    # for idx in list_total:
    #      client_ds_indices_list[idx] = np.append(client_ds_indices_list[idx], new_indices[:each_size], axis=0)
    #      new_indices = new_indices[each_size:]

    # #将客户端7的数据分一部分到其他客户端
    # random.shuffle(client_ds_indices_list[7])
    # split_idx = int(0.8*len(client_ds_indices_list[7]))
    # new_indices = client_ds_indices_list[7][split_idx:]
    # client_ds_indices_list[7] = client_ds_indices_list[7][:split_idx]

    # # list_1 = random.sample(range(0, 4), 2)
    # list_1 = random.sample(range(8, 16), 4)
    # # list_3 = random.sample(range(12, 16), 2)
    # list_2 = random.sample(range(16, 20), 2)
    # list_total = list_1 + list_2 
    # each_size = int(len(new_indices) / 6)
    # for idx in list_total:
    #      client_ds_indices_list[idx] = np.append(client_ds_indices_list[idx], new_indices[:each_size], axis=0)
    #      new_indices = new_indices[each_size:]
    
    # # #将客户端11的数据分一部分到其他客户端
    # # random.shuffle(client_ds_indices_list[11])
    # # split_idx = int(0.5*len(client_ds_indices_list[11]))
    # # new_indices = client_ds_indices_list[11][split_idx:]
    # # client_ds_indices_list[11] = client_ds_indices_list[11][:split_idx]

    # # # list_1 = random.sample(range(0, 4), 2)
    # # # list_2 = random.sample(range(4, 8), 2)
    # # # list_3 = random.sample(range(12, 16), 2)
    # # list_4 = random.sample(range(12, 20), 4)
    # # list_total =  list_4
    # # each_size = int(len(new_indices) / 4)
    # # for idx in list_total:
    # #      client_ds_indices_list[idx] = np.append(client_ds_indices_list[idx], new_indices[:each_size], axis=0)
    # #      new_indices = new_indices[each_size:]

    # # #将客户端15的数据分一部分到其他客户端
    # random.shuffle(client_ds_indices_list[15])
    # split_idx = int(0.8*len(client_ds_indices_list[15]))
    # new_indices = client_ds_indices_list[15][split_idx:]
    # client_ds_indices_list[15] = client_ds_indices_list[15][:split_idx]

    # list_1 = random.sample(range(0, 8), 4)
    # # list_2 = random.sample(range(4, 8), 2)
    # # list_3 = random.sample(range(8, 12), 2)
    # list_2 = random.sample(range(16, 20), 2)
    # list_total = list_1 + list_2 
    # each_size = int(len(new_indices) / 6)
    # for idx in list_total:
    #      client_ds_indices_list[idx] = np.append(client_ds_indices_list[idx], new_indices[:each_size], axis=0)
    #      new_indices = new_indices[each_size:]
    
    # # #将客户端19的数据分一部分到其他客户端
    # random.shuffle(client_ds_indices_list[19])
    # split_idx = int(0.8*len(client_ds_indices_list[19]))
    # new_indices = client_ds_indices_list[19][split_idx:]
    # client_ds_indices_list[19] = client_ds_indices_list[19][:split_idx]

    # list_1 = random.sample(range(0, 8), 4)
    # # list_2 = random.sample(range(4, 8), 2)
    # list_2 = random.sample(range(8, 16), 4)
    # # list_4 = random.sample(range(12, 16), 2)
    # list_total = list_1 + list_2
    # each_size = int(len(new_indices) / 8)
    # for idx in list_total:
    #      client_ds_indices_list[idx] = np.append(client_ds_indices_list[idx], new_indices[:each_size], axis=0)
    #      new_indices = new_indices[each_size:]

    return client_ds_indices_list



def split_data(client_ds_indices_list,train_size = 0.85):
    client_number = len(client_ds_indices_list)
    client_trainds_indices_list = []
    client_testds_indices_list = []
    for idx in range(client_number):
        random.shuffle(client_ds_indices_list[idx])
        split_idx = int(train_size*len(client_ds_indices_list[idx]))
        client_trainds_indices_list.append(client_ds_indices_list[idx][:split_idx])
        client_testds_indices_list.append(client_ds_indices_list[idx][split_idx:])
    return client_trainds_indices_list, client_testds_indices_list

class MyDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

if __name__ == '__main__':
    train_ds, test_ds = download_data('cifar10', './data/')
    train_datasets = dirichlet_split_data(train_ds)
    train_ds = add_gaussian_noise(train_ds, 1 , 0.1)
    train_ds = add_label_noise(train_ds, [1], 'pairflip',0.2)
    train_ds = mask(train_ds, [1], 1, 1)



