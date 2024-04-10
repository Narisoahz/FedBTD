import torch
import torch.nn as nn
import torch.nn.functional as F


def cat_to_tensor(nets,client_num):
    for idx in range(client_num):
        net_para = nets[idx].cpu().state_dict()
        if idx == 0:
            #添加conv层
            conv1_clients_weight_tensor = net_para['conv1.weight'].unsqueeze(4)
            conv2_clients_weight_tensor = net_para['conv2.weight'].unsqueeze(4)

            fc1_clients_weight_tensor = net_para['fc1.weight'].unsqueeze(2)
            fc2_clients_weight_tensor = net_para['fc2.weight'].unsqueeze(2)
            fc3_clients_weight_tensor = net_para['fc3.weight'].unsqueeze(2)
        else:
            conv1_clients_weight_tensor = torch.cat((conv1_clients_weight_tensor, net_para['conv1.weight'].unsqueeze(4)),4)
            conv2_clients_weight_tensor = torch.cat((conv2_clients_weight_tensor, net_para['conv2.weight'].unsqueeze(4)),4)

            fc1_clients_weight_tensor = torch.cat((fc1_clients_weight_tensor,net_para['fc1.weight'].unsqueeze(2)),2)
            fc2_clients_weight_tensor = torch.cat((fc2_clients_weight_tensor,net_para['fc2.weight'].unsqueeze(2)),2)
            fc3_clients_weight_tensor = torch.cat((fc3_clients_weight_tensor,net_para['fc3.weight'].unsqueeze(2)),2)
    return conv1_clients_weight_tensor, conv2_clients_weight_tensor, fc1_clients_weight_tensor, fc2_clients_weight_tensor, fc3_clients_weight_tensor

def btd_parameters(group_number, client_num, device):
    C = torch.zeros(client_num, group_number).to(device)
    #创建mlp层的conv层
    conv_list = []
    for i in range(3):
        cur_conv_list = []
        for r in range(group_number):
            cur_conv_list.append(nn.Conv2d(client_num, 1, 1).to(device))
        conv_list.append(cur_conv_list)
    #创建conv层的conv层
    # conv_list_2 = []
    # for i in range(2):
    #     cur_conv_list = []
    #     for r in range(group_number):
    #         cur_conv_list.append(nn.Conv2d(client_num, 1, 1).to(device))
    #     conv_list_2.append(cur_conv_list)    
    return conv_list, C

def btd(fc_tensor_list, conv_tensor_list, group_number, device, C, conv_list, round):
    fc_tensor_size = len(fc_tensor_list)
    for i in range(fc_tensor_size):
        fc_tensor_list[i] = fc_tensor_list[i].permute(2,0,1).to(device)
    # conv_tensor_size = len(conv_tensor_list)
    # for i in range(conv_tensor_size):
    #     conv_tensor_list[i] = conv_tensor_list[i].permute(4,0,1,2,3).to(device)
    #分组指示矩阵C
    softmax = nn.Softmax(dim=1)
    C.requires_grad = True
    #添加参数组
    params = []
    # for i in range(conv_tensor_size):
    #     for r in range(group_number):
    #         params.append({'params': conv_list_2[i][r].parameters(), 'lr': 0.001})
    for i in range(fc_tensor_size):
        for r in range(group_number):
            params.append({'params': conv_list[i][r].parameters(), 'lr': 0.001})
    params.append({'params': C, 'lr':0.001})
    optimizer = torch.optim.SGD([param for param in params], momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.SGD([param for param in params], momentum=0.9)
    # optimizer = torch.optim.Adam([param for param in params])
    # optimizer = torch.optim.Adagrad([param for param in params])
    if round == 0:
        epochs = 10000
    else:
        epochs = 1000
    for epoch in range(epochs):   
        #计算重构损失
        loss = 0
        C_softmax = softmax(C)
        # for i in range(conv_tensor_size):
        #     N = conv_tensor_list[i].shape[0]
        #     I1 = conv_tensor_list[i].shape[1]
        #     I2 = conv_tensor_list[i].shape[2]
        #     I3 = conv_tensor_list[i].shape[3]
        #     I4 = conv_tensor_list[i].shape[4]
        #     recover_tensor = torch.zeros([I1,I2,I3,I4,N]).to(device)
        #     for r in range(group_number):
        #         E_ir = conv_list_2[i][r](conv_tensor_list[i].reshape(N,I1*I2,I3*I4).unsqueeze(0))
        #         E_ir = E_ir.reshape((1, I1*I2*I3*I4, 1))
        #         C_r = C_softmax[:,r].reshape((1,1,N))
        #         cur_recover_tensor = torch.matmul(E_ir, C_r)
        #         recover_tensor += cur_recover_tensor.reshape((1,I1,I2,I3,I4,N)).squeeze(0)
        #     #求重建差，二范数
        #     loss += torch.norm(conv_tensor_list[i].permute(1,2,3,4,0) - recover_tensor)
        for i in range(fc_tensor_size):
            N = fc_tensor_list[i].shape[0]
            I = fc_tensor_list[i].shape[1]
            J = fc_tensor_list[i].shape[2]
            recover_tensor = torch.zeros([I,J,N]).to(device)
            for r in range(group_number):
                E_ir = conv_list[i][r](fc_tensor_list[i].unsqueeze(0))
                E_ir = E_ir.reshape((1, I*J, 1))
                C_r = C_softmax[:,r].reshape((1,1,N))
                cur_recover_tensor = torch.matmul(E_ir, C_r)
                recover_tensor += cur_recover_tensor.reshape((1,I,J,N)).squeeze(0)
            #求重建差，二范数
            loss += torch.norm(fc_tensor_list[i].permute(1,2,0) - recover_tensor)
        optimizer.zero_grad()  
        loss.backward()
        optimizer.step()
        if epoch%100 ==0:
            print('epoch:%d'%epoch) 
            print('loss:%.3f' %loss)
            print(softmax(C))
            # print(sharpen_softmax(softmax(C), temperature=0.3))
    return softmax(C), C, conv_list

def sharpen_softmax(softmax, temperature):
    sharpened = torch.pow(softmax, 1.0 / temperature)
    sharpened /= sharpened.sum(dim=1, keepdim=True)  # 确保分布总和为1
    return sharpened

def learn_E(fc_tensor_list, group_number, device, C, conv_list, round):
    fc_tensor_size = len(fc_tensor_list)
    for i in range(fc_tensor_size):
        fc_tensor_list[i] = fc_tensor_list[i].permute(2,0,1).to(device)
    #添加参数组
    params = []
    for i in range(fc_tensor_size):
        for r in range(group_number):
            params.append({'params': conv_list[i][r].parameters(), 'lr': 0.001})
    # optimizer = torch.optim.SGD([param for param in params], momentum=0.9, weight_decay = 1e-4)
    # optimizer = torch.optim.Adam([param for param in params])
    optimizer = torch.optim.Adam([param for param in params], weight_decay=1e-4)
    if round == 0 :
        epochs = 2000
    else:
        epochs = 2000
    for epoch in range(epochs):   
        #计算重构损失
        loss = 0
        for i in range(fc_tensor_size):
            N = fc_tensor_list[i].shape[0]
            I = fc_tensor_list[i].shape[1]
            J = fc_tensor_list[i].shape[2]
            recover_tensor = torch.zeros([I,J,N]).to(device)
            for r in range(group_number):
                E_ir = conv_list[i][r](fc_tensor_list[i].unsqueeze(0))
                E_ir = E_ir.reshape((1, I*J, 1))
                C_r = C[:,r].reshape((1,1,N))
                cur_recover_tensor = torch.matmul(E_ir, C_r)
                recover_tensor += cur_recover_tensor.reshape((1,I,J,N)).squeeze(0)
            loss = torch.norm(fc_tensor_list[i].permute(1,2,0) - recover_tensor)
            optimizer.zero_grad()  
            loss.backward()
            optimizer.step()     
        if epoch%100 ==0:
            print('epoch:%d'%epoch) 
            print('loss:%.3f' %loss)
    #记录E_list
    E_list = []
    for i in range(fc_tensor_size):
        N = fc_tensor_list[i].shape[0]
        I = fc_tensor_list[i].shape[1]
        J = fc_tensor_list[i].shape[2]
        cur_E_list = []
        for r in range(group_number):
            E_ir = conv_list[i][r](fc_tensor_list[i].unsqueeze(0))
            E_ir = E_ir.reshape((I, J))
            cur_E_list.append(E_ir)
        E_list.append(cur_E_list)
    return conv_list, E_list