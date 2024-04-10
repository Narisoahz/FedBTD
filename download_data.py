from torchvision.datasets import MNIST,  SVHN, FashionMNIST, CIFAR100, CIFAR10 , ImageFolder
import torchvision.transforms as transforms
import numpy as np
import torch
#下载数据集
def download_data(dataset, datadir,transform=None):
    if dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.5), (0.5))])
        train_ds = MNIST(datadir, train=True, transform=transform, download=True)
        test_ds = MNIST(datadir, train=False, transform=transform, download=True)
    elif dataset == 'fmnist':
        transform = transforms.Compose([transforms.ToTensor()])
        train_ds = FashionMNIST(datadir, train=True, transform=transform, download=True)
        test_ds = FashionMNIST(datadir, train=False, transform=transform, download=True)
    elif dataset == 'cifar10':
        # transform = transforms.Compose([transforms.ToTensor()])
        transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        train_ds = CIFAR10(datadir, train=True, transform=transform_train, download=True)
        test_ds = CIFAR10(datadir, train=False, transform=transform_test, download=True)
        train_ds.data = torch.tensor(train_ds.data)
        test_ds.data = torch.tensor(test_ds.data)
        train_ds.targets = torch.tensor(train_ds.targets)
        test_ds.targets = torch.tensor(test_ds.targets)
    elif dataset == 'cifar100':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_ds = CIFAR100(datadir, train=True, transform=transform, download=True)
        test_ds = CIFAR100(datadir, train=False, transform=transform, download=True)
        train_ds.data = torch.tensor(train_ds.data)
        test_ds.data = torch.tensor(test_ds.data)
        train_ds.targets = torch.tensor(train_ds.targets)
        test_ds.targets = torch.tensor(test_ds.targets)
    elif dataset == 'svhn':
        transform = transforms.Compose([transforms.ToTensor()])
        train_ds = SVHN(datadir, 'train', transform=transform, download=True)
        test_ds = SVHN(datadir, 'test', transform=transform, download=True)
    elif dataset == 'tinyimagenet':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(32)])
        train_ds = ImageFolder(datadir+'tiny-imagenet-200/train/', transform=transform)
        test_ds = ImageFolder(datadir+'tiny-imagenet-200/val/', transform=transform)

    return train_ds, test_ds
    
if __name__ == '__main__':
    train_ds, test_ds = download_data('mnist', './data/')
    print(test_ds.data.shape)