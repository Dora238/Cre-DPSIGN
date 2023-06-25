import torch
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
curPath = os.path.abspath(os.path.dirname(__file__))


def getData_mnist(setting, args):
    """
    Load MNIST dataset

    :param setting: 'iid' or 'noniid'
    :param config:  configuration of the method
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_noniid = transforms.Compose ([
        transforms.Resize((28, 28)),
        transforms.Grayscale (num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_batchSize = args.testbatchsize

    if setting == 'iid':
        train_dataset = datasets.MNIST(root=curPath + '/dataset', train=True, download=True,
                                       transform=transform)
    elif setting == 'noniid':
        train_dataset = datasets.ImageFolder(root=curPath + '/dataset_noniid/dataset_noniid/MNIST', transform=transform_noniid)
        class_dataset = [
            Subset(train_dataset, list(range(i * 6000, (i + 1) * 6000)))
            for i in range(10)]

    test_dataset = datasets.MNIST(curPath + '/dataset', train=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=test_batchSize,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batchSize,
                             shuffle=False, num_workers=4, pin_memory=True)

    # evenly split the data into workers
    num_data_node = int(len(train_dataset) / args.num_users)
    if setting == 'iid':
        train_dataset_subset = [
            Subset(train_dataset, list(range(i * num_data_node, (i + 1) * num_data_node)))
            for i in range(args.num_users)]
    elif setting == 'noniid':
        n = args.num_users
        train_dataset_subset1 = []
        train_dataset_subset2 = []
        train_dataset_subset = []
        for i in range(n):
            train_dataset_subset1.append(Subset(class_dataset[int(i/(n/10))],
                                                list(range(int((i % (n/10)) * (30000/n)),
                                                           int(((i % (n/10)) + 1) * (30000/n))))))
            for j in range(10):
                if j == 0:
                    train_dataset_subset2.append(Subset(class_dataset[j],
                                                        list(range(int(3000+i*(3000/n)),
                                                                   int(3000+(i+1)*(3000/n))))))
                    continue
                subset2 = Subset(class_dataset[j], list(range(int(3000+i*(3000/n)), int(3000+(i+1)*(3000/n)))))
                train_dataset_subset2[i] = ConcatDataset([train_dataset_subset2[i], subset2])
            train_dataset_subset.append(ConcatDataset([train_dataset_subset1[i], train_dataset_subset2[i]]))

    return train_dataset_subset, train_loader, test_loader

