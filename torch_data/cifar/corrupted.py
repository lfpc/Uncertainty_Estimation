from torch_data import cifar
from torch_data.src import CorruptedDataset

import os
import numpy as np

from PIL import Image
from torch.utils.data import Subset,Dataset, DataLoader
from torchvision import datasets
import random




class Cifar10C(CorruptedDataset):
    classes = cifar.Cifar10.classes
    n_classes = 10

    def __init__(self, data_dir :str, names:list = CorruptedDataset.corruptions,levels:tuple = (1,2,3,4,5),
                 transform=cifar.Cifar10.transforms_test, target_transform=None, natural_data = None):
        if 0 in levels:
            if natural_data is None:
                parent_data_dir = os.path.abspath(os.path.join(data_dir,os.pardir))
                natural_data = cifar.Cifar10(data_dir = parent_data_dir).test_data
            elif isinstance(natural_data,str):
                natural_data = cifar.Cifar10(data_dir = natural_data).test_data
        data_dir = os.path.join(data_dir,'CIFAR-10-C')
        super().__init__(
                   data_dir,im_size = (32,32,3), names = names,levels = levels,
                 transform= transform, target_transform = target_transform, natural_data = natural_data)

class Cifar100C(CorruptedDataset):
    classes = cifar.Cifar100.classes
    n_classes = 100

    def __init__(self, data_dir :str, names:list = CorruptedDataset.corruptions,levels:tuple = (1,2,3,4,5),
                 transform=cifar.Cifar100.transforms_test, target_transform=None, natural_data = None):
        if 0 in levels:
            if natural_data is None:
                parent_data_dir = os.path.abspath(os.path.join(data_dir,os.pardir))
                natural_data = cifar.Cifar100(data_dir = parent_data_dir).test_data
            elif isinstance(natural_data,str):
                natural_data = cifar.Cifar100(data_dir = natural_data).test_data
        data_dir = os.path.join(data_dir,'CIFAR-100-C')
        super().__init__(
                   data_dir,im_size = (32,32,3), names = names,levels = levels,
                 transform= transform, target_transform = target_transform, natural_data = natural_data)


def CIFAR_C_loader(n,batch_size = 100,**kwargs):
    if n == 10:
        dataset = Cifar10C(**kwargs)
    elif n == 100:
        dataset = Cifar100C(**kwargs)
    loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2,pin_memory=True)
    return loader


def extract_subset(dataset, num_subset :int, random_subset :bool):
    if random_subset:
        random.seed(0)
        indices = random.sample(list(range(len(dataset))), num_subset)
    else:
        indices = [i for i in range(num_subset)]
    return Subset(dataset, indices)