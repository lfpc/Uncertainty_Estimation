from torch_data import cifar

import os
import numpy as np

from PIL import Image
from torch.utils.data import Subset,Dataset, DataLoader
from torchvision import datasets
import random

class CorruptedDataset(datasets.VisionDataset):
    corruptions = ['natural',
                    'gaussian_noise',
                    'shot_noise',
                    'speckle_noise',
                    'impulse_noise',
                    'defocus_blur',
                    'gaussian_blur',
                    'motion_blur',
                    'zoom_blur',
                    'snow',
                    'fog',
                    'brightness',
                    'contrast',
                    'elastic_transform',
                    'pixelate',
                    'jpeg_compression',
                    'spatter',
                    'saturate',
                    'frost']

    def __init__(self, data_dir :str,im_size: tuple, names:list = corruptions,levels:tuple = (1,2,3,4,5),
                 transform= None, target_transform = None, natural_data = None):
        super().__init__(
                   data_dir, transform=transform,
                   target_transform=target_transform)
        im_size = (0,) + im_size
        self.data = np.empty(im_size).astype(np.uint8)
        self.targets = np.array([])
        
        for name in names:
            assert name in self.corruptions
            if name == 'natural' and natural_data is not False:
                data = natural_data.data
                targets = natural_data.targets
                self.data = np.concatenate((self.data,data))
                self.targets = np.concatenate((self.targets,targets))
            else: 
                data_path = os.path.join(data_dir, name + '.npy')
                target_path = os.path.join(data_dir, 'labels.npy')
                data = np.load(data_path)
                targets = np.load(target_path)
                for l in levels:
                    l1 = (l-1)*10000
                    l2 = l*10000
                    self.data = np.concatenate((self.data,data[l1:l2]))
                    self.targets = np.concatenate((self.targets,targets[l1:l2]))
            
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, np.int64(targets)
    
    def __len__(self):
        return len(self.data)


class Cifar10C(CorruptedDataset):
    classes = cifar.Cifar10.classes
    n_classes = 10

    def __init__(self, data_dir :str, names:list = CorruptedDataset.corruptions,levels:tuple = (1,2,3,4,5),
                 transform=cifar.Cifar10.transforms_test, target_transform=None, natural_data = None):
        if 'natural' in names:
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
        if 'natural' in names:
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