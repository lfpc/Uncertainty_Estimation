import cifar_data

import os
import numpy as np

from PIL import Image
from torch.utils.data import Subset,Dataset, DataLoader
from torchvision import datasets
import random
import torchvision.transforms as transforms




class CIFAR10C(Dataset):
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
    cifar_data.Cifar10.classes
    n_classes = 10  

    def __init__(self, root :str = "data", names:list = corruptions,levels:tuple = (1,2,3,4,5),
                 transform=cifar_data.Cifar10.transforms_test, target_transform=transforms.ToTensor(), natural_data = None):
        #super().__init__(
        #           root, transform=transform,
        #           target_transform=target_transform
        #        )
        self.transform = transform
        self.target_transform = target_transform
        self.data = np.empty((0,32, 32, 3))
        self.targets = np.array([])
        
        for name in names:
            assert name in self.corruptions
            if name == 'natural':
                if natural_data is None:
                    natural_data = root
                if isinstance(natural_data,str):
                    natural_data = cifar_data.Cifar10(data_dir = natural_data)
                    data = natural_data.test_data.data
                    targets = natural_data.test_data.targets
                else:
                    data = natural_data.data
                    targets = natural_data.targets
                self.data = np.concatenate((self.data,data))
                self.targets = np.concatenate((self.targets,targets))
            else: 
                data_path = os.path.join(root,'CIFAR-10-C', name + '.npy')
                target_path = os.path.join(root,'CIFAR-10-C', 'labels.npy')
                
                data = np.load(data_path)
                targets = np.load(target_path)
                for l in levels:
                    l1 = (l-1)*10000
                    l2 = l*10000
                    self.data = np.concatenate((self.data,data[l1:l2]))
                    self.targets = np.concatenate((self.targets,targets[l1:l2]))
            
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        #img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img.float(), targets.float()
    
    def __len__(self):
        return len(self.data)

class CIFAR100C(Dataset):
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
    classes = cifar_data.Cifar100.classes
    n_classes = 100

    def __init__(self, root :str = "data", names:list = corruptions,levels:tuple = (1,2,3,4,5),
                 transform=cifar_data.Cifar100.transforms_test, target_transform=transforms.ToTensor(), natural_data = None):
        #super().__init__(
        #        root, transform=transform,
        #        target_transform=target_transform
        #    )
        self.transform = transform
        self.target_transform = target_transform
        self.data = np.empty((0,32, 32, 3),dtype=float)
        self.targets = np.array([],dtype=float)
        for name in names:
            assert name in self.corruptions
            if name == 'natural':
                if natural_data is None:
                    natural_data = root
                if isinstance(natural_data,str):
                    natural_data = cifar_data.Cifar100(data_dir = natural_data)
                    data = natural_data.test_data.data
                    targets = natural_data.test_data.targets
                else:
                    data = natural_data.data
                    targets = natural_data.targets
                self.data = np.concatenate((self.data,data),dtype = float)
                self.targets = np.concatenate((self.targets,targets),dtype = float)
            else:
                data_path = os.path.join(root,'CIFAR-100-C', name + '.npy')
                target_path = os.path.join(root,'CIFAR-100-C', 'labels.npy')
                
                data = np.load(data_path)
                targets = np.load(target_path)
                for l in levels:
                    l1 = (l-1)*10000
                    l2 = l*10000
                    self.data = np.concatenate((self.data,data[l1:l2]))
                    self.targets = np.concatenate((self.targets,targets[l1:l2]))
        
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        #img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img.float(), targets.float()
    
    def __len__(self):
        return len(self.data)


def CIFAR_C_loader(n,batch_size = 100,**kwargs):
    if n == 10:
        dataset = CIFAR10C(**kwargs)
    elif n == 100:
        dataset = CIFAR100C(**kwargs)
    loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    return loader


def extract_subset(dataset, num_subset :int, random_subset :bool):
    if random_subset:
        random.seed(0)
        indices = random.sample(list(range(len(dataset))), num_subset)
    else:
        indices = [i for i in range(num_subset)]
    return Subset(dataset, indices)