import copy
from torch.utils.data import Subset
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from random import randrange
import torch

def split_data(training_data,val_size,val_transforms, method = 'range', seed = None):
    '''to develop'''
    assert method == 'range' or method == 'random' or method == 'idx'
    
    if method == 'random':
        val_size = int(val_size*len(training_data))
        train_size = len(training_data) - val_size
        if seed is not None:
            train_subset, val_subset = random_split(training_data, [train_size, val_size],generator=torch.Generator().manual_seed(seed))
        else:
            train_subset, val_subset = random_split(training_data, [train_size, val_size])
    elif method == 'range':
        train_idx, val_idx = train_test_split(list(range(len(training_data))), test_size=val_size, shuffle = False)
        train_subset = Subset(training_data, train_idx)
        val_subset = Subset(training_data, val_idx)

    val_subset = copy.deepcopy(val_subset)
    val_subset.dataset.transform = val_transforms

    return train_subset, val_subset

def get_dataloaders(training_data,test_data,params,validation_data = None):
    if validation_data is None:
        training_data,validation_data = split_data(training_data)
    train_dataloader = DataLoader(training_data, batch_size=params['train_batch_size'],shuffle = True,pin_memory=True)
    validation_dataloader = DataLoader(validation_data, batch_size=params['train_batch_size'],shuffle = False,pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=params['test_batch_size'],pin_memory=True)
    return train_dataloader, validation_dataloader, test_dataloader

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

    #to develop:
    # show images
    #imshow(torchvision.utils.make_grid(images))
    # print labels
    #print(' '.join(f'{classes[labels[j]]:5s}' for j in range(5)))


def corrupt_label(dataset,noise_size,n_classes,copy_ = True):
    if copy_: dataset = copy.deepcopy(dataset)

    if isinstance(dataset,DataLoader):
        dataset = dataset.dataset
    if isinstance(dataset,Subset):
        targets = dataset.dataset.targets
        idx = dataset.indices[0]
    elif isinstance(dataset,torch.utils.data.Dataset):
        targets = dataset.targets
        idx = 0
    

    for i,label in enumerate(targets[idx:]):
        if i==int((len(dataset)*noise_size)):
            break
        new_label = randrange(n_classes)
        while new_label == label:
            new_label = randrange(n_classes)
        targets[i+idx] = new_label
        
    return dataset

class DataGenerator():

    params = {'train_batch_size':128,'validation_size':0.0,'test_batch_size': 100}
    pre_seed = 42

    def __init__(self,
                    params = params,
                    name = 'DataGenerator',
                    training_data = None,
                    test_data = None,
                    seed = None):

        self.name = name
        if training_data is not None:
            self.training_data = training_data
        if test_data is not None:
            self.test_data = test_data

        self.params = params
        self.generate_dataloaders(seed)

    def generate_dataloaders(self,seed = None):

        if self.params['validation_size'] > 0 and self.training_data is not None:
            train_subset, val_subset = split_data(self.training_data,self.params['validation_size'],self.transforms_test)
            self.train_dataloader = DataLoader(train_subset, batch_size=self.params['train_batch_size'],shuffle = True,num_workers=2,pin_memory=True)
            self.validation_dataloader = DataLoader(val_subset, batch_size=self.params['train_batch_size'],shuffle = False,num_workers=2)
            self.train_len = len(train_subset)
            self.val_len = len(self.training_data)- self.train_len
        elif self.training_data is not None:
            self.train_dataloader = DataLoader(self.training_data, batch_size=self.params['train_batch_size'],shuffle = True,num_workers=2,pin_memory=True)
            self.train_len = len(self.training_data)
        if self.test_data is not None:
            self.test_dataloader = DataLoader(self.test_data, batch_size=self.params['test_batch_size'],num_workers=2,pin_memory=True)
            self.test_len = len(self.test_data)


    def get_sample(self,data = 'test',dev = None,size = None):
        if data == 'train':
            dataloader = iter(self.train_dataloader)
        elif data == 'test':
            dataloader = iter(self.test_dataloader)
        elif data == 'val':
            dataloader = iter(self.validation_dataloader)
        image,label = next(dataloader)
        if dev is not None:
            image,label = image.to(dev),label.to(dev)
        if size is not None:
            image,label = image[0:size],label[0:size]
        return image,label

    def change_transforms(self,transforms_train,transforms_test):
            self.transforms_train = transforms_train
            self.transforms_test = transforms_test

            self.training_data.transform = transforms_train
            self.test_data.transform = transforms_test

            self.generate_dataloaders()

    def get_complete_training_dataloader(self):
        if self.params['validation_size'] > 0:
            return DataLoader(self.training_data, batch_size=self.params['train_batch_size'],shuffle = True,pin_memory=True)
        else:
            return self.train_dataloader
            
    def __repr__(self):
        infos = f"{self.name} DATASET : \n Trainining data length = {self.train_len} \n"
        infos += f"Validation data length = {self.val_len} \n Test data length = {self.test_len}"
        return infos

#pass to another file
class Noisy_DataGenerator(DataGenerator):

    def __init__(self,noise_size,noisy_val = False,params = DataGenerator.params,
                 name = 'Noisy DataGenerator',n_classes = None,training_data = None, test_data = None):
        super().__init__(params,
                    name,
                    training_data,
                    test_data)
        if n_classes is not None:
            self.n_classes = n_classes
        elif not hasattr(self, 'n_classes'):
            self.n_classes = max(self.test_dataloader.dataset.targets)+1
        self.noise_size = noise_size
        self.noisy_val = noisy_val
        corrupt_label(self.train_dataloader.dataset,noise_size,self.n_classes,copy_ = False)
        if noisy_val and hasattr(self, 'validation_dataloader'):
            corrupt_label(self.validation_dataloader.dataset,noise_size,self.n_classes,copy_ = False)
    def get_clean_subset(self):
        clean_data = Subset(self.training_data, list(range(int((len(self.train_dataloader.dataset)*self.noise_size)),len(self.train_dataloader.dataset))))
        self.clean_data = DataLoader(clean_data,batch_size=self.params['train_batch_size'],shuffle = True,pin_memory=True)
        return self.clean_data

    def __repr__(self):
        infos = f"{self.name} DATASET : \n Trainining data length = {self.train_len} \n"
        infos += f"Noisy training data = {self.train_len*self.noise_size} \n"
        if self.noisy_val:
            infos += f"Noisy validation data = {self.val_len*self.noise_size} \n"
        infos += f"Validation data length = {self.val_len} \n Test data length = {self.test_len}"
        return infos    

from .cifar import Cifar10,Cifar100,Cifar100C,Cifar10C,CIFAR_C_loader