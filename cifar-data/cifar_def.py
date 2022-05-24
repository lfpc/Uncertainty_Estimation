import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import copy
import NN_utils.data_utils as data_utils
import torch
from random import randrange

class Cifar_10_data():

    params = {'train_batch_size':128,'validation_size':0.1,'test_batch_size': 100}
    transforms_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=False,
    transform=transforms_train)

    test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=False,
    transform=transforms_test)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    def __init__(self,params = params, download = True):
        if download:
            self.training_data.download()
            self.test_data.download()
        self.params = params
        self.generate_dataloaders()

    def generate_dataloaders(self):
        if self.params['validation_size'] > 0:
            train_subset, val_subset = data_utils.split_data(self.training_data,self.params['validation_size'],self.transforms_test)
            self.validation_dataloader = DataLoader(val_subset, batch_size=self.params['train_batch_size'],shuffle = False)
    
        self.train_dataloader = DataLoader(train_subset, batch_size=self.params['train_batch_size'],shuffle = True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.params['test_batch_size'])

        self.train_len = len(train_subset)
        self.val_len = len(val_subset)
        self.test_len = len(self.test_data)


    def get_sample(self,data = 'train',dev = torch.device('cpu'),size = None):
        if data == 'train':
            dataloader = iter(self.train_dataloader)
        elif data == 'test':
            dataloader = iter(self.test_dataloader)
        elif data == 'val':
            dataloader = iter(self.validation_dataloader)
        image,label = next(dataloader)
        image,label = image.to(dev),label.to(dev)
        if size is not None:
            image,label = image[0:size],label[0:size]
        return image,label

    def change_transforms(self,transforms_train,transforms_test):
            self.transforms_train = transforms_train
            self.transforms_test = transforms_test
            self.training_data = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transforms_train)

            self.test_data = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=transforms_test)

            self.generate_dataloaders()

    def get_complete_training_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.params['train_batch_size'],shuffle = True)
    def __repr__(self):
        infos = f"CIFAR 10 DATASET : \n Trainining data length = {self.train_len} \n"
        infos += f"Validation data length = {self.val_len} \n Test data length = {self.test_len}"
        return infos
        

class Noisy_CIFAR_10(Cifar_10_data):

    def __init__(self,noise_size,noisy_val = False,params = Cifar_10_data.params, download = True):
        super().__init__(params,download)

        self.noise_size = noise_size
        data_utils.corrupt_label(self.train_dataloader.dataset,noise_size,copy_ = False)
        if noisy_val:
            data_utils.corrupt_label(self.validation_dataloader.dataset,noise_size,copy_ = False)


    def get_clean_subset(self):
        clean_data = Subset(self.training_data, list(range(int((len(self.train_dataloader.dataset)*self.noisy_size)),len(self.train_dataloader.dataset))))
        self.clean_data = DataLoader(clean_data,batch_size=self.params['train_batch_size'],shuffle = True)
        return self.clean_data


       