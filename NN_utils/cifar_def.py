import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Lambda, Compose, Normalize
from torch.utils.data import random_split
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import copy


params = {'train_batch_size':128,
          'validation_size':0.1,
          'test_batch_size': 100}

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
 download=True,
transform=transforms_train)

test_data = datasets.CIFAR10(
root="data",
train=False,
download=True,
transform=transforms_test)

def split_data(training_data = training_data, method = 'range'):
    '''to develop'''
    assert method == 'range' or method == 'random' or method == 'idx'
    val_size = int(params['validation_size']*len(training_data))
    train_size = len(training_data) - val_size
    if method == 'random':
        training_data, validation_data = random_split(training_data, [train_size, val_size])
    elif method == 'range':
        train_idx, val_idx = train_test_split(list(range(len(training_data))), test_size=val_size, shuffle = False)
        train_subset = Subset(training_data, train_idx)
        val_subset = Subset(training_data, val_idx)


    val_subset = copy.deepcopy(val_subset)
    val_subset.dataset.transform = transforms_test

    return train_subset, val_subset

def get_dataloaders(training_data = training_data,test_data = test_data,validation_data = None):
    if validation_data is None:
        training_data,validation_data = split_data(training_data)
    train_dataloader = DataLoader(training_data, batch_size=params['train_batch_size'],shuffle = True)
    validation_dataloader = DataLoader(validation_data, batch_size=params['train_batch_size'],shuffle = False)
    test_dataloader = DataLoader(test_data, batch_size=params['test_batch_size'])
    return train_dataloader, validation_dataloader, test_dataloader