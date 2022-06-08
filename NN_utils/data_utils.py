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
    print(val_subset.__dict__)
    print(val_subset.dataset)
    val_subset.dataset.transform = val_transforms

    return train_subset, val_subset

def get_dataloaders(training_data,test_data,params,validation_data = None):
    if validation_data is None:
        training_data,validation_data = split_data(training_data)
    train_dataloader = DataLoader(training_data, batch_size=params['train_batch_size'],shuffle = True)
    validation_dataloader = DataLoader(validation_data, batch_size=params['train_batch_size'],shuffle = False)
    test_dataloader = DataLoader(test_data, batch_size=params['test_batch_size'])
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

