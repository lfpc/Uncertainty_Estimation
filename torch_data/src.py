import copy
from torch.utils.data import Subset, Dataset,random_split, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from random import randrange
from torchvision import datasets
import torch
from PIL import Image
import os

def split_data(training_data,val_size,val_transforms = None, method = 'pre_defined', seed = 0):
    '''to develop'''
    assert method == 'pre_defined' or method == 'random' or method == 'idx'
    
    if method == 'random':
        val_size = int(val_size*len(training_data))
        train_size = len(training_data) - val_size
        if seed is not None:
            train_subset, val_subset = random_split(training_data, [train_size, val_size],generator=torch.Generator().manual_seed(seed))
        else:
            train_subset, val_subset = random_split(training_data, [train_size, val_size])
    elif method == 'pre_defined':
        train_idx, val_idx = train_test_split(list(range(len(training_data))), random_state = seed,test_size=val_size,stratify = training_data.targets)
        train_subset = Subset(training_data, train_idx)
        val_subset = Subset(training_data, val_idx)
        
    if val_transforms is not None:
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
                    training_data = None,
                    validation_data = None,
                    test_data = None,
                    seed = None,
                    dataloader:bool = True,
                    validation_as_train:bool = False):


        self.params = params

        self.training_data = training_data
        self.train_len = len(self.training_data) if training_data is not None else 0
        self.test_data = test_data
        self.test_len = len(self.test_data) if test_data is not None else 0
        self.validation_data = validation_data
        self.validation_as_train = validation_as_train
        if validation_data is not None:
            self.params['validation_size'] = len(validation_data)/(self.train_len+len(validation_data))
        elif self.params['validation_size'] > 0:
            if self.training_data is not None:
                self.__split_validation()
            elif self.test_data is not None:
                self.__split_validation(origin = 'test')

        
        
        
        self.val_len = len(self.validation_data) if self.validation_data is not None else 0
        if dataloader:
            self.generate_dataloaders(seed)

    def __split_validation(self, origin = 'train'):
        if origin == 'train':
            data = self.training_data
            self.complete_train_data = copy.copy(data)
        elif origin == 'test':
            data = self.test_data
            self.complete_test_data = copy.copy(data)
        if self.validation_as_train:
            data, self.validation_data = split_data(data,self.params['validation_size'],self.transforms_train)
        else:
            data, self.validation_data = split_data(data,self.params['validation_size'])
        self.val_len = len(self.validation_data)
        if origin == 'train':
            self.training_data = data
        elif origin == 'test':
            self.test_data = data

    def generate_dataloaders(self,seed = None):
        if self.validation_data is not None:
            batch_size = self.params['train_batch_size'] if self.validation_as_train else self.params['test_batch_size'] 
            self.validation_dataloader = DataLoader(self.validation_data, batch_size=batch_size,shuffle = self.validation_as_train,num_workers=2,pin_memory=True)
        if self.training_data is not None:
            self.train_dataloader = DataLoader(self.training_data, batch_size=self.params['train_batch_size'],shuffle = True,num_workers=2,pin_memory=True)
        if self.test_data is not None:
            self.test_dataloader = DataLoader(self.test_data, batch_size=self.params['test_batch_size'],shuffle=False,num_workers=2,pin_memory=True)
            


    def get_sample(self,data = 'test',device = None,size = None):
        if data == 'train':
            dataloader = iter(self.train_dataloader)
        elif data == 'test':
            dataloader = iter(self.test_dataloader)
        elif data == 'val':
            dataloader = iter(self.validation_dataloader)
        image,label = next(dataloader)
        if device is not None:
            image,label = image.to(device),label.to(device)
        if size is not None:
            image,label = image[0:size],label[0:size]
        return image,label

    def __iter__(self):
        return iter(self.test_dataloader)


    def change_transforms(self,transforms_train = None,transforms_test = None):
            if transforms_train is not None:
                self.transforms_train = transforms_train
                if hasattr(self.training_data,'transform'):
                    self.training_data.transform = transforms_train
                else: self.training_data.dataset.transform = transforms_train
            if transforms_test is not None:
                self.transforms_test = transforms_test
                if hasattr(self.test_data,'transform'):
                    self.test_data.transform = transforms_test
                else: self.test_data.dataset.transform = transforms_test

            self.generate_dataloaders()

    def get_complete_training_dataloader(self):
        if self.params['validation_size'] > 0:
            return DataLoader(self.training_data, batch_size=self.params['train_batch_size'],shuffle = True,pin_memory=True)
        else:
            return self.train_dataloader
            
    def __repr__(self):
        infos = f"Trainining data length = {self.train_len} \n"
        infos += f"Validation data length = {self.val_len} \n Test data length = {self.test_len}"
        return infos

        
def Binary_DataGenerators(data,type = 'parity'):
    if hasattr(data,'n_classes'):
        data.n_classes = 2
    if  data.training_data is not None:
        Binary_Dataset(data.training_data)
    if  data.test_data is not None:
        Binary_Dataset(data.test_data)
    if  data.validation_data is not None:
        Binary_Dataset(data.validation_data)
    return data

def Binary_Dataset(data,type = 'parity'):
    data.targets = list((np.asarray(data.targets)%2).astype(int))
    return data




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


class custom_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.targets = labels
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)
    
class CorruptedDataset(datasets.VisionDataset):
    corruptions = ['gaussian_noise',
                    'shot_noise',
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
                    'frost']
    extra = ['glass_bur', 'saturate','spatter','speckle_noise']

    def __init__(self, data_dir :str,im_size: tuple, corruptions:list = corruptions,levels:tuple = (0,1,2,3,4,5),
                 transform= None, target_transform = None, natural_data = None):
        super().__init__(
                   data_dir, transform=transform,
                   target_transform=target_transform)
        im_size = (0,) + im_size
        self.data = np.empty(im_size).astype(np.uint8)
        self.targets = np.array([])
        
        if 0 in levels:
            data = natural_data.data
            targets = natural_data.targets
            self.data = np.concatenate((self.data,data))
            self.targets = np.concatenate((self.targets,targets))


        for name in corruptions:
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