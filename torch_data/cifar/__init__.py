from torchvision import datasets
import torchvision.transforms as transforms
import torch
import random
import numpy as np
from torch_data.src import DataGenerator,Noisy_DataGenerator


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Cifar10(DataGenerator):

    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2023, 0.1994, 0.2010)
    transforms_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    #transforms.RandomRotation(15),
                    transforms.Normalize(MEAN, STD)])
    transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),])

    training_data = datasets.CIFAR10

    test_data = datasets.CIFAR10

    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    n_classes = 10   


    def __init__(self,params = DataGenerator.params,
                download = True, 
                data_dir = "data",
                train = True,
                test = True,
                dataloader = True,
                **kwargs):
        training_data = self.training_data(root=data_dir,
                                                train=True,
                                                download=download,
                                                transform=self.transforms_train) if train else None
        test_data = self.test_data(root=data_dir,
                                        train=False,
                                        download=download,
                                        transform=self.transforms_test) if test else None
        super().__init__(params,training_data,None,test_data,dataloader,**kwargs)
        

class Cifar100(DataGenerator):
    MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    transforms_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    #transforms.RandomRotation(15),
                    transforms.Normalize(MEAN, STD)])
    transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),])

    training_data = datasets.CIFAR100

    test_data = datasets.CIFAR100

    classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')
    n_classes = 100


    def __init__(self,params = DataGenerator.params,
                download = True, 
                data_dir = "data",
                train = True,
                test = True,
                dataloader = True):
        training_data = self.training_data(root=data_dir,
                                                train=True,
                                                download=download,
                                                transform=self.transforms_train) if train else None

        test_data = self.test_data(root=data_dir,
                                    train=False,
                                    download=download,
                                    transform=self.transforms_test) if test else None
        super().__init__(params,training_data,None,test_data,dataloader)
        
#ARRUMAR NOISY CLASSES. PRECISAM CHAMAR CIFAR 10 ANTES  PARA DEFINIR A DATA E TAL
class Noisy_Cifar10(Noisy_DataGenerator,Cifar10):


    def __init__(self,noise_size,noisy_val = False,params = DataGenerator.params,
                download = True, name = 'Noisy CIFAR 10', data_dir = 'data'):
        self.training_data.root = data_dir
        self.test_data.root = data_dir
        super().__init__(noise_size,noisy_val,params,download, name)

class Noisy_Cifar100(Noisy_DataGenerator,Cifar100):


    def __init__(self,noise_size,noisy_val = False,params = DataGenerator.params,
                download = True, name = 'Noisy CIFAR 100', data_dir = 'data'):
        self.training_data.root = data_dir
        self.test_data.root = data_dir
        super().__init__(noise_size,noisy_val,params,
                download, name)
from .corrupted import Cifar100C,Cifar10C,CIFAR_C_loader

if __name__ == '__main__':
    data = Cifar10()
    print(list((np.asarray(data.training_data.targets)%2).astype(int)))
    
       