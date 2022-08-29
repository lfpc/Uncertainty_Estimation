"""
Basen on https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch
"""

import numpy as np
from .__init__ import Cifar10,Cifar100
from torch_data.imbalance_data import ImbalanceDataSet

class ImbalanceCifar10(Cifar10,ImbalanceDataSet):
    def __init__(self, imbalance_ratio = 0.01,imb_type = 'exp', train = True, test = True,
                        params = Cifar10.params,
                        download=True, 
                        data_dir="data"):
        Cifar10.__init__(self,params, download, data_dir, train = train, test = test, dataloader = False)
        ImbalanceDataSet.__init__(self,imbalance_ratio,
                    self.training_data,
                    self.validation_data,
                    self.test_data,
                    imb_type,
                    train,
                    test,
                    params)

class ImbalanceCifar100(Cifar100,ImbalanceDataSet):
    def __init__(self, imbalance_ratio = 0.01,imb_type = 'exp', train = True, test = True,
                        params = Cifar10.params,
                        download=True, 
                        data_dir="data"):
        Cifar100.__init__(params, download, data_dir, train = train, test = test, dataloader = False)
        ImbalanceDataSet.__init__(imbalance_ratio,
                    self.training_data,
                    self.validation_data,
                    self.test_data,
                    imb_type,
                    train,
                    test,
                    params)

from torch_data.src import Binary_DataGenerators

def ImbalanceBinaryCifar10(**kwargs):
    data = ImbalanceCifar10(**kwargs)
    data = Binary_DataGenerators(data)

    for i,_class in enumerate(data.classes):
        classes_0 = []
        classes_1 = []
        if i%2:
            classes_1.append(_class)
        else:
            classes_0.append(_class)
    classes_0 = tuple(classes_0)
    classes_1 = tuple(classes_1)
    data.classes = (classes_0,classes_1)

    return data
    
def ImbalanceBinaryCifar100(**kwargs):
    data = ImbalanceCifar100(**kwargs)
    data = Binary_DataGenerators(data)

    for i,_class in enumerate(data.classes):
        classes_0 = []
        classes_1 = []
        if i%2:
            classes_1.append(_class)
        else:
            classes_0.append(_class)
    classes_0 = tuple(classes_0)
    classes_1 = tuple(classes_1)
    data.classes = (classes_0,classes_1)

    return data