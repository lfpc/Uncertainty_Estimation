import torch
from torch_data.src import DataGenerator
import numpy as np
from os.path import exists, join
from torchvision import datasets


def get_classes(file = 'imagenet1k_classes.txt'):
    import json
    with open(file) as f:
        data = f.read()
    js = json.loads(data)
    return js
def get_transforms(model = 'resnet50'):
    from NN_models import pytorch,pretrained_models
    return pretrained_models[pytorch.__dict__[model]].transforms()


class ImageNet(DataGenerator):

    transforms_test = get_transforms()
    transforms_train = transforms_test
    n_classes = 1000
    classes = get_classes()
    def __init__(self,params=DataGenerator.params, 
                      data_dir = "data",
                      train = False,
                      val = False,
                      test = True,
                      dataloader=True,
                      transforms = None):

        if exists(join(data_dir,'ImageNet')):
            data_dir = join(data_dir,'ImageNet')
        else: raise Exception("Can't find ImageNet folder in data_dir")
        
        if transforms is not None:
            self.transforms_test = transforms

        training_data = datasets.imagenet.ImageNet(join(data_dir),split = 'train', transform = self.transforms_train) if train else None
        validation_data = datasets.imagenet.ImageNet(join(data_dir),split = 'val',transform = self.transforms_test) if val else None
        test_data = datasets.imagenet.ImageNet(join(data_dir),split = 'val',transform = self.transforms_test) if test else None
        if val and test:
            raise Warning("val and test are the same since original test has no labels")
        super().__init__(params, training_data, validation_data, test_data, dataloader)


