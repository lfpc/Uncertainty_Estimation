import torch
from torch_data.src import DataGenerator, split_data
import numpy as np
from os.path import exists, join
from torchvision import datasets
from copy import copy
from warnings import warn

def get_transforms(model = 'resnet50'):
    from NN_models import get_weight,torch_models
    import timm

    if model in torch_models.list_models():
        return get_weight(model).transforms()
    elif model in timm.list_models():
        return timm.data.create_transform(**timm.data.resolve_data_config(timm.models.generate_default_cfgs({model:timm.get_pretrained_cfg(model)})))

class ImageNet(DataGenerator):

    transforms_test = get_transforms()
    transforms_train = transforms_test
    n_classes = 1000
    classes = 'Apply get_classes()'#get_classes()
    def __init__(self,params=DataGenerator.params, 
                      data_dir = "data",
                      train = False,
                      test = True,
                      dataloader=True,
                      transforms_test = transforms_test,
                      **kwargs):

        if exists(join(data_dir,'ImageNet')):
            data_dir = join(data_dir,'ImageNet')
        else: raise Exception("Can't find ImageNet folder in data_dir")
        self.transforms_test = transforms_test
        training_data = datasets.imagenet.ImageNet(join(data_dir),split = 'train', transform = self.transforms_train) if train else None
        test_data = datasets.imagenet.ImageNet(join(data_dir),split = 'val',transform = transforms_test) if test else None
        super().__init__(params, training_data, None, test_data, dataloader,**kwargs)
        #self.classes = self.get_classes(join(data_dir,'imagenet1k_classes.txt'))
    def get_classes(self,file = 'imagenet1k_classes.txt'):
        import json
        with open(file) as f:
            data = f.read()
        js = json.loads(data)
        return js
    @staticmethod
    def v2(data_dir, **kwargs):
        from .imagenetv2 import ImageNetV2
        return ImageNetV2(data_dir=data_dir, **kwargs)
    def corrupted():
        warn('Corrupted ImageNet not implemented')


