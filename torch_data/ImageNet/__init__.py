import torch
from torch_data.src import DataGenerator, split_data
import numpy as np
from os.path import exists, join
from torchvision import datasets
from copy import copy

def get_transforms(model = 'resnet50'):
    from NN_models import pytorch,pretrained_models
    return pretrained_models[pytorch.__dict__[model]].transforms()


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
                      **kwargs):

        if exists(join(data_dir,'ImageNet')):
            data_dir = join(data_dir,'ImageNet')
        else: raise Exception("Can't find ImageNet folder in data_dir")

        training_data = datasets.imagenet.ImageNet(join(data_dir),split = 'train', transform = self.transforms_train) if train else None
        test_data = datasets.imagenet.ImageNet(join(data_dir),split = 'val',transform = self.transforms_test) if test else None
        super().__init__(params, training_data, None, test_data, dataloader,**kwargs)
        #self.classes = self.get_classes(join(data_dir,'imagenet1k_classes.txt'))
    def get_classes(self,file = 'imagenet1k_classes.txt'):
        import json
        with open(file) as f:
            data = f.read()
        js = json.loads(data)
        return js
    def __split_validation(self):
        self.complete_test_data = copy(self.test_data)
        if self.validation_as_train:
            self.test_data, self.validation_data = split_data(self.test_data,self.params['validation_size'],self.transforms_train)
        else:
            self.test_data, self.validation_data = split_data(self.test_data,self.params['validation_size'],self.transforms_test)
        self.val_len = len(self.test_data)- self.test_len


import pathlib
import tarfile
import requests
import shutil

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

URLS = {"matched-frequency" : "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-matched-frequency.tar.gz",
        "threshold-0.7" : "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-threshold0.7.tar.gz",
        "top-images": "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-top-images.tar.gz",
        "val": "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenet_validation.tar.gz"}

FNAMES = {"matched-frequency" : "imagenetv2-matched-frequency-format-val",
        "threshold-0.7" : "imagenetv2-threshold0.7-format-val",
        "top-images": "imagenetv2-top-images-format-val",
        "val": "imagenet_validation"}


V2_DATASET_SIZE = 10000
VAL_DATASET_SIZE = 50000


class ImageNetV2Dataset(Dataset):
    def __init__(self, variant="matched-frequency", transform=None, location="."):
        self.dataset_root = pathlib.Path(f"{location}/ImageNetV2-{variant}/")
        self.tar_root = pathlib.Path(f"{location}/ImageNetV2-{variant}.tar.gz")
        self.fnames = list(self.dataset_root.glob("**/*.jpeg"))
        self.transform = transform
        assert variant in URLS, f"unknown V2 Variant: {variant}"
        if not self.dataset_root.exists() or len(self.fnames) != V2_DATASET_SIZE:
            if not self.tar_root.exists():
                print(f"Dataset {variant} not found on disk, downloading....")
                response = requests.get(URLS[variant], stream=True)
                total_size_in_bytes= int(response.headers.get('content-length', 0))
                block_size = 1024 #1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(self.tar_root, 'wb') as f:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        f.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    assert False, f"Downloading from {URLS[variant]} failed"
            print("Extracting....")
            tarfile.open(self.tar_root).extractall(f"{location}")
            shutil.move(f"{location}/{FNAMES[variant]}", self.dataset_root)
            self.fnames = list(self.dataset_root.glob("**/*.jpeg"))
        

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        img, label = Image.open(self.fnames[i]), int(self.fnames[i].parent.name)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


