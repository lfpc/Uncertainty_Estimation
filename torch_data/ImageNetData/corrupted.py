from torch_data.src import DataGenerator,CorruptedDataset
import numpy as np
from os.path import exists, join
from torchvision import datasets
from PIL import Image
from torch.utils.data import Dataset

def get_transforms(model = 'resnet50'):
    from NN_models import get_weight,torch_models
    import timm

    if model in torch_models.list_models():
        return get_weight(model).transforms()
    elif model in timm.list_models():
        return timm.data.create_transform(**timm.data.resolve_data_config(timm.models.generate_default_cfgs({model:timm.get_pretrained_cfg(model)})))


class ImageNet_C_Dataset(Dataset):
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
    transforms = get_transforms()
    n_classes = 1000
    def __init__(self,  data_dir="data", corruptions:list = corruptions,levels:tuple = (0,1,2,3,4,5),
                  transforms=transforms,transforms_target = None, natural_data = None):

        self.transforms_target = transforms_target
        self.transform = transforms
        self.imgs = []
        self.targets = []
        if exists(join(data_dir,'ImageNet')):
            data_dir = join(data_dir,'ImageNet')

        for name in corruptions:
            for lvl in levels:
                if lvl == 0:
                    if natural_data is None:
                        natural_data = datasets.imagenet.ImageNet(join(data_dir),split = 'val',transform = None)
                    self.imgs.extend(natural_data.imgs)
                    self.targets.extend(natural_data.targets)
                else:
                    data = datasets.ImageFolder(root=join(data_dir,'corrupted',name,str(lvl)),transform=None)
                    self.imgs.extend(data.imgs)
                    self.targets.extend(data.targets)

    def __getitem__(self, index):
        img, targets = self.imgs[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.transforms_target is not None:
            targets = self.transforms_target(targets)
            
        return img, np.int64(targets)
    
    def __len__(self):
        return len(self.targets)

class ImageNet_C(DataGenerator):
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
    transforms_test = get_transforms()
    n_classes = 1000
    def __init__(self, data_dir="data",params=DataGenerator.params, corruptions = corruptions,levels:tuple = (0,1,2,3,4,5), 
                 seed=None, dataloader:bool = True,natural_data = None,transforms = transforms_test):
        data = ImageNet_C_Dataset(data_dir,corruptions=corruptions,levels=levels,natural_data=natural_data,transforms=transforms)
        super().__init__(params, test_data = data, seed = seed, dataloader = dataloader)
