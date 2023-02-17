from torch_data.src import DataGenerator
from torch_data import ImageNet
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset

class ImageNetV2(DataGenerator):
    types = ('imagenetv2-matched-frequency','imagenetv2-threshold07','imagenetv2-top-images')
    def __init__(self, params=DataGenerator.params, data_dir="data", types = types,
                        dataloader=True, **kwargs):
        data_dir = os.path.join(data_dir,'ImageNetV2')
        test_data = ConcatDataset([ImageFolder(root=os.path.join(data_dir,t), transform= ImageNet.transforms_test) for t in types])
        super().__init__(params, None, None, test_data, dataloader,**kwargs)
    
