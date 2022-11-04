from logging import raiseExceptions
from torch_data.src import DataGenerator
from torchvision import datasets
import torchvision.transforms as transforms
#from .tin import TinyImageNetDataset
from os.path import join,exists
from operator import xor

class TinyImageNet(DataGenerator):

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    transforms_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    transforms_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    classes = "Apply get_classes(data_dir)"
    n_classes = 200


    def __init__(self,params = DataGenerator.params, 
                data_dir = "data",
                train = True,
                val = False,
                test = True,
                dataloader = True):
        if exists(join(data_dir,'tiny-imagenet-200')):
            data_dir = join(data_dir,'tiny-imagenet-200')
        elif not (exists(join(data_dir,'train')) and exists(join(data_dir,'val')) and exists(join(data_dir,'test'))):
            raise Exception("Wrong Data Directory")
        training_data = datasets.ImageFolder(join(data_dir,'train'),self.transforms_train) if train else None
        validation_data = datasets.ImageFolder(join(data_dir,'val','images'),self.transforms_test) if val else None
        test_data = datasets.ImageFolder(join(data_dir,'val','images'),self.transforms_test) if test else None
        if val and test:
            raise Warning("val and test are the same since original test has no labels")
        #### TEST_DATA is actually VAL_DATA
        ####ORIGINAL TEST DATA HAVE NO LABELS


        super().__init__(params, training_data, validation_data, test_data, dataloader)
    def get_classes(self,data_dir):
        with open(join(data_dir,'words.txt')) as f:
            lines = f.readlines()
        self.classes = []
        for line in lines:
            words = line.split('\t')
            self.classes.append(words[1].split('\n')[0])
        self.classes = tuple(self.classes)
        return self.classes

class TinyImageNet_224(TinyImageNet):
    transforms_train = transforms.Compose([
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    #transforms.RandomRotation(15),
                    transforms.Normalize(TinyImageNet.MEAN, TinyImageNet.STD)])
    transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.Normalize(TinyImageNet.MEAN, TinyImageNet.STD),])
    def __init__(self, params=DataGenerator.params, data_dir="data", train=True, val=False, test=True, dataloader=True):
        super().__init__(params, data_dir, train, val, test, dataloader)




if __name__ == '__main__':
    print(transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC))