import os
import numpy as np
from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

def load_new_test_data(root, version='default'):
    data_path = root
    filename = 'cifar10.1'
    if version == 'default':
        pass
    elif version == 'v0':
        filename += '-v0'
    else:
        raise ValueError('Unknown dataset version "{}".'.format(version))
    label_filename = filename + '-labels.npy'
    imagedata_filename = filename + '-data.npy'
    label_filepath = os.path.join(data_path, label_filename)
    imagedata_filepath = os.path.join(data_path, imagedata_filename)
    labels = np.load(label_filepath).astype(np.int64)
    imagedata = np.load(imagedata_filepath)
    assert len(labels.shape) == 1
    assert len(imagedata.shape) == 4
    assert labels.shape[0] == imagedata.shape[0]
    assert imagedata.shape[1] == 32
    assert imagedata.shape[2] == 32
    assert imagedata.shape[3] == 3
    if version == 'default':
        assert labels.shape[0] == 2000
    elif version == 'v0':
        assert labels.shape[0] == 2021
    return imagedata, labels


class Cifar10_1_dataset(Dataset):
    images_url = 'https://github.com/modestyachts/CIFAR-10.1/blob/master/datasets/cifar10.1-data.npy?raw=true'
    images_md5 = '29615bb88ff99bca6b147cee2520f010'
    images_filename = 'cifar10.1-data.npy'

    labels_url = 'https://github.com/modestyachts/CIFAR-10.1/blob/master/datasets/cifar10.1-labels.npy?raw=true'
    labels_md5 = 'a27460fa134ae91e4a5cb7e6be8d269e'
    labels_filename = 'cifar10.1-labels.npy'
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2023, 0.1994, 0.2010)

    transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),])

    classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
        'ship', 'truck'
    ]

    @property
    def targets(self):
        return self.labels

    def __init__(self,
                 root,
                 transform= transforms_test,
                 target_transform=None,
                 download=False):
        
        root = os.path.join(root,'cifar10_1')
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        images, labels = load_new_test_data(root)

        self.data = images
        self.labels = labels

        self.class_to_idx = {
            _class: i
            for i, _class in enumerate(self.classes)
        }

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        data_path = os.path.join(self.root, self.images_filename)
        labels_path = os.path.join(self.root, self.labels_filename)
        return (check_integrity(data_path) and check_integrity(labels_path))

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.images_url, root, self.images_filename, self.images_md5)
        download_url(self.labels_url, root, self.labels_filename, self.labels_md5)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp,
            self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp,
            self.target_transform.__repr__().replace('\n',
                                                     '\n' + ' ' * len(tmp)))
        return 

def Cifar10_1(PATH,batch_size = 100,shuffle=False,num_workers=2,pin_memory=True, **kwargs):
    return DataLoader(Cifar10_1_dataset(PATH), batch_size=batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory = pin_memory, **kwargs)