# Define model
from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F
from os.path import join
from os import getcwd
import sys

def construct_conv_layer(blocks):
    #to elaborate
    #iterate over blocks and append to conv_layer conv2d with blocks_i size
    k = 0
    conv_layer = []
    for i,b in enumerate(blocks):
        conv_layer.extend(
            [nn.Conv2d(in_channels=b, out_channels=blocks(i+1), kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)])
        if (i+1)%2 == 0:
            conv_layer.extend(
            [nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)])
            k+=1
        else:
            nn.BatchNorm2d(blocks(i+1))
        if i+1 == len(blocks):
            break
    k = ((1/2)**(2*k))*blocks[-1]
    return conv_layer,k


# Define model
class Model_CNN(nn.Module):
    """CNN."""

    def __init__(self,num_classes,blocks, name = 'Model_CNN', softmax = 'log'):
        """CNN Builder."""
        super().__init__()
        self.name = name
        self.softmax = softmax
        if isinstance(blocks,list):
            self.main_layer = nn.Sequential(*blocks)
        else:
            self.main_layer = blocks
        
        self.classifier_layer = nn.Linear(int(512), num_classes)


    def forward(self, x):
        """Perform forward."""
        x = self.main_layer(x)
        y = self.classifier_layer(x).float()

        if self.softmax == 'log':
            y = F.log_softmax(y,dim=-1)
        elif self.softmax:
            y = F.softmax(y,dim = -1)
        return y

    def save_state_dict(self,path, name = None):
        if name is None: name = self.name
        name = name + '.pt'
        torch.save(self.state_dict(), join(path,name))#path + r'/' + name + '.pt')

class Softmax_Model(nn.Module):
    def __init__(self,model) -> None:
        self.model = model
    def forward(self,x):
        return F.softmax(self.model(x),dim=-1)

from .wide_resnet import WideResNet28_10, WideResNet
from .CNN8 import CNN8
from .pytorch_cifar import *
from .resnet_dropout import ResNet18_Dropout, ResNet50_Dropout
import torchvision.models as pytorch
from torchvision.models import vgg16_bn, convnext_small, efficientnet_b0


pretrained_models = {
pytorch.resnet50: pytorch.ResNet50_Weights.DEFAULT,
pytorch.efficientnet_b0:pytorch.EfficientNet_B0_Weights.DEFAULT,
pytorch.vgg16_bn:pytorch.VGG16_BN_Weights.DEFAULT,
pytorch.convnext_small:pytorch.ConvNeXt_Small_Weights.DEFAULT}