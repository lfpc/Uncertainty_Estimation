# Define model
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

'''# Define model
class Model_CNN_with_g(Model_CNN):
    """CNN."""

    def __init__(self,num_classes=10,input = (32,32),blocks = None, g_arq = 'parallel', name = 'Model_CNN_with_g'):
        super().__init__(num_classes,input,blocks,name)
        self.g_arq = g_arq
        
        self.return_g = True
        if g_arq == 'mixed':
            self.g_layer_1 = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
            )
            self.g_layer_2 = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
            )
        
        elif g_arq == 'softmax':
            self.g_layer = nn.Sequential(
            
            nn.Linear(num_classes, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
            )
        else:
            self.g_layer = nn.Sequential(
                
                nn.Linear(512, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )


    def forward(self, x):
        """Perform forward."""
    
        x = self.main_layer(x)
        y = self.classifier_layer(x).float()

        if self.g_arq == 'mixed':
            self.g = self.g_layer_1(x)
            self.g = torch.cat((self.g,y),dim=1)
            self.g = self.g_layer_2(self.g).float()
        elif self.g_arq == 'softmax':
            self.g = self.g_layer(y).float()
        else:
            self.g = self.g_layer(x).float()

        if self.return_g:
            return y,self.g
        else:
            return y
    
    def get_g(self):
        return self.g

# Define model
class Model_CNN_with_g_and_h(Model_CNN_with_g):
    """CNN."""

    def __init__(self,num_classes=10,input = (32,32),blocks = None, g_arq = 'parallel', name = 'Model_CNN_g_and_h'):
        super().__init__(num_classes,input,blocks,g_arq,name)

        self.h_layer  = nn.Sequential(
            nn.Linear(int(512), num_classes),
            nn.LogSoftmax(dim=1)
        )


    def forward(self, x):
        """Perform forward."""
    
        x = self.main_layer(x)
        y = self.classifier_layer(x).float()

        if self.g_arq == 'mixed':
            self.g = self.g_layer_1(x)
            self.g = torch.cat((self.g,y),dim=1)
            self.g = self.g_layer_2(self.g).float()
        elif self.g_arq == 'softmax':
            self.g = self.g_layer(y).float()
        else:
            self.g = self.g_layer(x).float()

        self.h = self.h_layer(x)

        if self.return_g:
            return y,self.g
        else:
            return y

    def get_h(self):
        return self.h'''


from .wide_resnet import WideResNet
from .vgg import VGG_16
from .CNN8 import CNN8
from .pytorch_cifar import *
import torchvision.models as pytorch

pretrained_models = {
pytorch.resnet50: pytorch.ResNet50_Weights.DEFAULT,
pytorch.efficientnet_b0:pytorch.EfficientNet_B0_Weights.DEFAULT,
pytorch.vgg16_bn:pytorch.VGG16_BN_Weights.DEFAULT,
pytorch.convnext_small:pytorch.ConvNeXt_Small_Weights.DEFAULT}