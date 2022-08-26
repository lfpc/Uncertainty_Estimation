
from torch import nn
import torchvision
from NN_models import Model_CNN

def get_vgg_layers(pretrained = False, conv_drop = True, drop_rate = 0.3):
    if not pretrained:
        pretrained = None
    #returns 
    i=0
    conv_layer = []
    for layer in torchvision.models.vgg16(weights=pretrained).features:
        conv_layer.append(layer)
        if isinstance(layer,nn.Conv2d):
            layer.padding = 'same'
            out_channels = layer.out_channels 
            i+=1
        if isinstance(layer,nn.ReLU):
            conv_layer.extend([nn.BatchNorm2d(out_channels)])
            if i%2==1 and conv_drop:
                conv_layer.append(nn.Dropout(drop_rate))
        
    
    return conv_layer


#generalizar para qualquer input (???)
class VGG_16(Model_CNN):
    def __init__(self,num_classes=10,drop_rate = 0.3,conv_drop = True,input_size = (32,32,3), pretrained = False, name = 'VGG16', softmax = 'log'):
        """CNN Builder."""
        cifar_input = (32,32,3) #CHANGE THIS EVENTUALLY
        k = 1
        for i,c in enumerate(cifar_input):
            k *= input_size[i]/c
        conv_layer = get_vgg_layers(pretrained,conv_drop,drop_rate)
        fc_layer = [nn.Flatten(),
        nn.Linear(int(k)*512, 512),
        nn.Dropout(drop_rate),
        nn.ReLU(inplace = True),
        nn.BatchNorm1d(512)]
        main_layer = conv_layer + fc_layer

        super().__init__(num_classes,main_layer,name = name, softmax=softmax)

'''class VGG_16_g(Model_CNN_with_g):
    def __init__(self,num_classes=10,input = (32,32), pretrained = False, name = 'VGG16_g'):
        """CNN Builder."""
        super().__init__(num_classes,input,get_vgg_layers(pretrained),name = name)

        self.g_layer = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Linear(512, 1),
                nn.Sigmoid())'''

