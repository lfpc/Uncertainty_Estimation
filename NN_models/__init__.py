# Define model
import torch
from torch import nn

 
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

    def __init__(self,n_classes=10,input = (32,32),blocks = None):
        """CNN Builder."""
        super().__init__()
        
        if blocks is None:
            conv_layer = [
                nn.Conv2d(in_channels=3, out_channels=int(16), kernel_size=3, padding='same'),
                nn.BatchNorm2d(int(16)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(16), out_channels=int(32), kernel_size=3, padding='same'),
                nn.Dropout(p=0.2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

            ]
            k = int(32/4)
        else:
            conv_layer,k = construct_conv_layer(blocks)

        
        fc_layer = [
            nn.Flatten(),
            nn.Linear(int(input[0]*input[1]*k), int(1024)),
            nn.Dropout(p=0.4),
            nn.ReLU(inplace=True),
            nn.Linear(int(1024), int(512)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4)]
        
        main_layer = conv_layer+fc_layer
        
        self.main_layer = nn.Sequential(*main_layer)
        
        self.classifier_layer = nn.Sequential(
            nn.Linear(int(512), n_classes),
            nn.LogSoftmax(dim=1)
        )


    def forward(self, x):
        """Perform forward."""
        x = self.main_layer(x)
        y = self.classifier_layer(x).float()

        return y


# Define model
class Model_CNN_with_g(Model_CNN):
    """CNN."""

    def __init__(self,n_classes=10, blocks = None, g_arq = 'parallel'):
        """CNN Builder."""
        '''g_arq:
        parallel: head that gets as input x, the output of the main layer.
        Gets no information of the classificer layer
        softmax: input is y, the classifier softmax. 
        mixed: input is a concatenation between x and y.'''
        super().__init__(n_classes,blocks)
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
            
            nn.Linear(n_classes, 64),
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

    def __init__(self,n_classes=10, blocks = None, g_arq = 'parallel'):
        """CNN Builder."""
        '''g_arq:
        parallel: head that gets as input x, the output of the main layer.
        Gets no information of the classificer layer
        softmax: input is y, the classifier softmax. 
        mixed: input is a concatenation between x and y.'''
        super().__init__(n_classes,blocks,g_arq)

        self.h_layer  = nn.Sequential(
            nn.Linear(int(512), n_classes),
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
        return self.h
