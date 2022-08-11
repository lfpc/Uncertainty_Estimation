
from torch import nn
from NN_models import Model_CNN



#generalizar para qualquer input
class CNN8(Model_CNN):
    input = (32,32)
    conv_layer = [
                nn.Conv2d(in_channels=3, out_channels=int(16), kernel_size=3, padding='same'),
                nn.BatchNorm2d(int(16)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(16), out_channels=int(32), kernel_size=3, padding='same'),
                nn.Dropout(p=0.2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)]

    k = int(32/4)
    fc_layer = [
                nn.Flatten(),
                nn.Linear(int(input[0]*input[1]*k), int(1024)),
                nn.Dropout(p=0.4),
                nn.ReLU(inplace=True),
                nn.Linear(int(1024), int(512)),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.4)]
    

    def __init__(self,num_classes=10, name = 'CNN8', softmax = 'log'):
        """CNN Builder."""
        main_layer = self.conv_layer+self.fc_layer
        super().__init__(num_classes,blocks = main_layer,name = name, softmax=softmax)