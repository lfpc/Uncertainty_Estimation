# Define model
import torch
from torch import nn

class Model_CNN_10(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super().__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=int(32), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(32)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=int(32), out_channels=int(64), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=int(64), out_channels=int(128), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(128)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=int(128), out_channels=int(128), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),

            # Conv Layer block 3
            nn.Conv2d(in_channels=int(128), out_channels=int(256), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(256)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=int(256), out_channels=int(256), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_x_layer = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(4096, int(1024)),
            nn.ReLU(inplace=True),
            nn.Linear(int(1024), int(512)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(int(512), 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        #x = self.fc_layer(x)


        y = self.fc_x_layer(x)

        y = y.float()

        

        return y


# Define model
class Model_CNN_10_with_g(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super().__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=int(32), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(32)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=int(32), out_channels=int(64), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=int(64), out_channels=int(128), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(128)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=int(128), out_channels=int(128), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),

            # Conv Layer block 3
            nn.Conv2d(in_channels=int(128), out_channels=int(256), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(256)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=int(256), out_channels=int(256), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_x_layer = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(4096, int(1024)),
            nn.ReLU(inplace=True),
            nn.Linear(int(1024), int(512)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(int(512), 10),
            nn.LogSoftmax(dim=1)
        )
        
        self.fc_g_layer = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(4096, int(1024)),
            nn.ReLU(inplace=True),
            nn.Linear(int(1024), int(512)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(int(512), 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        #x = self.fc_layer(x)


        y = self.fc_x_layer(x)

        self.g = self.fc_g_layer(x)

        #x = torch.stack((self.x1,self.g))
        self.g = (self.g).float()
        y = y.float()

        

        return y
    def get_g(self):
        return self.g