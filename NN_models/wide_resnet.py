import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import NN_utils as utils

import sys
import numpy as np
import math



#got from  github.com/meliketoy/wide-resnet.pytorch
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out
    

class OLD_Wide_ResNet(nn.Module):
    def __init__(self,num_classes, depth = 28, widen_factor = 10, dropout_rate = 0.3, 
                    softmax = False, name = 'WideResnet'):
        super().__init__()
        self.in_planes = 16
        self.softmax = softmax
        self.name = name
        
        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.classifier_layer = nn.Sequential(
            nn.Linear(nStages[3], num_classes)) 

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.classifier_layer(out)
        if self.softmax:
            out = F.softmax(out,dim = -1)
        elif self.softmax == 'log':
            out = F.log_softmax(out,dim=-1)

        return out

    def save_state_dict(self,path, name = None):
        if name is None: name = self.name
        torch.save(self.state_dict(), path + r'/' + name + '.pt')


class BasicBlock(nn.Module):
    droprate = 0.3
    use_bn = True
    use_fixup = False
    fixup_l = 12

    def __init__(self, in_planes, out_planes, stride):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(self.droprate)

        self.equalInOut = in_planes == out_planes
        self.conv_res = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_res = not self.equalInOut and self.conv_res or None

        assert self.use_fixup or self.use_bn, "Need to use at least one thing: Fixup or BatchNorm"

        if self.use_fixup:
            self.multiplicator = nn.Parameter(torch.ones(1,1,1,1))
            self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1,1,1,1))] * 4)

            k = self.conv1.kernel_size[0] * self.conv1.kernel_size[1] * self.conv1.out_channels
            self.conv1.weight.data.normal_(0, self.fixup_l ** (-0.5) * math.sqrt(2. / k)) 
            self.conv2.weight.data.zero_()
            
            if self.conv_res is not None:
                k = self.conv_res.kernel_size[0] * self.conv_res.kernel_size[1] * self.conv_res.out_channels
                self.conv_res.weight.data.normal_(0, math.sqrt(2. / k))

    def forward(self, x):
        if self.use_bn:
            x_out = self.relu(self.bn1(x))
            out = self.relu(self.bn2(self.conv1(x_out)))
            if self.droprate > 0:
                out = self.dropout(out)
            out = self.conv2(out)
        else:
            x_out = self.relu(x + self.biases[0])
            out = self.conv1(x_out) + self.biases[1]
            out = self.relu(out) + self.biases[2]
            if self.droprate > 0:
                out = self.dropout(out)
            out = self.multiplicator * self.conv2(out) + self.biases[3]

        if self.equalInOut:
            return torch.add(x, out)

        return torch.add(self.conv_res(x_out), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride):
        layers = []

        for i in range(int(nb_layers)):
            _in_planes = i == 0 and in_planes or out_planes
            _stride = i == 0 and stride or 1
            layers.append(block(_in_planes, out_planes, _stride))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

#@registry.Model
class Wide_ResNet(nn.Module):
    def __init__(self, num_classes, depth = 28, widen_factor=10, droprate=0.3, use_bn=True, use_fixup=True,
                    softmax = False, name = 'WideResnet'):
        super().__init__()
        
        self.num_classes = num_classes
        self.softmax = softmax
        self.name = name
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        assert (depth - 4) % 6 == 0, "You need to change the number of layers"
        n = (depth - 4) / 6

        BasicBlock.droprate = droprate
        BasicBlock.use_bn = use_bn
        BasicBlock.fixup_l = n * 3
        BasicBlock.use_fixup = use_fixup
        block = BasicBlock
        
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2)
        
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.classifier_layer = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                k = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / k))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                if use_fixup:
                    m.weight.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.classifier_layer(out)
        if self.softmax == 'log':
            out = F.log_softmax(out,dim=-1)
        elif self.softmax:
            out = F.softmax(out,dim = -1)
        
        return out

if __name__ == '__main__':
    net=Wide_ResNet(10,28, 10, 0.3)

