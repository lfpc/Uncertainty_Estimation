import torch
import numpy as np
import os
from .losses import HypersphericalLoss
from copy import copy,deepcopy
from NN_utils import freeze_params, unfreeze_params

class Platt_Model(torch.nn.Module):
    def __init__(self,model,A = 1.0,B = 0.0):
        '''Model with outputs z' = Az+ B, where z is the logits vector output of the main model'''
        super().__init__()
        self.model = model
        self.A = torch.nn.Parameter(torch.tensor(A,requires_grad = True))
        self.B = torch.nn.Parameter(torch.tensor(B,requires_grad = True))
        self.to(next(model.parameters()).device)
    def forward(self,x):
        logits = self.model(x)
        return logits*self.A + self.B


class HypersphericalPrototypeNetwork(torch.nn.Module):
    def __init__(self,model, polars) -> None:
        super().__init__()
        self.model = model
        self.polars = torch.nn.parameter.Parameter(polars)
    @classmethod
    def from_file(cls,model,polars_file):
        classpolars = torch.from_numpy(np.load(polars_file)).float()
        return cls(model,classpolars)
    @classmethod
    def from_values(cls,model,path,num_classes:int,dims:int):
        if os.path.isdir(path):
            polars_file = os.path.join(path,f'prototypes-{dims}d-{num_classes}c.npy')
            classpolars = torch.from_numpy(np.load(polars_file)).float()
        else: 
            classpolars = HypersphericalLoss.get_prototypes(num_classes,dims,save_dir = path)
        return cls(model,classpolars)
    def forward(self,x):
        y = self.model(x)
        y = self.predict(y)
        return y
    def predict(self, x):
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        x = torch.mm(x, self.polars.t())
        return x

class GNet(torch.nn.Module):
    def __init__(self, features,classifier, g_layer) -> None:
        super().__init__()
        self.features= features
        self.classifier = classifier
        self.g_layer = g_layer
        self.freeze = False

    def forward(self,x):
        z = self.features(x)
        y = self.classifier(z)
        g = self.g_layer(z)
        return y,g
    def freeze(self, layers = None):
        freeze_params(self.features, layers= layers)
        freeze_params(self.classifier, layers= layers)
        self.freeze = True
    def unfreeze(self, layers = None):
        unfreeze_params(self.features, layers= layers)
        unfreeze_params(self.classifier, layers= layers)
        self.freeze = False
    def train(self, mode):
        super().train(mode)
        if self.freeze:
            self.features.eval()
            self.classifier.eval()
        return self
    @classmethod
    def from_model(cls, model, g_layer = None,**kwargs):
        model = copy(model)
        if any('classifier' in n for n,_ in model.named_children()):
            classifier = copy(model.classifier)
            model.classifier = torch.nn.Identity()
        elif any('fc' in n for n,_ in model.named_children()):
            classifier = copy(model.fc)
            model.fc = torch.nn.Identity()
        elif any('linear' in n for n,_ in model.named_children()):
            classifier = copy(model.linear)
            model.linear = torch.nn.Identity()
        if g_layer is None:
            in_features = classifier.in_features
            g_layer = torch.nn.Sequential(
                        torch.nn.Linear(in_features, 512),
                        torch.nn.ReLU(),
                        torch.nn.BatchNorm1d(512),
                        torch.nn.Linear(512,1),
                        torch.nn.Sigmoid())

        return cls(model, classifier,g_layer, **kwargs)


class SelectiveNet(GNet):
    def __init__(self, features,classifier, g_layer, aux_head:str = 'aux') -> None:
        super().__init__(features, classifier, g_layer)
        if aux_head == 'aux':
            self.aux_head = deepcopy(classifier)
        elif aux_head == 'self':
            self.aux_head = self.classifier
    def forward(self,x):
        z = self.features(x)
        y = self.classifier(z)
        g = self.g_layer(z)
        h = self.aux_head(z)
        return y,g,h

