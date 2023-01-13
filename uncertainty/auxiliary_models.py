import torch
import numpy as np
import os
from .losses import HypersphericalLoss

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