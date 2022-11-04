import torch
import numpy as np
import os
from .losses import HypersphericalLoss

class HypersphericalPrototypeNetwork(torch.nn.Module):
    def __init__(self,model, polars) -> None:
        super().__init__()
        self.model = model
        self.polars = polars
    @classmethod
    def from_file(cls,model,polars_file):
        classpolars = torch.from_numpy(np.load(polars_file)).float()
        return cls(model,classpolars)
    @classmethod
    def from_values(cls,model,path,classes:int,dims:int):
        polars_file = os.path.join(path,f'prototypes-{dims}d-{classes}c.npy')
        if os.path.isdir(path):
            classpolars = torch.from_numpy(np.load(polars_file)).float()
        else: 
            classpolars = HypersphericalLoss.get_prototypes(classes,dims,save_dir = path)
        return cls(model,classpolars)
    def forward(self,x):
        y = super().forward(x)
        return self.predict(y)
    def predict(self, x):
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        x = torch.mm(x, self.polars.t())
        return x