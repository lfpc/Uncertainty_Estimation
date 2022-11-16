import torch
import numpy as np
import os
from .losses import HypersphericalLoss

class HypersphericalPrototypeNetwork(torch.nn.Module):
    def __init__(self,model, polars, eval_predict = True) -> None:
        super().__init__()
        self.model = model
        self.polars = polars
        self.eval_predict = eval_predict
    @classmethod
    def from_file(cls,model,polars_file):
        classpolars = torch.from_numpy(np.load(polars_file)).float()
        return cls(model,classpolars)
    @classmethod
    def from_values(cls,model,path,classes:int,dims:int):
        if os.path.isdir(path):
            polars_file = os.path.join(path,f'prototypes-{dims}d-{classes}c.npy')
            classpolars = torch.from_numpy(np.load(polars_file)).float()
        else: 
            classpolars = HypersphericalLoss.get_prototypes(classes,dims,save_dir = path)
        return cls(model,classpolars)
    def forward(self,x):
        y = self.model(x)
        if self.eval_predict and not self.training:
            y = self.predict(y)
        return y
    def predict(self, x):
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        x = torch.mm(x, self.polars.t())
        return x