
import torch
#import albumentation
from ensemble import Ensemble
from copy import copy
import uncertainty as unc
from torchvision.transforms import functional as F
import torchvision


def TestTimeAugmentation(model,X, transforms):
    with torch.no_grad(): 
        samples = []
        for t in transforms:
            x = t(X)
            pred = model(x)
            samples.append(pred)
        samples = torch.stack(samples)
    return samples

class Rotate:
    """Rotate by one of the given angles."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        rot = F.rotate(x, self.angle)
        return rot
class Affine:
    def __init__(self, angle,translate,scale,shear):
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, x):
        new = F.affine(x,self.angle,self.translate,self.scale,self.shear)
        return new

class Scale(Affine):
    """Rotate by one of the given angles."""

    def __init__(self, scale):
        super().__init__(0,[0,0],scale,0)
class Multiply:       
    def __init__(self, a:float):
        self.a = a

    def __call__(self, x):
        return torch.clamp(x*self.a,min = 0.0,max = 1.0)
class Add:       
    def __init__(self, a:float):
        self.a = a

    def __call__(self, x):
        return torch.clamp(x+self.a,min = 0.0,max = 1.0)


class TTA(Ensemble):
    transforms = [F.hflip,
                  Scale(1.1),
                  Scale(1.2),
                  Rotate(15),
                  Rotate(-15),
                  Multiply(0.8),
                  Multiply(1.2),
                  Add(0.1),
                  Add(-0.1),
                  torchvision.transforms.FiveCrop(32)]

    def __init__(self, model,n_samples,as_ensemble = True, transforms = transforms,
                 return_uncs=False, softmax=False):
        models_dict = {'model':model}
        super().__init__(models_dict, return_uncs, softmax)
        self.model = model
        self.n_samples = n_samples
        self.as_ensemble = as_ensemble
        self.transforms = transforms
        if not self.as_ensemble:
            self.uncs = copy(self.uncs)
            self.uncs['MCP (Ensemble)'] = lambda x: unc.MCP_unc(torch.mean(x,axis = 0))
            self.uncs['Entropy (Ensemble)'] = lambda x: unc.entropy(torch.mean(x,axis = 0))
    def get_samples(self,x):
        self.ensemble = TestTimeAugmentation(self.model,x,self.n_samples, self.transforms)
        return self.ensemble

    def deterministic(self,x):
        y = self.model(x)
        if self.softmax and not self.model.softmax:
            y = torch.nn.functional.softmax(y,dim=-1)
        if self.return_uncs:
            d_uncs = self.get_unc(x)
            return y,d_uncs
        else:
            self.get_samples(x) #needed if get_unc get called, as usually does
            return y
    
    def forward(self,x):
        if self.as_ensemble:
            y = super().forward(x)
        else:
            y = self.deterministic(x)
        return y

