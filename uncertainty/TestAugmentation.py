
import torch
#import albumentation
from uncertainty import ensemble
from copy import copy
import uncertainty as unc
from torchvision.transforms import functional as F
import torchvision


def TestTimeAugmentation(model,X, transforms,use_main = True):
    with torch.no_grad(): 
        samples = []
        if use_main:
            pred = model(X)
            samples.append(pred)
        for t in transforms:
            x = t(X)
            if isinstance(x,tuple):
                for x_ in x:
                    pred = model(x_)
                    samples.append(pred)
            else:
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
        #torch.clamp(x*self.a,min = 0.0,max = 1.0)
        return x*self.a
class Add:       
    def __init__(self, a:float):
        self.a = a

    def __call__(self, x):
        #torch.clamp(x+self.a,min = 0.0,max = 1.0)
        return x+self.a
class FiveCrop():
    def __init__(self,size,pad = 0) -> None:
        self.pad = torchvision.transforms.Pad(pad)
        self.fivecrop = torchvision.transforms.FiveCrop(size)
    def __call__(self, x):
        x = self.pad(x)
        return self.fivecrop(x)

class TTA(ensemble.Ensemble):
    # ver se essas coisas funcionam com batches
    transforms = [F.hflip,
                  Scale(1.04),
                  Scale(1.1),
                  Rotate(15),
                  Rotate(-15),
                  Multiply(0.8),
                  Multiply(1.2),
                  Add(0.1),
                  Add(-0.1),
                  FiveCrop(32,4)]

    def __init__(self, model, transforms = transforms,
                  use_main:bool = True, inference:str = 'mean', apply_softmax:bool = True):
        super().__init__(model, inference=inference, apply_softmax= apply_softmax)
        self.use_main = use_main
        self.transforms = transforms

    def get_samples(self,x):
        ensemble = TestTimeAugmentation(self.model,x,self.transforms,use_main = self.use_main)
        return ensemble

