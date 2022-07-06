import torch
from torch import nn
import utils as unc_utils
import uncertainty as unc
from NN_utils import indexing_3D

def MonteCarlo_meanvar(MC_array):
    '''Returns the average variance of a tensor'''
    var = torch.var(MC_array, axis=0) 
    var = torch.mean(var,axis= -1)
    return var

def MonteCarlo_maxvar(MC_array, y = None):
    '''Returns the average variance of a tensor'''
    if y is None:
        y = torch.argmax(torch.mean(MC_array,dim=0),dim = -1)
    var = torch.var(indexing_3D(MC_array,y), axis=0)
    return var

def mutual_info(pred_array):
    '''Returns de Mutual Information (Gal, 2016) of a probability tensor'''
    ent = unc.entropy(torch.mean(pred_array, axis=0))
    MI = ent - torch.mean(unc.entropy(pred_array), axis=0) 
    return MI


class Ensemble(nn.Module):
    uncs = {'var_mean': MonteCarlo_meanvar,
            'var_max': MonteCarlo_maxvar,
            'MI': mutual_info}


    def __init__(self,models_dict, return_uncs = False):
        super().__init__()
        self.models_dict = models_dict
        self.return_uncs = return_uncs

        self.p = nn.Parameter(torch.tensor(0.5,requires_grad = True)) #dummy parameter
        self.eval()
    
    def to(self,device):
        super().to(device)
        for _,model in self.models_dict.items():
            model.to(device)
    def eval(self):
        super().eval()
        for _,model in self.models_dict.items():
            model.eval()

    def apply_softmax(self):
        for _,model in self.models_dict.items():
            model.softmax = True
    def get_samples(self,x):
        ensemble = []
        for _,model in self.models_dict.items():
            model.eval()
            pred = model(x)
            ensemble.append(pred)
        self.ensemble = torch.stack(ensemble)
        return self.ensemble

    def forward(self,x):
        self.get_samples(x)
        mean = torch.mean(self.ensemble,axis = 0)
        if self.return_uncs:
            d_uncs = self.get_unc()
            return mean,d_uncs
        return mean

    def get_unc(self, x = None):
        if x is not None:
            self.get_samples(x)
        d_uncs = {}
        for name,fn in self.uncs.items():
            d_uncs[name] = fn(self.ensemble)
        return d_uncs
    