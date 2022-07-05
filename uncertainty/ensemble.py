import torch
from torch import nn
import utils as unc_utils
import uncertainty as unc

class Ensemble(nn.Module):
    uncs = ['var_mean',
               'var_max',
               'MI']
    def __init__(self,models_dict):
        super().__init__()
        self.models_dict = models_dict

        self.p = nn.Parameter(torch.tensor(0.5,requires_grad = True)) #dummy parameter
    
    def to(self,device):
        super().to(device)
        for _,model in self.models_dict.items():
            model.to(device)

    def forward(self,x):
        
        ensemble = []
        for _,model in self.models_dict.items():
            model.eval()
            pred = model(x)
            ensemble.append(pred)
        self.ensemble = torch.stack(ensemble)
        mean = torch.mean(self.ensemble,axis = 0)
        return mean
    def get_unc(self):
        var_mean = unc_utils.MonteCarlo_meanvar(torch.exp(self.ensemble))
        var_max = unc_utils.MonteCarlo_maxvar(torch.exp(self.ensemble))
        MI = unc.mutual_info(torch.exp(self.ensemble))
        return {'var_mean':var_mean,
               'var_max':var_max,
               'MI': MI}
    