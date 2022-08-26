import torch
from torch import nn
#import utils as unc_utils
import uncertainty as unc
from NN_utils import indexing_3D
from copy import copy


def MonteCarlo_meanvar(MC_array):
    '''Returns the average variance of a tensor'''
    var = torch.var(MC_array, axis=0, unbiased=False) 
    var = torch.mean(var,axis= -1)
    return var

def MonteCarlo_maxvar(MC_array, y = None):
    '''Returns the average variance of a tensor'''
    if y is None:
        y = torch.argmax(torch.mean(MC_array,dim=0),dim = -1)
    var = torch.var(indexing_3D(MC_array,y), axis=0,unbiased=False)
    return var

def mutual_info(pred_array):
    '''Returns de Mutual Information (Gal, 2016) of a probability tensor'''
    ent = unc.entropy(torch.mean(pred_array, axis=0))
    MI = ent - torch.mean(unc.entropy(pred_array), axis=0) 
    return MI


class Ensemble(nn.Module):
    uncs = {'Var(Mean)': MonteCarlo_meanvar,
            'Var(Max)': MonteCarlo_maxvar,
            'MI': mutual_info}

    def __init__(self,model, #model
                 return_uncs:bool = False, #return tuple (output,uncs_dict)
                 as_ensemble:bool = True, #output as average of models
                 softmax = False, #apply SM to ensemble output
                 name = 'Ensemble'
                 ):
        super().__init__()

        self.return_uncs = return_uncs
        self.as_ensemble = as_ensemble
        self.softmax = softmax
        self.model = model
        self.name = name
        try:
            self.device = next(self.model.parameters()).device
        except:
            self.device = torch.device('cpu')
            
        self.eval()
        if not self.as_ensemble:
            self.uncs = copy(self.uncs)
            self.uncs['MCP (Ensemble)'] = lambda x: unc.MCP_unc(torch.mean(x,axis = 0))
            self.uncs['Entropy (Ensemble)'] = lambda x: unc.entropy(torch.mean(x,axis = 0))
    
    def to(self,device):
        super().to(device)
        self.device = device
        self.model.to(device)
        return self

    def eval(self):
        super().eval()
        self.model.eval()
        return self
        
    def get_samples(self,x):
        '''Default ensemble model is to assume that self.model returns samples'''
        self.ensemble = self.model(x)
        return self.ensemble

    def deterministic(self,x):
        self.eval()
        self.y = self.model(x)
        if self.softmax and not self.model.softmax:
            self.y = torch.nn.functional.softmax(self.y,dim=-1)
        return self.y

    def ensemble_forward(self,x):

        self.get_samples(x)
        mean = torch.mean(self.ensemble,axis = 0)
        if self.return_uncs:
            d_uncs = self.get_unc()
            return mean,d_uncs
        if self.softmax:
            mean = nn.functional.softmax(mean,dim=-1)
        return mean
        
    def forward(self,x):
        if self.as_ensemble:
            y = self.ensemble_forward(x)
        else:
            y = self.deterministic(x)
            self.get_samples(x) #needed if get_unc get called, as usually does
        if self.return_uncs:
            d_uncs = self.get_unc()
            return y,d_uncs
        return y

    def get_unc(self, x = None):
        if x is not None:
            self.get_samples(x)
            self.deterministic(x)
        d_uncs = {}
        for name,fn in self.uncs.items():
            if name == 'Var(Max)' and self.as_ensemble is False:
                d_uncs[name] = fn(self.ensemble,torch.argmax(self.y,dim = -1)) 
            else:    
                d_uncs[name] = fn(self.ensemble)
        return d_uncs

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.model.load_state_dict(state_dict, strict)


class DeepEnsemble(Ensemble):

    def __init__(self,models, #models for deep ensemble
                 return_uncs:bool = False, #return tuple (output,uncs_dict)
                 softmax = False, #apply SM to ensemble output
                 name = 'DeepEnsemble'
                 ):
        if isinstance(models,dict):
            self.models_dict = models
            models = tuple(models.values())
        super().__init__(models,return_uncs, softmax = softmax, name = name)
        self.p = nn.Parameter(torch.tensor(0.5,requires_grad = True)) #dummy parameter
        self.device = next(self.model[0].parameters()).device
    
    def to(self,device):
        super(Ensemble,self).to(device)
        for model in self.model:
            model.to(device)
        self.device = device
        return self
    def eval(self):
        super(Ensemble,self).eval()
        for model in self.model:
            model.eval()

    def apply_softmax(self, method = 'all'):
        if method == 'all':
            for model in self.model:
                model.softmax = True
        elif method == 'final':
            for _,model in self.model:
                model.softmax = False
            self.softmax = True
        
    def get_samples(self,x):
        ensemble = []
        for model in self.model:
            pred = model(x)
            ensemble.append(pred)
        self.ensemble = torch.stack(ensemble)
        return self.ensemble

    def deterministic(self,x):
        model = self.model[0]
        y = model(x)
        if self.softmax and not model.softmax:
            y = torch.nn.functional.softmax(y,dim=-1)
        return y