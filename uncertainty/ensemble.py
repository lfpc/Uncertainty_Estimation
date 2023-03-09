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

def mutual_info(pred_array, **kwargs):
    '''Returns de Mutual Information (Gal, 2016) of a probability tensor'''
    ent = unc.entropy(torch.mean(pred_array, axis=0), **kwargs)
    MI = ent - torch.mean(unc.entropy(pred_array,**kwargs), axis=0) 
    return MI


class Ensemble(nn.Module):
    uncs = {'Var(Mean)': MonteCarlo_meanvar,
            'Var(Max)': MonteCarlo_maxvar,
            'MI': mutual_info}

    def __init__(self,model, #model
                 inference = 'mean', #output as average of models
                apply_softmax:bool = False):
        super().__init__()

        self.inference = inference
        self.model = model
        self.apply_softmax = apply_softmax

        if self.inference == 'deterministic':
            self.uncs = copy(self.uncs)
            self.uncs['SR (Ensemble)'] = lambda x: unc.MCP_unc(torch.mean(x,axis = 0))
            self.uncs['Entropy (Ensemble)'] = lambda x: unc.entropy(torch.mean(x,axis = 0))
    
    def get_samples(self,x):
        '''Default ensemble model is to assume that self.model returns samples'''
        ensemble = self.model(x)
        return ensemble

    def deterministic(self,x):
        self.y = self.model(x)
        return self.y

    def ensemble_forward(self,x):
        ensemble = self.get_samples(x)
        if self.apply_softmax:
            ensemble = torch.nn.functional.softmax(ensemble,dim = -1)
        mean = torch.mean(ensemble,axis = 0)
        return mean
        
    def forward(self,x):
        if self.inference == 'mean':
            return self.ensemble_forward(x)
        elif self.inference == 'samples':
            return self.get_samples(x)
        elif self.inference == 'deterministic':
            #self.get_samples(x) #needed if get_unc get called, as usually does
            return self.deterministic(x)
            

    def get_unc(self, x):
        ensemble = self.get_samples(x)
        d_uncs = {}
        for name,fn in self.uncs.items():
            if name == 'Var(Max)' and self.inference == 'deterministic':
                d_uncs[name] = fn(ensemble,torch.argmax(self.y,dim = -1)) 
            else:    
                d_uncs[name] = fn(ensemble)
        return d_uncs

    def load_state_dict(self, state_dict, **kwargs):
        return self.model.load_state_dict(state_dict, **kwargs)


class DeepEnsemble(Ensemble):

    def __init__(self,models, #models for deep ensemble
                 inference = 'mean',
                 apply_softmax:bool = True, #apply SM to models before mean
                 ):
        if isinstance(models,dict):
            self.models_dict = models
            models = list(models.values())
        super().__init__(models, inference, apply_softmax)
        self.model = torch.nn.ParameterList(models)
    
    def get_samples(self,x):
        ensemble = []
        for model in self.model:
            pred = model(x)
            ensemble.append(pred)
        ensemble = torch.stack(ensemble)
        return ensemble