import torch
from NN_utils import indexing_2D, is_probabilities,round_decimal
#import uncertainty.utils as utils
from warnings import warn

def entropy(y, **kwargs):
    return Entropy.entropy(y,**kwargs)

def normalized_entropy(y,**kwargs):
    '''Entropy normalized so it returns values between 0 and 1'''
    n_classes = y.shape[-1]
    return entropy(y,**kwargs)/torch.log(n_classes)

class Entropy(torch.nn.Module):
    @staticmethod
    def entropy(y, normalize = False,reduction = 'none'):
        '''Returns the entropy of a probabilities tensor.'''
        
        if y.numel() == 0:
            return torch.nan
        if not normalize and not is_probabilities(y):
            warn(f'Input vector is not probabilty vector')
        #if y is not a probabilities tensor
        if not is_probabilities(y) and normalize: 
            idx = is_probabilities(y).nonzero()
            y = torch.nn.functional.softmax(y,dim=-1) #apply softmax 
        
        entropy = torch.special.entr(y) #entropy element wise
        entropy = torch.sum(entropy,-1)
        
        
        if reduction == 'mean':
            entropy = torch.mean(entropy)
        elif reduction == 'sum':
            entropy = torch.sum(entropy)
            
        return entropy

    def __init__(self,reduction = None, normalization = None) -> None:
        super().__init__()
        if isinstance(normalization,str):
            self.normalization = torch.nn.functional.__dict__[normalization]
        else: self.normalization = normalization
        self.reduction = reduction
    def forward(self,y):
        if self.normalization is not None:
            y = self.normalization(y,dim=-1)
        return entropy(y,False,self.reduction)

    

def get_MCP(y,normalize = False):
    ''' Returns the Maximum Class/Softmax Probability of a predicted output.
    Returns the value of the probability of the class with more probability'''
    if not is_probabilities(y) and normalize: #if y is not a probabilities tensor
        y = torch.nn.functional.softmax(y,dim=-1) #apply softmax

    return torch.max(y,-1).values

def MCP_unc(y,normalize = False):
    '''Returns the Maximum Class/Softmax Probability of a predicted output
     as an uncertainty estimation, since MCP is a certainty quantification.
    '''
    return 1-get_MCP(y, normalize)


def get_TCP(y_pred,y_true, normalize = True):
    ''' Returns the True Class/Softmax Probability of a predicted output.
    Returns the value of the probability of the class that is true'''
    if not is_probabilities(y_pred) and normalize: #if y is not a probabilities tensor
        y_pred = torch.nn.functional.softmax(y_pred,dim=-1) #apply softmax
    TCP = indexing_2D(y_pred,y_true)

    return TCP

def TCP_unc(y,label, normalize = True):
    '''Returns the True Class Probability of a predicted output
     as an uncertainty estimation, since TCP is a certainty quantification.
    '''
    return (1-get_TCP(y,label, normalize))

def energy(z:torch.Tensor, T = 1.0):
    return -T*((z/T).exp().sum(-1).log())
class EnergyFunction(torch.nn.Module):
    def __init__(self,T=1.0,reduction = 'none') -> None:
        super().__init__()
        self.T = T
        self.reduction = reduction
    def forward(self,z:torch.Tensor):
        E = energy(z,self.T)
        if self.reduction == 'mean':
            E = E.mean()
        elif self.reduction == 'sum':
            E = E.sum()
        return E