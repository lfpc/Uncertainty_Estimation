import torch
from NN_utils import indexing_2D, is_probabilities,round_decimal
import uncertainty.utils as utils

def entropy(y, reduction = 'none'):
    '''Returns the entropy of a probabilities tensor.'''
    
    if not is_probabilities(y): #if y is not a probabilities tensor
        y = torch.nn.functional.softmax(y,dim=-1) #apply softmax
    
    entropy = torch.special.entr(y) #entropy element wise
    entropy = torch.sum(entropy,-1)
    
    
    if reduction == 'mean':
        entropy = torch.mean(entropy)
    elif reduction == 'sum':
        entropy = torch.sum(entropy)
        
    return entropy


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


def get_TCP(y_pred,y_true):
    ''' Returns the True Class/Softmax Probability of a predicted output.
    Returns the value of the probability of the class that is true'''
    if not is_probabilities(y_pred): #if y is not a probabilities tensor
        y_pred = torch.nn.functional.softmax(y_pred,dim=-1) #apply softmax
    TCP = indexing_2D(y_pred,y_true)

    return TCP

def TCP_unc(y,label):
    '''Returns the True Class Probability of a predicted output
     as an uncertainty estimation, since TCP is a certainty quantification.
    '''
    return (1-get_TCP(y,label))