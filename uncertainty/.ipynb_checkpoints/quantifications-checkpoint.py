import numpy as np
import torch
from NN_utils import indexing_2D,round_decimal

def get_MCP(y):
    ''' Returns the Maximum Class/Softmax Probability of a predicted output.
    Returns the value of the probability of the class with more probability'''
    return torch.max(y,-1).values

def MCP_unc(y):
    '''Returns the Maximum Class/Softmax Probability of a predicted output
     as an uncertainty estimation, since MCP is a certainty quantification.
    '''
    return 1-get_MCP(y)


def get_TCP(y_pred,y_true):
    ''' Returns the True Class/Softmax Probability of a predicted output.
    Returns the value of the probability of the class that is true'''
    TCP = indexing_2D(y_pred,y_true)
    return TCP

def TCP_unc(y,label):
    '''Returns the True Class Probability of a predicted output
     as an uncertainty estimation, since TCP is a certainty quantification.
    '''
    return (1-get_TCP(y,label))


def entropy(y_pred, reduction = 'mean'):
    '''Returns the entropy of a probabilities tensor.'''
    entropy = -y_pred*torch.log(y_pred)
    entropy = torch.sum(entropy,-1)
    
    if reduction == 'mean':
        entropy = torch.mean(entropy)
    elif reduction == 'sum':
        entropy = torch.sum(entropy)
        
    return entropy

def mutual_info(pred_array):
    '''Returns de Mutual Information (Gal, 2016) of a probability tensor'''
    ent = entropy(torch.mean(pred_array, axis=0))
    MI = ent - torch.mean(entropy(pred_array), axis=0) 
    return MI

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def dropout_pred(model,X):
    '''Enable Dropout in the model and evaluate one prediction'''
    model.eval()
    enable_dropout(model)
    output = (model(X))
    return output

def montecarlo_pred(model,X,n=10):
    '''Returns an array with n evaluations of the model with dropout enabled.'''
    with torch.no_grad():  
        MC_array = []
        for i in range(n):
            pred = dropout_pred(model,X)
            MC_array.append(pred)
        MC_array = torch.stack(MC_array)
    return MC_array

def MonteCarlo_var(MC_array):
    '''Returns the average variance of a tensor'''
    var = torch.var(MC_array, axis=0) 
    var = torch.mean(var,axis= -1)
    return var

def get_MCD(model,X,n=10):

    '''Evaluates n predictions on input with dropout enabled and
     returns the mean, variance and mutual information
    of them. '''
    MC_array = montecarlo_pred(model,X,n = n)
    
    mean = torch.mean(MC_array, axis=0)
    
    var = MonteCarlo_var(MC_array)
        
    MI = mutual_info(MC_array) 

    return mean, var, MI

class MonteCarloDropout(torch.nn.Module):

    def __init__(self, model, n=10):
        super().__init__()
        self.model = model
        self.n = n

    def forward(self,X):
        self.MC_array = montecarlo_pred(X,self.n)
        mean = torch.mean(self.MC_array, axis=0) 
        return mean

    def get_var(self):
        var = MonteCarlo_var(self.MC_array)
        return var
    def get_MI(self):
        MI = mutual_info(self.MC_array)
        return MI