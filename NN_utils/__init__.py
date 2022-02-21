import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import torch
import numpy as np
def dataset_cut_classes(data,indices = (0,1)):

    '''Get a dataset (in default form from Pytorch) and returns only the ones with label (target)
    in indices tuple. 
    '''
    
    idx = np.isin(data.targets,indices)
    data.targets= (np.asarray(data.targets)[idx]).tolist()
    data.data = data.data[idx]
    return data

def indexing_2D(ar,idx):
    ''' Index a 2D tensor by a 1D tensor along dimension 1.'''
    return ar[np.arange(len(ar)), idx]
    
def array_to_binary(ar, invert = False):
    '''Convert an array (ar) with 2 classes to an binary array - Change the value assigned to the classes
    to 0 and 1.
    If invert, the first (sorted by number assigned to class) class will be equal to 1
    and the second onde equal to 0. Otherwise, the opposite is made.
    '''
    ar = np.asarray(ar)
    labels = np.unique(ar)
    if not (1 in labels and 0 in labels):
        ar[ar == labels[0]] = int(invert)
        ar[ar == labels[1]] = int(not invert)
    return ar
    
def dataset_to_binary(data,indices = (0,1), invert = False):
    '''Get a dataset (in default form from Pytorch) and returns only the ones with label (target)
    in indices tuple. Then, applys a binarization in it: Convert the 2 classes tobinary - 
    Change the value assigned to the classes to 0 and 1.

    If invert, the first (sorted by number assigned to class) class will be equal to 1
    and the second onde equal to 0. Otherwise, the opposite is made.
    '''
    
    labels = np.unique(data.targets)
    if len(labels) != 2:
        data = dataset_cut_classes(data,indices)
        
    data.targets = array_to_binary(data.targets, invert).tolist()
        
    return data

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.'''
    ave_grads = []
    max_grads= []
    layers = []
    for n,p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    #plt.xlim(left=0, right=len(ave_grads))
    #plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([matplotlib.lines.Line2D([0], [0], color="c", lw=4),
                matplotlib.lines.Line2D([0], [0], color="b", lw=4),
                matplotlib.lines.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])




def apply_mask(y_pred,y_true,mask):
    '''For a given mask, returns only the predictions and targets where the mask is equal to 1'''
    with torch.no_grad():
        mask = mask.bool()
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        return y_pred, y_true

def min_lists(a,b):
    '''Given 2 arrays with same size, returns an array which each element is the minimum
    between a and b in that position'''
    a,b = np.asarray(a),np.asarray(b)
    min_ = np.where(a < b, a, b)
    return min_

def round_decimal(y,n_digits = 1):
    '''Round a tensor with n_digits digits'''
    if not torch.is_tensor(y):
        y = torch.Tensor([y])
    rounded = torch.round(y * 10**n_digits) / (10**n_digits)
    return rounded
def normalize(x, min_x= None,max_x = None):
    '''Normalize an array to be in 0 to 1 interval'''
    if min_x is None:
        min_x = torch.min(x)
    if max_x is None:
        max_x = torch.max(x)

    normalized = (x-min_x)/(max_x-min_x)
    return normalized

def freeze_params(model, layers = None):
    ''' Set requires_grad of model parameters with name inlayers to False.
    Freeze parameters to avoid training.'''
    for n,param in model.named_parameters():
        if layers is None:
            param.requires_grad = False
        elif any(name in n for name in layers):
            param.requires_grad = False
            
def unfreeze_params(model, layers = None):
    ''' Set requires_grad of model parameters with name in layers to True.
    Freeze parameters, turning it possible to train. '''
    
    for n,param in model.named_parameters():
        if layers is None:
            param.requires_grad = True
        elif any(name in n for name in layers):
            param.requires_grad = True
            
def model_eval_layers(model,layers):
    '''Set layers to eval mode'''
    for n,p in model.named_modules():
        #If some element in layers is a string, 
        #implement loop in 'named_modules' and apply in those with the desired name
        if n in layers:
            p.eval()
    for layer in layers: #if some element in layers is the layer module, apply train directly
        if not isinstance(layer,str):
            layer.eval()
        
def model_train_layers(model,layers):
    '''Set layers to train mode'''
    for n,p in model.named_modules(): 
        #If some element in layers is a string, 
        #implement loop in 'named_modules' and apply in those with the desired name
        if n in layers:
            p.train()
    for layer in layers: #if some element in layers is the layer module, apply train directly
        if not isinstance(layer,str):
            layer.train()
        
def ignore_layers(model,layers,reset = True):
    '''Set layers to train mode and freeze them (set requires grad to False).
    If reset, set to train mode and unfreeza all layers of model before ignore'''
    if reset:
        unfreeze_params(model)
        model.train()
    model_eval_layers(model,layers)
    freeze_params(model,layers)

def pond_sum(a,b,alpha,beta):
    return a*alpha+b*beta

def pond_mean(a,b,alpha,beta):
    return pond_sum(a,b,alpha,beta)/(alpha+beta)

def is_probabilities(y, tol = 1e-5, dim = -1):
    '''Check if tensor y can be considered as a probabilite tensor, i.e., 
    if it sums to 1 (with float tol) and have all values greater than 0'''

    is_prob = torch.logical_and(torch.all(torch.abs((torch.sum(y,dim=dim) - 1)) < tol),
    torch.all(y>0)) 

    return is_prob



