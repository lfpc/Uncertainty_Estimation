from uncertainty import mutual_info
import torch
from NN_utils import apply_mask,get_n_biggest
import numpy as np
from operator import xor
from NN_utils.train_and_eval import correct_total



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
    model.eval()
    enable_dropout(model)
    with torch.no_grad(): 
        MC_array = []
        for i in range(n):
            pred = model(X)
            MC_array.append(pred)
        MC_array = torch.stack(MC_array)
    return MC_array

def MonteCarlo_var(MC_array):
    '''Returns the average variance of a tensor'''
    var = torch.var(MC_array, axis=0) 
    var = torch.mean(var,axis= -1)
    return var

def get_most_uncertain(data,uncertainty_method, n = 1):
    if not isinstance(uncertainty_method,(torch.Tensor,list,np.ndarray)):
        uncertainty_method = uncertainty_method(data)
    dk_vec = get_n_biggest(uncertainty_method,n)
    unc_data = data[dk_vec]
    return unc_data

def get_most_certain(data,uncertainty_method, n = 1):
    if not isinstance(uncertainty_method,(torch.Tensor,list,np.ndarray)):
        uncertainty_method = uncertainty_method(data)
    know_vec = get_n_biggest(-uncertainty_method,n)
    unc_data = data[know_vec]
    return unc_data

def dontknow_mask(uncertainty, coverage = None, threshold = None):
    '''Returns a DontKnow Tensor: 1 for the most (coverage most) uncertain samples
    and 0 for the rest'''
    assert xor(isinstance(coverage,float) or isinstance(coverage,int), isinstance(threshold,float) or isinstance(threshold,int))
    if threshold is None and coverage is not None:
        with torch.no_grad():
            n_pred = uncertainty.shape[0]

            num_cut = round((1-coverage)*n_pred)
            dontknow = torch.zeros(n_pred)
            biggest_unc = get_n_biggest(uncertainty,num_cut)
            dontknow[biggest_unc] = 1
            #for i in range(num_cut):
            #    p = argsort_uncertainty[i]
            #    dontknow[p] = 1
    elif coverage is None:
        with torch.no_grad():
            dontknow = uncertainty>threshold

    return dontknow

def acc_coverage_list(y_pred,y_true,uncertainty, c_list = np.arange(0,1,0.05)):
    ''' Returns an array with the accuracy of the model in the data dataset
     excluding the most uncertain (total number set by the coverage) samples.
     Each item in the output array is the accuracy when the coverage is given by same item in c_list'''

    acc_list = np.array([])
    for c in c_list:
        acc = acc_coverage(y_pred,y_true, uncertainty, c)
        acc_list = np.append(acc_list,acc)

    return acc_list


def acc_coverage(y_pred,y_true, uncertainty, coverage):
    '''Returns the total accuracy of model in some dataset excluding the c most uncertain samples'''
    dk_mask = dontknow_mask(uncertainty, coverage)
    y_pred, y_true = apply_mask(y_pred,y_true,1-dk_mask)
    acc = correct_total(y_pred,y_true)/y_true.size(0)
    return acc

    
def acc_coverage_per_batch(model,data,unc_fn,c):
    '''Returns the total accuracy of model in some dataset
     excluding the c most uncertain samples in each batch'''
    model.eval()
    dev = next(model.parameters()).device
    total = 0
    correct= 0
    for image,label in data:
        image,label = image.to(dev), label.to(dev)

        y = (model(image))

        unc = unc_fn(model,image)

        dk = dontknow_mask(uncertainty = unc, coverage = c)
        dk = dk.to(dev)

        y, label = apply_mask(y,label,1-dk)

        total += label.size(0)
        correct += correct_total(y,label)

    return (correct/total)

def selective_risk(y_pred,label,c,loss_fn = torch.nn.NLLLoss(),unc_type = None):

    if unc_type is None:
        return 0
    elif unc_type == 'g':
        y_pred,g = y_pred
        unc = 1-g
    else:
        unc = unc_type(y_pred)
    dk_mask = dontknow_mask(unc, 1-c)
    y_pred, label = apply_mask(y_pred,label,1-dk_mask)
    risk = loss_fn(y_pred,label)
    return risk
