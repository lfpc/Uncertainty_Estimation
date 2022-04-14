import torch
import numpy as np
from NN_utils.train_and_eval import correct_total
from NN_utils import apply_mask,get_n_biggest


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

def dontknow_mask(uncertainty, coverage = 0.8):
    '''Returns a DontKnow Tensor: 1 for the most (coverage most) uncertain samples
    and 0 for the rest'''
    with torch.no_grad():
        n_pred = uncertainty.shape[0]

        num_cut = round((1-coverage)*n_pred)
        dontknow = torch.zeros(n_pred)
        biggest_unc = get_n_biggest(uncertainty,num_cut)
        dontknow[biggest_unc] = 1
        #for i in range(num_cut):
        #    p = argsort_uncertainty[i]
        #    dontknow[p] = 1

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

def selective_risk(y_pred,label,loss_fn = torch.nn.NLLLoss(),c=0.8,unc_type = None):

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
