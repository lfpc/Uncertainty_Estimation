import torch
import numpy as np
from NN_utils.train_and_eval import correct_total
from NN_utils import apply_mask

def dontknow_mask(uncertainty, coverage = 0.1):
    '''Returns a DontKnow Tensor: 1 for the most (coverage most) uncertain samples
    and 0 for the rest'''
    with torch.no_grad():
        n_pred = uncertainty.shape[0]


        num_cut = int((coverage)*n_pred)
        dontknow = torch.zeros(n_pred)
        argsort_uncertainty = torch.argsort(uncertainty, descending=True)
        for i in range(num_cut):
            p = argsort_uncertainty[i]
            dontknow[p] = 1

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
