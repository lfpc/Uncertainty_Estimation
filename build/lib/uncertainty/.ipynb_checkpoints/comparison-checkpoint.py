import torch
import numpy as np
from NN_utils.train_and_eval import correct_total
from NN_utils import apply_mask

def dontknow_mask(y_pred, uncertainty, coverage = 0.1):
    '''Returns a DontKnow Tensor: 1 for the most (coverage most) uncertain samples
    and 0 for the rest'''
    with torch.no_grad():
        n_pred = y_pred.shape[0]


        num_cut = int((coverage)*n_pred)
        dontknow = torch.zeros(n_pred)
        argsort_uncertainty = torch.argsort(uncertainty, descending=True)
        for i in range(num_cut):
            p = argsort_uncertainty[i]
            dontknow[p] = 1

        return dontknow

def model_acc_mask(model,data,unc_fn,c):
    '''Returns the total accuracy of model in some dataset excluding the c most uncertain samples'''
    model.eval()
    dev = next(model.parameters()).device
    total = 0
    correct= 0
    for image,label in data:
        image,label = image.to(dev), label.to(dev)

        y = torch.exp(model(image))

        unc = unc_fn(model,image,y)

        dk = dontknow_mask(y,uncertainty = unc, coverage = c)
        dk = dk.to(dev)

        y, label = apply_mask(y,label,1-dk)

        total += label.size(0)
        correct += correct_total(y,label)

    return (correct*100/total)

def acc_x_coverage(model,unc_fn, data, c_list = np.arange(0,1,0.05)):
    ''' Returns an array with the accuracy of the model in the data dataset
     excluding the most uncertain (total number set by the coverage) samples.
     Each item in the output array is the accuracy when the coverage is given by same item in c_list'''

    model.eval()
    acc_list = np.array([])
    for c in c_list:
        acc = model_acc_mask(model,data,unc_fn,c)
        acc_list = np.append(acc_list,acc.item())

    return acc_list



    
