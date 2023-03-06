import torch
from NN_utils import apply_mask,get_n_biggest,indexing_3D, is_number
import numpy as np
from operator import xor
from NN_utils.train_and_eval import correct_total
from collections import defaultdict



def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
def enable_BatchNormalization(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('BatchNorm'):
            m.train()
            m.track_running_stats = True
def disable_BatchNormalization(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('BatchNorm'):
            m.training = model.training
            m.track_running_stats = False

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
    assert xor(is_number(coverage),is_number(threshold))
    if threshold is None:
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

def accumulate_results(model,data, output_and_label = (True,True)):
    '''Accumulate output (of model), label and the uncertainties dict of a entire dataset.'''
    dev = next(model.parameters()).device
    uncs = defaultdict(torch.Tensor)
    with torch.no_grad():
        output_list = torch.Tensor([]).to(dev)
        label_list = torch.Tensor([]).to(dev)
        for image,label in data:
            image,label = image.to(dev,non_blocking=True), label.to(dev,non_blocking=True)
            if output_and_label[1]:
                label_list = torch.cat((label_list,label))

            output = model(image)
            d_uncs = model.get_unc()
            if output_and_label[0]:
                output_list = torch.cat((output_list,output))
            for name,unc in d_uncs.items():
                try:
                    uncs[name] = torch.cat((uncs[name],unc))
                except:
                    uncs[name] = torch.cat((uncs[name].to(dev),unc))

    return output_list,label_list.long(),uncs

def accumulate_ensemble_results(model,data):
    '''Accumulate output (of ensemble model) and label of a entire dataset.'''
    dev = next(model.parameters()).device

    output_list = torch.Tensor([]).to(dev)
    label_list = torch.Tensor([]).to(dev)
    with torch.no_grad():
        for image,label in data:
            image,label = image.to(dev), label.to(dev)
            output = model(image)

            label_list = torch.cat((label_list,label))
            output_list = torch.cat((output_list,output),dim=1)
        
    return output_list,label_list.long()