from collections import defaultdict
import torch
import numpy as np
from uncertainty.quantifications import get_MCP
from utils import round_decimal,normalize
from train_and_eval import correct_class



def calibration_curve(model,data):
    dev = next(model.parameters()).device
    p_list = np.around(np.arange(0,1.1,0.1),1)
    d = defaultdict(list)  
    for n in p_list:
        d[n].append(0)
        d[n].append(0)
    model.eval()
    for image,label in data:
        image,label = image.to(dev), label.to(dev)
        
        y = model(image)
        MCP = get_MCP(y)
        
        for n in p_list:
            p = round_decimal(MCP)
            correct = correct_class(y[p==n],label[p==n])
            d[n][0] += correct
            d[n][1] += (p==n).sum().item()
    acc = []
    for n in d:
        if d[n][1] == 0:
            acc.append(np.nan)
        else:
            acc.append(d[n][0]/d[n][1])
    
    return acc

    

def MCP_samples_distribution(model,data):
    dev = next(model.parameters()).device
    p_list = np.around(np.arange(0,1.1,0.1),1)
    d = defaultdict(list)  
    for n in p_list:
        d[n].append(0)
    model.eval()
    for image,label in data:
        image = image.to(dev)
        y = model(image)
        MCP = get_MCP(y)
        
        for n in p_list:
            p = round_decimal(MCP)
            d[n][0] += (p==n).sum().item()

    n_samples = []
    for n in d:
        n_samples.append((d[n][1])/len(data))
    return n_samples

