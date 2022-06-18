from turtle import forward
import uncertainty as unc
import uncertainty.utils as unc_utils
import torch


def dropout_pred(model,X,enable = True):
    '''Enable Dropout in the model and evaluate one prediction'''
    if enable:
        model.eval()
        unc_utils.enable_dropout(model)
    output = (model(X))
    return output

def mcd_pred(model,X,n=10, enable = True):
    '''Returns an array with n evaluations of the model with dropout enabled.'''
    if enable:
        model.eval()
        unc_utils.enable_dropout(model)
    with torch.no_grad(): 
        MC_array = []
        for i in range(n):
            pred = model(X)
            MC_array.append(pred)
        MC_array = torch.stack(MC_array)
    return MC_array


def get_MCD(model,X,n=10):

    '''Evaluates n predictions on input with dropout enabled and
     returns the mean, variance and mutual information
    of them. '''
    MC_array = mcd_pred(model,X,n = n)
    
    mean = torch.mean(MC_array, axis=0)
    
    var = unc_utils.MonteCarlo_meanvar(MC_array)
        
    MI = unc.mutual_info(MC_array) 

    return mean, var, MI

class MonteCarloDropout(torch.nn.Module):
    unc_dict = {
        'MI':unc.mutual_info,
        'mean_var':unc_utils.MonteCarlo_meanvar,
        'max_var':unc_utils.MonteCarlo_maxvar}

    def __init__(self,model, n_samples, return_unc = False):
        super().__init__()
        self.model = model
        self.n_samples = n_samples
        self.return_unc = return_unc
        self.model.eval()
        unc_utils.enable_dropout(model)


    def forward(self,x,return_unc = None):

        if return_unc is None:
            return_unc = self.return_unc

        MC_array = mcd_pred(x,self.n_samples)
        mean = torch.mean(MC_array, axis=0)

        if return_unc:
            var_mean = unc_utils.MonteCarlo_meanvar(MC_array)
            var_max = unc_utils.MonteCarlo_maxvar(MC_array)
            MI = unc.mutual_info(MC_array)
            return mean,var_mean,var_max,MI
        else:
            return mean

def accumulate_results_ensemble(model,data):
    '''Accumulate output (of model) and label of a entire dataset.'''
    dev = next(model.parameters()).device
    output_list = torch.Tensor([]).to(dev)
    var_list = torch.Tensor([]).to(dev)
    MI_list = torch.Tensor([]).to(dev)
    label_list = torch.Tensor([]).to(dev)
    for image,label in data:
        with torch.no_grad():
            image,label = image.to(dev), label.to(dev)
            output = torch.exp(model(image)[0])
            var = model(image)[1]
            MI = model(image)[2]

            label_list = torch.cat((label_list,label))
            var_list = torch.cat((var_list,var))
            MI_list = torch.cat((MI_list,MI))
            output_list = torch.cat((output_list,output))
    return output_list,label_list.long(),var_list,MI_list

