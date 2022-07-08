import uncertainty as unc
import uncertainty.utils as unc_utils
import torch
from . import ensemble

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

class MonteCarloDropout(ensemble.Ensemble):
    def __init__(self,model, n_samples, as_ensemble = True,return_uncs = False):
        super().__init__(models_dict = {'model':model}, return_uncs= return_uncs)
        self.model = model
        self.n_samples = n_samples
        self.as_ensemble = as_ensemble
        if self.as_ensemble:
            self.uncs['MCP (MCD)'] = lambda x: unc.MCP_unc(torch.mean(x,axis = 0))
            self.uncs['Entropy (MCD)'] = lambda x: unc.entropy(torch.mean(x,axis = 0))
            
        self.set_dropout()

    def set_dropout(self):
        self.model.eval()
        unc_utils.enable_dropout(self.model)

    def get_samples(self,x, enable = False):
        self.ensemble = mcd_pred(self.model,x,self.n_samples, enable=enable)
        return self.ensemble

    def deterministic(self,x):
        self.eval()
        y = self.model(x)
        if self.return_uncs:
            self.set_dropout()
            d_uncs = self.get_unc()
            return y,d_uncs
        return y

    def forward(self,x):
        if self.as_ensemble:
            y = super().forward(x)
        else:
            y = self.deterministic(x)
        return y

if __name__ == "__main__":
    import NN_models
    import cifar_data
    model = NN_models.Model_CNN()
    mcd = MonteCarloDropout(model,10)
    data = cifar_data.Cifar_10_data()
    x,_ = data.get_sample()
    output = mcd(x)
    print(output.shape)


