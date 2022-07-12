
import torch
#import albumentation
from ensemble import Ensemble
from copy import copy
import uncertainty as unc


def TestTimeAugmentation():
    pass

class TTA(Ensemble):
    transforms = ()
    def __init__(self, model,n_samples,as_ensemble = True, transforms = transforms,
                 return_uncs=False, softmax=False):
        models_dict = {'model':model}
        super().__init__(models_dict, return_uncs, softmax)
        self.model = model
        self.n_samples = n_samples
        self.as_ensemble = as_ensemble
        self.transforms = transforms
        if not self.as_ensemble:
            self.uncs = copy(self.uncs)
            self.uncs['MCP (MCD)'] = lambda x: unc.MCP_unc(torch.mean(x,axis = 0))
            self.uncs['Entropy (MCD)'] = lambda x: unc.entropy(torch.mean(x,axis = 0))
    def get_samples(self,x):
        self.ensemble = TestTimeAugmentation(self.model,x,self.n_samples, self.transforms)
        return self.ensemble

    def deterministic(self,x):
        y = self.model(x)
        if self.softmax and not self.model.softmax:
            y = torch.nn.functional.softmax(y,dim=-1)
        if self.return_uncs:
            d_uncs = self.get_unc(x)
            return y,d_uncs
        else:
            self.get_samples(x) #needed if get_unc get called, as usually does
            return y
    
    def forward(self,x):
        if self.as_ensemble:
            y = super().forward(x)
        else:
            y = self.deterministic(x)
        return y

