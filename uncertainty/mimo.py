from NN_models.mimo import *
from uncertainty import ensemble

class MIMO_ensemble(ensemble.Ensemble):
    def __init__(self, model, return_uncs=False, softmax=False):
        super().__init__(models_dict = {'model':model}, return_uncs= return_uncs, softmax = softmax)
        self.model = model
        self.n_ensembles = model.ensemble_num

    def get_samples(self,x):
        x = [x for _ in range(self.n_ensembles)]
        x = torch.stack(x)
        self.ensemble = self.model(x)
        return self.ensemble
