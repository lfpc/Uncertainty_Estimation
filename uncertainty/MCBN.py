#Monte Carlo Batch Normalization
from uncertainty import ensemble
import torch
from copy import copy
import uncertainty as unc
from collections import defaultdict

class MonteCarloBatchNormalization(ensemble.Ensemble):
    def __init__(self,model, n_samples, batch_loader,
                      as_ensemble = True,return_uncs = False, 
                      softmax = False, name = 'MCBN'):
        self.as_ensemble = as_ensemble
        super().__init__(model, return_uncs= return_uncs, as_ensemble = as_ensemble, softmax = softmax, name=name)
        #assert isinstance(batch_loader.sampler,torch.utils.data.sampler.RandomSampler), "Batch Loader should have shuffle set to True to give randomness"
        self.batch_loader = batch_loader
        self.n_samples = n_samples

        self.__get_BN_modules()
        self.__save_main_attributes()

    def __get_BN_modules(self):
        self.__modules = []
        for m in self.model.modules():
            if m.__class__.__name__.startswith('BatchNorm'):
                self.__modules.append(m)
        assert len(self.__modules)>0, "No BatchNormalization modules found in model"

    def __save_main_attributes(self):
        self.__momentum = {}
        self.__running_mean= {}
        self.__running_var= {}

        for m in self.__modules:
            self.__momentum[m] = copy(m.momentum)
            self.__running_mean[m] = copy(m.running_mean)
            self.__running_var[m] = copy(m.running_var)
    def __set_main_attributes(self):
        for m in self.__modules:
            m.momentum = copy(self.__momentum[m])
            m.running_mean = copy(self.__running_mean[m])
            m.running_var = copy(self.__running_var[m])

    def set_BN_mode(self):
        for m in self.__modules:
            m.train()
            m.track_running_stats = True
            m.momentum = 1
            
    def reset_normal_mode(self):
        self.eval()
        self.__set_main_attributes()


    def get_samples(self,x):
        ensemble = []
        batch_loader = iter(self.batch_loader)
        for _ in range(self.n_samples):
            im_train,_ = next(batch_loader)
            im_train = im_train.to(self.device)
            self.set_BN_mode()
            with torch.no_grad():
                self.model(im_train)
                self.model.eval()
                y = self.model(x)
                ensemble.append(y)
                self.ensemble = torch.stack(ensemble)
        return self.ensemble

    def deterministic(self,x):
        self.reset_normal_mode()
        return super().deterministic(x)

class Fast_MCBN(MonteCarloBatchNormalization):
    def __init__(self, model, n_samples, batch_loader, as_ensemble=True, return_uncs=False, softmax=False, name='MCBN'):
        super().__init__(model, n_samples, None, as_ensemble, return_uncs, softmax, name)
        self.__get_BN_parameters(batch_loader)
        self.reset_normal_mode()

    def __get_BN_parameters(self,batch_loader):
        self.set_BN_mode()
        self.params = defaultdict(list)
        batch_loader = iter(batch_loader)
        for _ in range(self.n_samples):
            im,_ = next(batch_loader)
            im = im.to(self.device)
            with torch.no_grad():
                self.model(im)
                for m in self.__modules:
                    self.params[m].append((m.running_mean,m.running_var))
    def __set_BN_parameters(self,i):
        for m in self.__modules:
            mean,var = self.params[m][i]
            m.running_mean = mean
            m.running_var = var

    def get_samples(self,x):
        ensemble = []
        for i in range(self.n_samples):
            with torch.no_grad():
                self.__set_BN_parameters(i)
                y = self.model(x)
                ensemble.append(y)
                self.ensemble = torch.stack(ensemble)
        return self.ensemble