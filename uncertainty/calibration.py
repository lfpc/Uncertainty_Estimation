
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.nn.functional import softmax
from uncertainty import get_MCP
from NN_utils.train_and_eval import correct_class


class Platt_Model(torch.nn.Module):
    def __init__(self,model,A = 1.0,B = 0.0):
        '''Model with outputs z' = Az+ B, where z is the logits vector output of the main model'''
        super().__init__()
        self.model = model
        self.A = torch.nn.Parameter(torch.tensor(A,requires_grad = True))
        self.B = torch.nn.Parameter(torch.tensor(B,requires_grad = True))
        self.to(next(model.parameters()).device)
    def forward(self,x):
        logits = self.model(x)
        return logits*self.A + self.B

'''From https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py :'''
class _ECELoss(torch.nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15, softmax = False):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.SM = softmax

    def forward(self, logits, labels):
        if self.SM:
            y = softmax(logits, dim=1)
        confidences, predictions = torch.max(y, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class Binning():
    def __init__(self,p,hits,n = 10, bounds = (0,1),division = 'normal'):
        
        self.n = n
        if division == 'adaptative':
            self.bounds = self.Adaptative_Binning_Bounds(p,n,bounds).cpu()
        else:
            self.bounds = torch.linspace(bounds[0],bounds[1],n+1)
        self.accs = self.calc_acc_bins(p,hits)
        self.confs = self.calc_conf_bins(p)
        self.lens = self.calc_len_bins(p)
        self.ece = self.__ECE(p)  
        
    @classmethod
    def from_model(cls,model,dataloader,method = get_MCP,n = 10, bounds = (0,1),division = 'normal'):
        dev = next(model.parameters()).device
        p = torch.tensor([],device = dev)
        hits = torch.tensor([],device = dev)
        with torch.no_grad():
            for im,label in dataloader:
                im,label = im.to(dev),label.to(dev)
                output = model(im)
                p1 = method(output)
                p = torch.cat((p,p1))
                h = correct_class(output,label)
                hits = torch.cat((hits,h))
        return cls(p,hits,n,bounds,division)
    @staticmethod
    def Adaptative_Binning_Bounds(p,n=10,bounds = (0,1)):
        q = torch.linspace(bounds[0],bounds[1],n+1, device = p.device)
        div = torch.quantile(p,q)
        div[0] = bounds[0]
        div[-1] = bounds[1]
        return div
    def get_bin(self,p):
        return np.searchsorted(self.bounds,p)-1
    
    def calc_acc_bins(self,p,hits):
        with torch.no_grad():
            accs = dict.fromkeys(range(self.n),None)
            for i in accs.keys():
                b = torch.logical_and(self.bounds[i]<p,p<=self.bounds[i+1])
                hits_bin = hits[b]
                accs[i] = hits_bin.sum().item()/hits_bin.numel() if hits_bin.numel()!=0 else None
        return accs
    def calc_conf_bins(self,p):
        with torch.no_grad():
            confs = dict.fromkeys(range(self.n),None)
            for i in confs.keys():
                b = torch.logical_and(self.bounds[i]<p,p<=self.bounds[i+1])
                confs[i] = torch.mean(p[b]).item()
        return confs
    def calc_len_bins(self,p):
        lens = dict.fromkeys(range(self.n),None)
        for i in lens.keys():
            b = torch.logical_and(self.bounds[i]<p,p<=self.bounds[i+1])
            lens[i] = p[b].numel()
        return lens        
    def get_calibrated_prob(self,p):
        bins = self.get_bin(p.cpu())
        return torch.tensor([self.accs[b.item()] for b in bins])
    
    def plot_bins(self):
        plt.plot(self.confs.values(),self.accs.values(),linewidth = 2)
        plt.plot(self.bounds,self.bounds,'k--',label = 'Perfectly Calibrated')
        plt.grid()
        plt.legend()
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Calibration curve - ECE = {self.ece:.4f}')
        
    @staticmethod
    def plot_hist(p):
        plt.hist(p.cpu().numpy())
    
    def __ECE(self,p):
        ece = 0
        for b,acc in self.accs.items():
            if acc is not None:
                ece += np.abs(acc-self.confs[b])*self.lens[b]/sum(self.lens.values())
        return ece