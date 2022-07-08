import numpy as np
import torch
from NN_utils import train_and_eval as TE
from sklearn.metrics import roc_curve as ROC
from NN_utils import apply_mask
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from itertools import cycle
import uncertainty as unc
from uncertainty import utils as unc_utils
from sklearn.metrics import auc,brier_score_loss
#from sklearn.calibration import calibration_curve as sk_calibration_curve



def acc_coverage(y_pred,y_true, uncertainty, coverage):
    '''Returns the total accuracy of model in some dataset excluding the 1-c most uncertain samples'''
    dk_mask = unc_utils.dontknow_mask(uncertainty, coverage)
    y_pred, y_true = apply_mask(y_pred,y_true,1-dk_mask)
    acc = TE.correct_total(y_pred,y_true)/y_true.size(0)
    return acc

def error_coverage(y_pred,y_true, uncertainty, coverage):
    '''Returns the 0-1 loss of model in some dataset excluding the 1-c most uncertain samples'''
    return 1-acc_coverage(y_pred,y_true, uncertainty, coverage)

def RC_curve(y_pred,y_true,uncertainty, risk = error_coverage, c_list = np.arange(0,1,0.05)):
    ''' Returns an array with the accuracy of the model in the data dataset
     excluding the most uncertain (total number set by the coverage) samples.
     Each item in the output array is the accuracy when the coverage is given by same item in c_list'''

    risk_list = np.array([])
    with torch.no_grad():
        for c in c_list:
            r = risk(y_pred,y_true, uncertainty, c)
            risk_list = np.append(risk_list,r)
    return risk_list

def ROC_curve(output,y_true, uncertainty, return_threholds = False):
    if callable(uncertainty):
        uncertainty = uncertainty(output)
    y_true = TE.correct_class(output,y_true).cpu().numpy()
    fpr, tpr, thresholds = ROC(y_true,uncertainty.cpu().numpy())
    if return_threholds:
        return fpr,tpr,thresholds
    else:
        return fpr,tpr

def AURC(y_pred,y_true,uncertainty, risk = error_coverage, c_list = np.arange(0,1,0.05)):
    risk_list = RC_curve(y_pred,y_true,uncertainty, risk, c_list)
    return auc(c_list,risk_list)

def AUROC(output,y_true,uncertainty):
    fpr,tpr = ROC_curve(output,y_true,uncertainty)
    return auc(fpr, tpr)

def Precision_Recall():
    pass
def AUPR():
    pass

    

def optimum_RC(y_pred,y_true,risk = error_coverage, c_list = np.arange(0,1,0.05)):
    uncertainty = 1-TE.correct_class(y_pred,y_true)
    return RC_curve(y_pred,y_true,uncertainty, risk, c_list)

def E_AURC(y_pred,y_true,uncertainty, risk = error_coverage, c_list = np.arange(0,1,0.05)):
    aurc = AURC(y_pred,y_true,uncertainty, risk, c_list)
    opt_aurc = auc(c_list,optimum_RC(y_pred,y_true,risk, c_list))
    return aurc - opt_aurc

def Brier(y_pred,y_true,uncertainty):
    wrong = 1-TE.correct_class(y_pred,y_true)
    return brier_score_loss(wrong, uncertainty)

def calibration_curve(
    y_true,
    y_prob,
    normalize=False,
    n_bins=10,
    strategy="uniform",
    return_ece = False
):
    
    if normalize:  # Normalize predicted values into interval [0, 1]
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            f"Only binary classification is supported. Provided labels {labels}."
        )

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    binids = np.searchsorted(bins[1:-1], y_prob)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))

    if return_ece:
        return prob_true, prob_pred, ece
    else:    
        return prob_true, prob_pred

def ECE(y_true,
    y_prob,
    normalize=False,
    n_bins=5,
    strategy="uniform"):

    _,_, ece = calibration_curve(
    y_true,
    y_prob,
    normalize,
    n_bins,
    strategy,
    return_ece = True)
    
    return ece




class selective_metrics():
    LINES = ["-","--","-.",":"]
    FIGSIZE = (8,6)
    LABEL_FONTSIZE = 18
    TICKS_FONTSIZE = 12
    LINEWIDTH = 3
    SoftMax_uncs = {'MCP': unc.MCP_unc,
                    'Entropy': unc.entropy}  #finalizar isso
    def __init__(self,model,dataset, c_list = np.arange(0,1,0.05)) -> None:
        self.c_list = c_list
        self.d_uncs = {}
        if callable(getattr(model, "get_unc", None)):
            self.output,self.label,self.d_uncs = unc_utils.accumulate_results(model,dataset)
        else:
            self.output,self.label = TE.accumulate_results(model,dataset)
        self.add_uncs(self.SoftMax_uncs)

        #if isinstance(self.output,tuple):
        #    self.output,self.g = self.output
    def add_uncs(self,unc_fn:dict):
        for name,un in unc_fn.items():
            if callable(un):
                self.d_uncs[name] = un(self.output)
            else:
                self.d_uncs[name] = un
        
    def RC_curves(self,uncs: dict = {},risk = error_coverage):
        self.risk = {}
        for name,un in self.d_uncs.items():
            self.risk[name] = RC_curve(self.output,self.label,un,risk, self.c_list)
        for name,un in uncs.items():
            self.risk[name] = RC_curve(self.output,self.label,un,risk, self.c_list)
        
        return self.risk

    def plot_RC(self,aurc = False,*args):
        figure(figsize=self.FIGSIZE, dpi=80)
        self.RC_curves(*args)

        linecycler = cycle(self.LINES)
        for name,risk in self.risk.items():
            label = name+f' AURC = {auc(self.c_list,risk)}' if aurc else name
            plt.plot(self.c_list,risk, label = label, linewidth = self.LINEWIDTH,linestyle = next(linecycler))
        
        plt.legend()
        plt.xlabel("Coverage", fontsize=self.LABEL_FONTSIZE)
        plt.ylabel("Risk", fontsize=self.LABEL_FONTSIZE)
        plt.xticks(fontsize=self.TICKS_FONTSIZE)
        plt.yticks(fontsize=self.TICKS_FONTSIZE)
        plt.grid()
        #plt.show()

    def ROC_curves(self,uncs: dict = {}):

        self.ROC = {}
        y_true = TE.correct_class(self.output,self.label).cpu().numpy()
        for name,un in self.d_uncs.items():
            fpr, tpr, _ = ROC(y_true,un.cpu().numpy())
            self.ROC[name] = (fpr,tpr)
        for name,un in uncs.items():
            fpr, tpr, _ = ROC(y_true,un.cpu().numpy())
            self.ROC[name] = (fpr,tpr)
        return self.ROC
    def plot_ROC(self, auroc = True,*args):
        figure(figsize=self.FIGSIZE, dpi=80)
        self.ROC_curves(*args)

        linecycler = cycle(self.LINES)
        for name,risk in self.ROC.items():
            (fpr,tpr) = self.ROC[name]
            label = name+f' AUROC = {auc(fpr,tpr)}' if auroc else name
            plt.plot(fpr,tpr, label = label, linewidth = self.LINEWIDTH,linestyle = next(linecycler))
        
        plt.legend()
        plt.xlabel("False Positive Rate", fontsize=self.LABEL_FONTSIZE)
        plt.ylabel("True Positive Rate", fontsize=self.LABEL_FONTSIZE)
        plt.xticks(fontsize=self.TICKS_FONTSIZE)
        plt.yticks(fontsize=self.TICKS_FONTSIZE)
        plt.grid()
        plt.show()

    def E_AURC(self):
        d = {}
        for name,risk in self.risk.items():
            AURC = {auc(self.c_list,risk)}
            opt_aurc = auc(self.c_list,optimum_RC(self.output,self.label,risk, self.c_list))
            EAURC = AURC-opt_aurc
            d[name] = EAURC
        return d
    def AURC(self):
        pass


