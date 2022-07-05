import numpy as np
import torch
from NN_utils import indexing_2D, is_probabilities,round_decimal
import utils
from NN_utils import train_and_eval as TE
from sklearn.metrics import roc_curve as ROC
from NN_utils import apply_mask
from NN_utils.train_and_eval import correct_total
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from itertools import cycle
import uncertainty as unc
from sklearn.metrics import auc,brier_score_loss
from sklearn.calibration import sk_calibration_curve



def acc_coverage(y_pred,y_true, uncertainty, coverage):
    '''Returns the total accuracy of model in some dataset excluding the 1-c most uncertain samples'''
    dk_mask = utils.dontknow_mask(uncertainty, coverage)
    y_pred, y_true = apply_mask(y_pred,y_true,1-dk_mask)
    acc = correct_total(y_pred,y_true)/y_true.size(0)
    return acc

def error_coverage(y_pred,y_true, uncertainty, coverage):
    '''Returns the 0-1 loss of model in some dataset excluding the 1-c most uncertain samples'''
    return 1-acc_coverage(y_pred,y_true, uncertainty, coverage)

def RC_curve(y_pred,y_true,uncertainty, risk = error_coverage, c_list = np.arange(0,1,0.05)):
    ''' Returns an array with the accuracy of the model in the data dataset
     excluding the most uncertain (total number set by the coverage) samples.
     Each item in the output array is the accuracy when the coverage is given by same item in c_list'''

    risk_list = np.array([])
    for c in c_list:
        risk = risk(y_pred,y_true, uncertainty, c)
        rick_list = np.append(risk_list,risk)

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
    opt_aurc = auc(optimum_RC(y_pred,y_true,risk, c_list))
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
    LINE_WIDTH = 3
    def __init__(self,model,dataset, uncertainties = None) -> None:

        self.output,self.label = TE.accumulate_results(model,dataset)
        if isinstance(self.output,tuple):
            self.output,self.g = self.output

    def RC_curves(self,uncs: dict,risk = error_coverage, c_list = np.arange(0,1,0.05)):
        self.risk = {}
        for name,un in uncs.items():
            self.risk[name] = RC_curve(self.output,self.label,un,risk, c_list)
        self.c_list = c_list
        return self.risk

    def plot_RC(self,AURC = False,*args):
        figure(figsize=self.FIGSIZE, dpi=80)
        if args != ():
            self.RC_curves(*args)
        linecycler = cycle(self.LINES)
        for name,risk in self.risk.items():
            label = name if AURC else name+f' AURC = {auc(risk,self.c_list)}'
            plt.plot(self.c_list,risk, label = name, linewidth = self.LINEWIDTH,linestyle = next(linecycler))
        
        plt.legend()
        plt.xlabel("Coverage", fontsize=self.LABEL_FONTSIZE)
        plt.ylabel("Risk", fontsize=self.LABEL_FONTSIZE)
        plt.xticks(fontsize=self.TICKS_FONTSIZE)
        plt.yticks(fontsize=self.TICKS_FONTSIZE)
        plt.grid()
        plt.show()



        #plt.plot(fpr, tpr, label = f'MCP - AUC = {auc(fpr, tpr)}')
