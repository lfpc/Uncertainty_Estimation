from tkinter import FALSE
import numpy as np
import torch
from NN_utils import train_and_eval as TE
from sklearn.metrics import roc_curve as ROC
from NN_utils import slice_dict
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from itertools import cycle
import uncertainty as unc
from uncertainty import MCP_unc, utils as unc_utils
from sklearn.metrics import auc,brier_score_loss
from scipy.stats import spearmanr,pearsonr
from pandas import DataFrame
#from sklearn.calibration import calibration_curve as sk_calibration_curve


class log_NLLLoss(torch.nn.NLLLoss):
    def __init__(self, reduction: str = 'mean',eps = 1e-20) -> None:
        super().__init__(reduction = reduction)
        self.eps = eps
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward((input+self.eps).log(), target)

def RC_curve_old(y_pred,y_true,uncertainty,risk = TE.wrong_class, coverages = torch.arange(0.01,1.01,0.01)):
    ''' Returns an array with the accuracy of the model in the data dataset
     excluding the most uncertain (total number set by the coverage) samples.
     Each item in the output array is the accuracy when the coverage is given by same item in coverages'''
    N = y_true.size(0)
    r = risk(y_pred,y_true)
    r = r[uncertainty.argsort(descending=False)]
    risk_list = []
    with torch.no_grad():
        for c in coverages:
            cN = torch.floor(c*N).int()
            risk_list.append((r[:cN].sum()/cN).item())
    return risk_list

def RC_curve(y_pred:torch.tensor,y_true:torch.tensor,uncertainty = None,risk_fn = TE.wrong_class,
                coverages = None, expert=False, expert_cost=0,confidence = None,return_coverages = False):
    risk = risk_fn(y_pred,y_true)
    if uncertainty is None:
        if callable(confidence):
            confidence = confidence(y_pred)
        uncertainty = -confidence
    elif callable(uncertainty):
        uncertainty = uncertainty(y_pred)
    return RC_curve_raw(risk,uncertainty,coverages,expert,expert_cost,return_coverages=return_coverages)

def RC_curve_raw(loss:torch.tensor, uncertainty:torch.tensor = None,coverages = None, expert=False, expert_cost=0, confidence = None,return_coverages = False):
    loss = loss.view(-1)
    if uncertainty is None:
        if confidence is not None:
            uncertainty = -confidence
    uncertainty = uncertainty.view(-1)
    n = len(loss)
    assert len(uncertainty) == n
    uncertainty,indices = uncertainty.sort(descending = False)
    loss = loss[indices]
    if coverages is not None:
        coverages = torch.as_tensor(coverages,device = uncertainty.device)
        thresholds = uncertainty.quantile(coverages)
        indices = torch.searchsorted(uncertainty,thresholds)
    else:
        thresholds,indices = uncertainty.unique_consecutive(return_inverse = True)
    coverages = (1 + indices)/n
    print(indices)
    risks = (loss.cumsum(0)[indices])/n

    if expert:
        if np.any(expert_cost):
            expert_cost = np.array(expert_cost).reshape(-1)
            if expert_cost.size == 1:
                risks += (1 - coverages)*expert_cost
            else:
                assert len(expert_cost) == n
                expert_cost = np.cumsum(expert_cost)
                expert_cost = expert_cost[-1] - expert_cost
                risks += expert_cost[indices]/n
    else:
        risks /= coverages
    if return_coverages:
        return coverages.cpu().numpy(), risks.cpu().numpy()
    else: return risks.cpu().numpy()


def optimal_RC(y_pred,y_true,risk = TE.wrong_class, coverages = torch.arange(0.05,1.05,0.05),return_coverages = True):
    uncertainty = risk(y_pred,y_true)
    return RC_curve_raw(uncertainty,uncertainty, coverages,return_coverages=return_coverages)

def ROC_curve(output,y_true, uncertainty, return_threholds = False):
    if callable(uncertainty):
        uncertainty = uncertainty(output)
    y_true = TE.wrong_class(output,y_true).cpu().numpy()
    fpr, tpr, thresholds = ROC(y_true,uncertainty.cpu().numpy())
    if return_threholds:
        return fpr,tpr,thresholds
    else:
        return fpr,tpr


def AURC(y_pred,y_true,uncertainty, risk = TE.wrong_class, coverages = torch.arange(0.05,1.05,0.05)):
    coverages,risk_list = RC_curve(y_pred,y_true,uncertainty, risk, coverages,return_coverages = True)
    return auc(coverages,risk_list)

def AURC_raw(loss,uncertainty, coverages = torch.arange(0.05,1.05,0.05)):
    coverages,risk_list = RC_curve_raw(loss,uncertainty, coverages,return_coverages = True)
    return auc(coverages,risk_list)

def AUROC(output,y_true,uncertainty):
    fpr,tpr = ROC_curve(output,y_true,uncertainty)
    return auc(fpr, tpr)

def Precision_Recall():
    pass
def AUPR():
    pass

def correlation(self,a,b,metric = 'spearman'):
    if metric == 'spearman':
        fn = spearmanr
    rho = fn(a,b)    
    return rho

def Brier(y_pred,y_true,n_classes = -1):
    if n_classes == -1:
        n_classes = y_pred.size(-1)
    y_true = torch.nn.functional.one_hot(y_true, n_classes)
    return torch.mean(torch.sum(torch.square(y_pred-y_true),-1))
    #return brier_score_loss(TE.correct_class(y_pred,y_true), 1-uncertainty)

    
class log_NLLLoss(torch.nn.NLLLoss):
    '''Cross Entropy for softmax input'''
    def __init__(self, reduction: str = 'mean',eps = 1e-20) -> None:
        super().__init__(reduction = reduction)
        self.eps = eps
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward((input+self.eps).log(), target)


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

class ECE(torch.nn.Module):
    
    '''From https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py :'''
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
    def __init__(self, n_bins=10, softmax = False):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECE, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.SM = softmax

    def forward(self, y, labels):
        if self.SM:
            y = torch.nn.functional.softmax(y, dim=1)
        confidences, predictions = torch.max(y, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=y.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class AdaptiveECE(torch.nn.Module):
    '''
    Compute Adaptive ECE
    '''
    def __init__(self, n_bins=15):
        super(AdaptiveECE, self).__init__()
        self.nbins = n_bins

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                     np.arange(npt),
                     np.sort(x))
    def forward(self, logits, labels):
        softmaxes = torch.nn.functional.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        #print(n,confidences,bin_boundaries)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
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


class ClasswiseECE(torch.nn.Module):
    '''
    Compute Classwise ECE
    '''
    def __init__(self, n_bins=15):
        super(ClasswiseECE, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        num_classes = int((torch.max(labels) + 1).item())
        softmaxes = torch.nn.functional.softmax(logits, dim=1)
        per_class_sce = None

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            class_sce = torch.zeros(1, device=logits.device)
            labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if (i == 0):
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

        sce = torch.mean(per_class_sce)
        return sce



class selective_metrics():
    LINES = ["-","--","-.",":"]
    FIGSIZE = (8,6)
    LABEL_FONTSIZE = 18
    TICKS_FONTSIZE = 12
    LINEWIDTH = 3
    SoftMax_uncs = {'SR': lambda x: unc.MCP_unc(x,normalize=True),
                    'Entropy': unc.entropy}
    
    def __init__(self,model,
                dataset = None, 
                coverages = np.arange(0.05,1.05,0.05),
                name = None,
                labels = None) -> None:
        if name is None:
            self.name = model.name
        else:
            self.name = name
        self.coverages = coverages
        self.d_uncs = {}
        self.model = model
        if labels is not None:
            self.share_dataset(labels)
        else:
            self.__share_label = False
        self.__share_output = False
        if dataset is not None:
            self.get_uncs(dataset)
        self.fix_scale = False
        self.linecycler = cycle(self.LINES)
        

    def set_uncs(self,uncs):
        self.d_uncs = slice_dict(self.d_uncs,uncs)

    def get_uncs(self,dataset=None,extra_uncs:dict = {}):
        if dataset is None:
            dataset = self.dataset
        if callable(getattr(self.model, "get_unc", None)):
            output,label,self.d_uncs = unc_utils.accumulate_results(self.model,dataset,(not self.__share_output,not self.__share_label))
            if not self.__share_output:
                self.output = output
        else:
            self.output,label = TE.accumulate_results(self.model,dataset)
        if not self.__share_label:
            self.label = label
        self.add_uncs(extra_uncs)
        self.add_uncs(self.SoftMax_uncs)
        return self.d_uncs

    def share_dataset(self,label,output = None):
        self.__share_label = True
        self.label = label
        if output is not None:
            self.__share_output = True
            self.output = output

    def fix_plot_scale(self,x_range = None,y_range= None):
        self.fix_scale = True
        self.x_range = x_range
        self.y_range = y_range

    def config_plot(self, title = True):
        plt.legend()
        plt.xticks(fontsize=self.TICKS_FONTSIZE)
        plt.yticks(fontsize=self.TICKS_FONTSIZE)
        plt.grid()
        if title:
            plt.title(self.name)
        if self.fix_scale:
            plt.ylim(self.y_range)
            plt.xlim(self.x_range)

    def add_uncs(self,unc_fn:dict):
        for name,un in unc_fn.items():
            if callable(un):
                with torch.no_grad():
                    self.d_uncs[name] = un(self.output).view(-1)
            else:
                self.d_uncs[name] = un.view(-1)
        
    def RC_curves(self,risk = TE.wrong_class, optimal = False,baseline = None):
        self.risk = {}
        for name,un in self.d_uncs.items():
            self.risk[name] = RC_curve(self.output,self.label,un,risk, self.coverages)
        if optimal:
            self.risk['Optimal'] = optimal_RC(self.output,self.label,risk, self.coverages)
        if baseline is not None:
            self.risk['Baseline'] = baseline
        
        return self.risk

    def plot_RC(self,aurc = False,**kwargs):
        #adicionar ideal
        figure(figsize=self.FIGSIZE, dpi=80)
        self.RC_curves(**kwargs)
        for name,risk in self.risk.items():
            label = name+f' | AURC = {torch.trapz(risk,x = torch.tensor(self.coverages,device = risk.device), dim = -1).item()}' if aurc else name
            plt.plot(self.coverages,risk.cpu(), label = label, linewidth = self.LINEWIDTH,linestyle = next(self.linecycler))
        
        plt.xlabel("Coverage", fontsize=self.LABEL_FONTSIZE)
        plt.ylabel("Risk", fontsize=self.LABEL_FONTSIZE)
        self.config_plot()


    def ROC_curves(self,baseline = None):

        self.ROC = {}
        y_true = np.logical_not(TE.correct_class(self.output,self.label).cpu().numpy())
        for name,un in self.d_uncs.items():
            fpr, tpr, _ = ROC(y_true,un.cpu().numpy())
            self.ROC[name] = (fpr,tpr)
        if baseline is not None:
            self.ROC['Baseline'] = baseline
        return self.ROC

    def plot_ROC(self, auroc = True,**kwargs):
        figure(figsize=self.FIGSIZE, dpi=80)
        self.ROC_curves(**kwargs)
        
        for name,(fpr,tpr) in self.ROC.items():
            label = name+f' | AUROC = {auc(fpr,tpr)}' if auroc else name
            plt.plot(fpr,tpr, label = label, linewidth = self.LINEWIDTH,linestyle = next(self.linecycler))
        
        plt.xlabel("False Positive Rate", fontsize=self.LABEL_FONTSIZE)
        plt.ylabel("True Positive Rate", fontsize=self.LABEL_FONTSIZE)
        self.config_plot()
    def risk_diference(self,ref = 'MCP'):
        risks_dif = {}
        for name,risk in self.risk.items():
            if ref in self.risk.keys():
                risks_dif[name] = risk-self.risk[ref]
            else:
                risks_dif[name] = risk-ref
        return risks_dif
    def get_best(self):
        best = np.array([r for r in self.risk.values()]).min(axis=0)
        return best

    def E_AURC(self):
        d = {}
        for name,risk in self.risk.items():
            AURC = {auc(self.coverages,risk)}
            opt_aurc = auc(self.coverages,optimal_RC(self.output,self.label,risk, self.coverages))
            EAURC = AURC-opt_aurc
            d[name] = EAURC
        return d
    def AURC(self):
        pass
    def get_thresholds(self):
        self.thresholds = {}
        for name,un in self.d_uncs.items():
            self.thresholds[name] = np.array([np.percentile(un.cpu(),100*c) for c in self.coverages])
    def plot_thresholds(self,normalize = False):
        self.get_thresholds()
        figure(figsize=self.FIGSIZE, dpi=80)
        for name,tau in self.thresholds.items():
            if normalize:
                assert np.all(tau>0), "normalize non positive array"
                tau /= tau.max()
            plt.plot(self.coverages,tau, label = name, linewidth = self.LINEWIDTH,linestyle = next(self.linecycler))

        plt.xlabel("Coverage", fontsize=self.LABEL_FONTSIZE)
        plt.ylabel("Threshold", fontsize=self.LABEL_FONTSIZE)
        self.config_plot()

    def correlation(self,metric = 'spearman'):
        if metric == 'spearman':
            fn = spearmanr
        df = DataFrame(index = self.d_uncs.keys(),columns=self.d_uncs.keys())
        for name,un in self.d_uncs.items():
            for name_2,un_2 in self.d_uncs.items():
                df[name][name_2] = fn(un.cpu().numpy(),un_2.cpu().numpy()).correlation
        return df
    def plot_ROC_and_RC(self, aurc = False, auroc = True, *args):
        self.RC_curves(*args)
        self.ROC_curves(*args)
        f, (ax1, ax2) = plt.subplots(1, 2,figsize=self.FIGSIZE,dpi=80)
        for name,risk in self.risk.items():
            label = name+f' | AURC = {auc(self.coverages,risk)}' if aurc else name
            ax1.plot(self.coverages,risk, label = label, linewidth = self.LINEWIDTH,linestyle = next(self.linecycler))
        ax1.set_title('Risk Coverage')
        ax1.set_xlabel("Coverage", fontsize=self.LABEL_FONTSIZE*0.8)
        ax1.set_ylabel("Risk", fontsize=self.LABEL_FONTSIZE*0.8)
        ax1.legend()
        ax1.tick_params(axis="x",labelsize=self.TICKS_FONTSIZE)
        ax1.tick_params(axis="y",labelsize=self.TICKS_FONTSIZE)
        ax1.grid()
        self.fix_scale = False
        for name,(fpr,tpr) in self.ROC.items():
            label = name+f' | AUROC = {auc(fpr,tpr)}' if auroc else name
            ax2.plot(fpr,tpr, label = label, linewidth = self.LINEWIDTH,linestyle = next(self.linecycler))
        ax2.set_title('ROC curve')
        ax2.set_xlabel("False Positive Rate", fontsize=self.LABEL_FONTSIZE*0.8)
        ax2.set_ylabel("True Positive Rate", fontsize=self.LABEL_FONTSIZE*0.8)
        ax2.legend()
        ax2.tick_params(axis="x",labelsize=self.TICKS_FONTSIZE)
        ax2.tick_params(axis="y",labelsize=self.TICKS_FONTSIZE)
        ax2.grid()
        f.suptitle("Main Title", fontsize=15)