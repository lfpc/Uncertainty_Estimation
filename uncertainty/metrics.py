import numpy as np
import torch
from NN_utils import indexing_2D, is_probabilities,round_decimal
import utils
from NN_utils import train_and_eval as TE
from sklearn.metrics import roc_curve as ROC
import sklearn


class selective_metrics():
    def __init__(self,model,dataset, c = (1.0,), uncertainties = None) -> None:

        assert hasattr(c, '__iter__'), "c must be iterable, pass it as tuple: c = (0.8,)"
        self.c = c
        self.output,self.label = TE.accumulate_results(model,dataset)
        if isinstance(self.output,tuple):
            self.output,self.g = self.output
        

def RC_curve():
        pass
def ROC_curve(output,y_true):
    fpr, tpr, thresholds = ROC(y_true,unc.get_MCP(output).cpu().numpy())
def AURC():
    pass
def AUROC():
    pass
def E_AURC():
    pass
def Brier():
    pass

