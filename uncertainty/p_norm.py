from . import metrics
import torch
from uncertainty import MCP_unc
import numpy as np

def norm_p_heuristic(outputs:torch.tensor,p):
    norm_p = outputs.norm(dim=-1,p=p).unsqueeze(-1)
    norm_p /= norm_p.mean()
    return outputs/norm_p

class TempAnalysis():
    AURC_fn = lambda x,y: metrics.AURC(x,y,MCP_unc(x))
    c_list = torch.arange(0.05,1.05,0.05)
    T_range = torch.round(torch.cat((torch.arange(0.1,2.0,0.01),torch.arange(2.0,2.5,0.1))),decimals=2)
    def __init__(self,T:float,y_pred:torch.tensor,labels:torch.tensor):
        self.T = T
        self.y_pred = y_pred
        self.labels = labels
    def RC_curve(self, method = MCP_unc) -> torch.tensor:
        return metrics.RC_curve(self.y_pred,self.labels,method(self.y_pred))
    def AURC(self,method = MCP_unc):
        return metrics.AURC(self.y_pred,self.labels,method(self.y_pred))
    def AUROC(self,method = MCP_unc):
        return metrics.AUROC(self.y_pred,self.labels,method(self.y_pred))
    def ECE(self, bins = 10):
        fn = metrics.ECE(n_bins = bins)
        return fn(self.y_pred,self.labels)
    @staticmethod
    def T_grid(outputs,labels,metric = AURC_fn,T_range = T_range):
        vals = []
        for T in T_range:
            y_pred = torch.nn.functional.softmax(outputs/(T),dim=-1)
            vals.append(metric(y_pred,labels).item())
        return vals
    @classmethod
    def optimize_T(cls,outputs, labels,metric = AURC_fn,T_range = T_range):
        vals = cls.T_grid(outputs,labels,metric,T_range)
        T = T_range[np.argmin(vals)]
        y_pred = (outputs/(T)).softmax(-1)
        return cls(T,y_pred,labels)
   
class pNormAnalysis(TempAnalysis):
    p_range = torch.arange(11)
    def __init__(self,p, T: float, y_pred: torch.tensor, labels: torch.tensor):
        super().__init__(T,y_pred,labels)
        self.p = p
    @staticmethod
    def p_grid(outputs,labels,metric = TempAnalysis.AURC_fn,p_range = p_range):
        vals = []
        for p in p_range:
            y_pred = (norm_p_heuristic(outputs,p)).softmax(-1)
            vals.append(metric(y_pred,labels).item())
        return vals
    @classmethod
    def optimize_p(cls,outputs, labels,metric = TempAnalysis.AURC_fn,p_range = p_range):
        vals = cls.p_grid(outputs,labels,metric,p_range)
        p = p_range[np.argmin(vals)]
        y_pred = norm_p_heuristic(outputs,p).softmax(-1)
        return cls(p,1.0,y_pred,labels)
    @staticmethod
    def p_T_grid(outputs,labels,metric = TempAnalysis.AURC_fn,p_range = p_range,T_range = TempAnalysis.T_range):
        vals = []
        for p in p_range:
            vals_T = TempAnalysis.T_grid(norm_p_heuristic(outputs,p),labels,metric,T_range)
            vals.append(vals_T)
        return vals
    @classmethod
    def optimize_p_T(cls,outputs, labels,metric = TempAnalysis.AURC_fn,p_range = p_range,T_range = TempAnalysis.T_range):
        vals = cls.p_T_grid(outputs,labels,metric,p_range,T_range)
        p,T = np.unravel_index(np.argmin(vals),np.shape(vals))
        p = p_range[p]
        T = T_range[T]
        y_pred = (norm_p_heuristic(outputs,p)/T).softmax(-1)
        return cls(p,T,y_pred,labels)