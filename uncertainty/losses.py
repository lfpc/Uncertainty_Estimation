import torch
from math import log
import NN_utils as utils
from uncertainty.quantifications import entropy
from NN_utils.train_and_eval import correct_class
import uncertainty.comparison as unc_comp

class aux_loss_fs(torch.nn.Module):
    '''Cross Entropy between g (uncertainty variable) and a 'right' vector,
    with 1 to the samples that the classifier gets correct and 0 if it gets wrong.'''
    def __init__(self,loss_criterion = torch.nn.BCELoss()):
        super().__init__()
        self.criterion = loss_criterion
    def forward(self, output,y_true):
        y_pred,g = output
        g = g.view(-1)
        right = correct_class(y_pred,y_true).float()
        loss = self.criterion(g,right)
        loss = torch.mean(loss)
        return loss



class LCE(torch.nn.Module):
    '''Defines LCE loss - Devries(2018)'''
    def __init__(self,criterion,lamb_init,beta,adjust_factor = 1.01):
        super().__init__()
        self.criterion = criterion
        self.lamb = lamb_init
        self.beta = beta
        self.adjust_factor = adjust_factor
 
    def forward(self, y_pred,g,y_true):
        
        with torch.no_grad():
            y_true_onehot = torch.nn.functional.one_hot(y_true,y_pred.shape[-1])
        y = g*y_pred+(1-g)*y_true_onehot
        loss_t = self.criterion(y,y_true)
        loss_g = torch.mean(-torch.log(g))
        loss = loss_t + self.lamb*loss_g
        #loss = torch.sum(loss)
        self.update_lamb(loss_g.item())
        
        return loss
    
    def update_lamb(self,loss_g):
        with torch.no_grad():
            if loss_g > self.beta:
                self.lamb = self.lamb*self.adjust_factor
            elif loss_g < self.beta:
                self.lamb = self.lamb/self.adjust_factor

class penalized_uncertainty(torch.nn.Module):
    def __init__(self,criterion, def_loss = log(10)):
        super().__init__()
        self.L0 = def_loss
        self.criterion = criterion #criterion must have reduction set to 'none'
        
    def forward(self,output,g,y_true):
        
        y_pred, g = output
        g = g.view(-1)
        loss = g*self.criterion(y_pred,y_true)+(1-g)*self.L0
        loss = torch.mean(loss)
        

        return loss

    def update_L0(self,new_L0):
        with torch.no_grad():
            self.L0 = new_L0

def entropy_const(w):
    H = torch.exp(entropy(w,reduction = 'sum'))/w.size(0)
    return H
normalize_tensor = (lambda x,dim=-1: torch.nn.functional.normalize(x, p=1,dim=dim))

def IPM_selectivenet(r,const,lamb = 32):
    #optimize x such that const >0
    gama = lamb*torch.square(torch.maximum(torch.tensor([0]).cuda(),const))
    objective = r + gama
    return objective

class selective_net_2(torch.nn.Module):
    def __init__(self,criterion,w_fn = normalize_tensor,c_fn = torch.mean,optim_method = IPM_selectivenet, c = 0.8,
                 alpha = 1.0, head = None,const_var = 'g'):
        super().__init__()

        self.criterion = criterion #criterion must have reduction set to 'none'
        self.w_fn = w_fn #transform applied to g
        self.c_fn = c_fn #transform applied to w that goes onto constraint
        self.optim_method = optim_method #transform applied to risk (loss) and constraint and returns a equivalent unconstrained objective
        self.c = c #coverage
        self.alpha = alpha
        self.head = head
        self.const_var = const_var
    
    def get_loss(self,y_pred,w,y_true):
        
        loss = w*self.criterion(y_pred,y_true)
        loss = torch.sum(loss) #sum? mean? When w is a normalization, is must be sum. How to generalize?

        return loss

    def get_constraint(self,w): 
        H = self.c_fn(w) #must be >= c
        constraint = self.c - H #must be <=0
        return constraint

    def forward(self,output,y_true):
        
        y_pred,g = output
        g = g.view(-1)
        w = self.w_fn(g)
        
        loss = self.get_loss(y_pred,w,y_true)
        if self.optim_method is not None:
            if self.const_var == 'w':
                const = self.get_constraint(w)
            elif self.const_var == 'g':
                const = self.get_constraint(g)
            loss = self.optim_method(loss, const)

        if self.head is None:
            loss_h = 0
        else:
            w = self.w_fn(torch.ones([torch.numel(g)]),dim=-1).to(y_pred.device)
            if self.head == 'y':
                loss_h = self.get_loss(y_pred,w,y_true)
            else: 
                h = self.head()
                loss_h = self.get_loss(h,w,y_true)
        loss = self.alpha*loss + (1-self.alpha)*loss_h

        return loss