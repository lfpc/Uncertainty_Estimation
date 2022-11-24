import numpy as np
import torch
from math import log
import NN_utils as utils
from uncertainty import entropy,get_TCP
from NN_utils.train_and_eval import correct_class
from scipy.optimize import root_scalar
import torch.nn.functional as F
import os




class HypersphericalLoss(torch.nn.Module):
    def __init__(self, polars, reduction = 'mean') -> None:
        super().__init__()
        self.polars = polars
        self.loss_fn = torch.nn.CosineSimilarity(eps=1e-9)
        self.reduction = reduction
    def forward(self,y_pred,y_true):
        y_true = self.polars[y_true]
        loss = (1 - self.loss_fn(y_pred,y_true)).pow(2)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
    def to(self,dev):
        self.polars = self.polars.to(dev)
        return super().to(dev)
    @classmethod
    def from_file(cls,polars_file,**kwargs):
        classpolars = torch.from_numpy(np.load(polars_file)).float()
        return cls(classpolars,**kwargs)
    @classmethod
    def from_values(cls,path,classes,dims,**kwargs):
        polars_file = os.path.join(path,f'prototypes-{dims}d-{classes}c.npy')
        if os.path.isdir(path):
            classpolars = torch.from_numpy(np.load(polars_file)).float()
        else: 
            classpolars = HypersphericalLoss.get_prototypes(classes,dims,save_dir = path)
        return cls(classpolars,**kwargs)
    @staticmethod
    def __prototype_loss(prototypes):
        # Dot product of normalized prototypes is cosine similarity.
        product = torch.matmul(prototypes, prototypes.t()) + 1
        # Remove diagnonal from loss.
        product -= 2. * torch.diag(torch.diag(product))
        # Minimize maximum cosine similarity.
        loss = product.max(dim=1)[0]
        return loss.mean(), product.max()
    @staticmethod
    def __prototype_loss_sem(prototypes, triplets):
        product = torch.matmul(prototypes, prototypes.t()) + 1
        product -= 2. * torch.diag(torch.diag(product))
        loss1 = -product[triplets[:,0], triplets[:,1]]
        loss2 = product[triplets[:,2], triplets[:,3]]
        return loss1.mean() + loss2.mean(), product.max()
    @staticmethod
    def get_prototypes(classes, dims,
                        save_dir = None,
                        learning_rate = 0.1,
                        momentum = 0.9,
                        epochs:int = 10000,
                        wtvfile = "",
                        nn:int = 2):

        # Initialize prototypes and optimizer.
        if os.path.exists(wtvfile):
            use_wtv = True
            wtvv = np.load(wtvfile)
            for i in range(wtvv.shape[0]):
                wtvv[i] /= np.linalg.norm(wtvv[i])
            wtvv = torch.from_numpy(wtvv)
            wtvsim = torch.matmul(wtvv, wtvv.t()).float()
            
            # Precompute triplets.
            nns, others = [], []
            for i in range(wtvv.shape[0]):
                sorder = np.argsort(wtvsim[i,:])[::-1]
                nns.append(sorder[:nn])
                others.append(sorder[nn:-1])
            triplets = []
            for i in range(wtvv.shape[0]):
                for j in range(len(nns[i])):
                    for k in range(len(others[i])):
                        triplets.append([i,j,i,k])
            triplets = np.array(triplets).astype(int)
        else:
            use_wtv = False


        prototypes = torch.randn(classes, dims)
        prototypes = torch.nn.Parameter(F.normalize(prototypes, p=2, dim=1))
        optimizer = torch.optim.SGD([prototypes], lr=learning_rate, momentum=momentum)

        # Optimize for separation.
        for i in range(epochs):
            # Compute loss.
            loss, sep = HypersphericalLoss.__prototype_loss(prototypes)
            if use_wtv:
                loss2 = HypersphericalLoss.__prototype_loss_sem(prototypes, triplets)
                loss += loss2

            # Update.
            loss.backward()
            optimizer.step()
            # Renormalize prototypes.
            prototypes = torch.nn.Parameter(F.normalize(prototypes, p=2, dim=1))
            optimizer = torch.optim.SGD([prototypes], lr=learning_rate, \
                    momentum=momentum)
            print(f'Epoch {i+1}/{epochs}')
        
        # Store result.
        if save_dir is not None:
            np.save(os.path.join(save_dir,f'prototypes-{dims}d-{classes}c.npy'),prototypes.data.numpy())
        return prototypes
    
        



class FocalLoss(torch.nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma,  reduction='mean'):
        super().__init__(reduction='none')
        self._reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        input_prob = torch.gather(F.softmax(input_, dim = -1), 1, target.unsqueeze(1)).view(-1)
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss) if self._reduction == 'mean' else torch.sum(loss) if self._reduction == 'sum' else loss



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
            y_true_onehot = F.one_hot(y_true,y_pred.shape[-1])
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
normalize_tensor = (lambda x,dim=-1: F.normalize(x, p=1,dim=dim))

def IPM_selectivenet(r,const,lamb = 32):
    #optimize x such that const >0 with quadratic penalty
    gama = lamb*torch.square(torch.maximum(torch.tensor([0]).cuda(),const))
    objective = r + gama
    return objective

def IPM_log(r,const,t = 32):
    #optimize x such that const >0 with log barrier
    gama = -(1/t)*torch.log(-const)
    return r+gama
def IPM_log_adap(r,const, t = 32):
    if const <= -(1/(t**2)):
        gama = IPM_log(r,const,t)
    else:
        gama = t*const-(1/t)*log(1/(t**2))+(1/t)
    return r+gama



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

def w_fn(g,lamb = 1):
    lamb = torch.as_tensor(lamb)
    w = torch.nn.softmax(lamb*g,dim=-1)
    return w
def H_fn(g,lamb = 1):
    w = w_fn(g,lamb)
    H = entropy(w)
    return H
def H_const(lamb,*args):
    g,c = args
    H = H_fn(g,lamb)
    const = (torch.exp(H)/torch.numel(g))-c
    return const

class selective_net_lambda(torch.nn.Module):
    def __init__(self,criterion,w_fn = w_fn, c_fn = H_const,c = 0.8,alpha = 1.0, head = None):
        super().__init__()

        self.criterion = criterion #criterion must have reduction set to 'none'
        self.w_fn = w_fn #transform applied to g
        self.c_fn = c_fn #transform applied to w that goes onto constraint
        self.c = c #coverage
        self.alpha = alpha
        self.head = head
        self.lamb = 1

    
    def get_loss(self,y_pred,w,y_true):
        
        loss = w*self.criterion(y_pred,y_true)
        loss = torch.sum(loss) #sum? mean? When w is a normalization, is must be sum. How to generalize?
        return loss

    def update_lambda(self,g):
        self.lamb = root_scalar(self.c_fn,bracket=[0, 100],args = (g,self.c)).root
        return self.lamb

    def forward(self,output,y_true):
        
        y_pred,g = output
        g = g.view(-1)
        self.update_lambda(g)
        w = self.w_fn(g,self.lamb)
        
        loss = self.get_loss(y_pred,w,y_true)

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

class confid_loss(torch.nn.Module):
    def __init__(self, criterion = torch.nn.MSELoss()):
        super().__init__()
        self.criterion = criterion
        
    def forward(self,output,y_true):
        y_pred,g = output
        g = g.view(-1)
        tcp = get_TCP(y_pred,y_true)
        loss = self.criterion(g,tcp)
        return loss


class OVALoss(torch.nn.Module):
    def __init__(self,n_classes:int,from_logits:bool = True, reduction = 'mean', eps:float = 1e-20,bound = 100):
        super().__init__()
        self.n_classes = n_classes
        self.reduction = reduction
        self.eps = torch.tensor(eps)
        self.bound = bound
        self.from_logits = from_logits
    def forward(self, output,y_true):
        if self.from_logits:
            output = F.sigmoid(output)
        y = F.one_hot(y_true,self.n_classes)        
        loss = self.CrossEntropy(output,y)
        loss += self.CrossEntropy(1-output,1-y)
        # arrumar [(1-y).bool()]
        loss = torch.sum(loss,dim=-1)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
    def CrossEntropy(self,output,y):
        return torch.clamp(-y*torch.log(output+self.eps),min=0.0,max=self.bound)