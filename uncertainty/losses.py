import numpy as np
import torch
from math import log
import NN_utils as utils
from uncertainty import entropy,get_TCP
from NN_utils.train_and_eval import correct_class
from scipy.optimize import root_scalar
import torch.nn.functional as F
import os
from scipy.spatial.distance import cdist


class MaximumClassSeparation(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def V(order):
        if order == 1:
            return np.array([[1, -1]])
        else:
            col1 = np.zeros((order, 1))
            col1[0] = 1
            row1 = -1 / order * np.ones((1, order))
            return np.concatenate((col1, np.concatenate((row1, np.sqrt(1 - 1 / (order**2)) * MaximumClassSeparation.V(order - 1)), axis=0)), axis=1)
    @staticmethod
    def create_prototypes(nr_prototypes):
        assert nr_prototypes > 0
        prototypes = MaximumClassSeparation.V(nr_prototypes - 1).T
        assert prototypes.shape == (nr_prototypes, nr_prototypes - 1)
        assert np.all(np.abs(np.sum(np.power(prototypes, 2), axis=1) - 1) <= 1e-6)
        distances = cdist(prototypes, prototypes)
        assert distances[~np.eye(*distances.shape, dtype=bool)].std() <= 1e-3
        return prototypes.astype(np.float32)



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

class SelectiveNetLoss(torch.nn.Module):
    def __init__(self,c:float, criterion = torch.nn.CrossEntropyLoss(reduction='none'), lamb = 32,
                 alpha:float = 1.0, reduction = 'mean'):
        super().__init__()

        self.criterion = criterion #criterion must have reduction set to 'none'
        try:
            self.criterion.reduction = 'none'
        except: pass
        self.lamb = lamb
        self.c = c #coverage
        self.alpha = alpha
        self.reduction = reduction

    def lagrangian(self,const):
        #variant of the well-known Interior Point Method (IPM)
        return self.lamb*torch.square(torch.nn.functional.relu(const))
    
    def selective_risk(self,y_pred,g,y_true):
        
        loss = self.criterion(y_pred,y_true)*g
        loss /= g.mean()
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

    def constraint(self,g): 
        return self.c - g.mean() #must be <=0

    def forward(self,output,y_true):
        
        y_pred,g,y_h = output
            
        g = g.view(-1)
        
        loss = self.selective_risk(y_pred,g,y_true)
        loss += self.lagrangian(self.constraint(g))

        loss_h = self.selective_risk(y_h,g,y_true)
        loss = self.alpha*loss + (1-self.alpha)*loss_h
        return loss

class TCP_Loss(torch.nn.Module):
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
            output = torch.sigmoid(output)
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