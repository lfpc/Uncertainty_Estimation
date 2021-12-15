import torch
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
    def __init__(self,criterion, def_loss):
        super().__init__()
        self.L0 = def_loss
        self.criterion = criterion #criterion must have reduction set to 'none'
        
    def forward(self, y_pred,g,y_true):
        
        loss = g*self.criterion(y_pred,y_true)+(1-g)*self.L0
        loss = torch.mean(loss)

        return loss

    def update_L0(self,new_L0):
        with torch.no_grad():
            self.L0 = new_L0