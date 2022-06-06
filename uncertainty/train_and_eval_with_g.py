from NN_utils import train_and_eval as TE
import torch
import uncertainty.metrics as unc
import uncertainty.comparison as unc_comp
import NN_utils as utils
from tqdm.notebook import tqdm,trange
from IPython.display import display

def accumulate_results_g(model,data):
    '''Accumulate output (of model) and label of a entire dataset.'''

    dev = next(model.parameters()).device

    output_list = torch.Tensor([]).to(dev)
    label_list = torch.Tensor([]).to(dev)
    g_list = torch.Tensor([]).to(dev)

    for image,label in data:
        image,label = image.to(dev),label.to(dev)

        output = model(image)
        g_bool = isinstance(output, tuple)
        
        if g_bool:
            output,g = output
            g = g.view(-1)
            g_list = torch.cat((g_list,g))

        label_list = torch.cat((label_list,label))
        output_list = torch.cat((output_list,output))
        
    if g_bool:    
        output_list = (output_list,g_list)
        
    return output_list,label_list.long()





def train_NN_with_g(model,optimizer,data,loss_criterion,n_epochs=1, print_loss = True,set_train_mode = True):
    '''Train a NN that has a g layer'''
    dev = next(model.parameters()).device
    if set_train_mode:
        model.train()

    for epoch in range(n_epochs):
        running_loss = 0
        for image,label in data:
            image,label = image.to(dev), label.to(dev)
            optimizer.zero_grad()
            output = model(image)
            g = model.get_g()
            loss = loss_criterion(output,g,label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if print_loss:
            print('Epoch ', epoch+1, ', loss = ', running_loss/len(data))


class hist_train_g(TE.hist_train):
    '''Accumulates results while training. Every time update_hist() is called, 
    it evaluates the usefull metrics over the dataset data and stores it in a list.
    Equal to hist_train class, but keeps g (uncertainty estimation) values'''
    
    def __init__(self,model,loss_criterion,data,c = 1.0, risk_dict = None):
        super().__init__(model,loss_criterion,data,c,risk_dict)
        
        self.g_list = []
        if c<1:
            self.acc_c_g = []
            
    def update_hist_c(self):
        '''Update acc_list's and loss_list.
        Redefined so it update also g_list and (possibly) acc_c_g'''
        self.model.eval()
        with torch.no_grad():
            
            #output and label are accumulated for all dataset so that accuracy by coverage can by calculated
            output,label = accumulate_results_g(self.model,self.data)
            y_pred,g = output
            g = g.view(-1)
            
            try:
                loss = self.loss_criterion(output,label).item()
            except: 
                loss = TE.calc_loss_batch(self.model,self.loss_criterion,self.data).item()
            acc = TE.correct_total(y_pred,label)/label.size(0)
            self.acc_list.append(acc)
            self.loss_list.append(loss)
            
            self.g_list.append(torch.mean(g).item())

            if self.c<1:
                #acc_c represents accuracy when the c most uncertain samples are ignored
                mcp = unc.MCP_unc(y_pred) #maximum softmax value
                ent = unc.entropy(y_pred) #entropy of softmax
                self.acc_c_g.append(unc_comp.acc_coverage(y_pred,label,1-g,1-self.c))
                self.acc_c_mcp.append(unc_comp.acc_coverage(y_pred,label,mcp,1-self.c))
                self.acc_c_entropy.append(unc_comp.acc_coverage(y_pred,label,ent,1-self.c))
            if self.risk_dict is not None:
                for name, risk_fn in self.risk_dict.items():
                    risk = risk_fn(output,label).item() 
                    self.risk[name].append(risk)


class Trainer_with_g(TE.Trainer):
    '''Class for easily training/fitting a Pytorch's NN model. Creates 2 'hist' classes,
    keeping usefull metrics and values.
    Identical to Trainer class but with method for training only g's layers.'''
    def __init__(self,model,optimizer,loss_fn,training_data,validation_data = None, c = 0.8, risk_dict = None,update_lr = (0,1)):
        super().__init__(model,optimizer,loss_fn,training_data,validation_data,update_lr = update_lr)
        
        self.hist_train = hist_train_g(model,loss_fn,training_data, c=c,risk_dict = risk_dict)
        if validation_data is not None:
            self.hist_val = hist_train_g(model,loss_fn,validation_data,c=c,risk_dict = risk_dict)
        self.update_hist()

    def fit_g(self,data,n_epochs,ignored_layers = ['main_layer','classifier_layer'], live_plot = True):
        '''Train only the layer specific for g, freezing (disables grad and set eval mode) the rest'''
        progress_epoch = trange(n_epochs,position=0, leave=True, desc = 'Total progress:')
        for e in range(n_epochs):
            progress_epoch.set_description(f'Loss: {self.hist_train.loss_list[-1]:.4f} | Acc_train: {self.hist_train.acc_list[-1]:.2f} | Acc_val: {self.hist_val.acc_list[-1]:.2f} | Progress:')
            self.epoch += 1
            self.model.train()
            #ignore_layers is applied every iteration because 'update_hist method set model to eval mode'
            utils.ignore_layers(self.model,ignored_layers, reset = False) 
            progress = tqdm(data,position=0, leave=False, desc = 'Epoch progress:')
            loss = TE.train_NN(self.model,self.optimizer,progress,self.loss_fn,n_epochs=1, print_loss = False,set_train_mode = False)
            if (self.update_lr_epochs>0) and (self.epoch%self.update_lr_epochs == 0):
                TE.update_optim_lr(self.optimizer,self.update_lr_rate)
            if live_plot:
                utils.live_plot({'Train loss': self.hist_train.loss_list,
                'Validation loss': self.hist_val.loss_list})
                display(progress_epoch.container)
            elif live_plot == 'print':
                print('Epoch ', self.epoch, ', loss = ', loss)
            self.update_hist()
        utils.unfreeze_params(self.model) #unfreeze params to avoid future mistakes
            
            