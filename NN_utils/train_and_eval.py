import torch
import numpy as np
import NN_utils as utils
from collections import defaultdict
import pickle
import pandas as pd
from tqdm.notebook import tqdm,trange
from IPython.display import display
from os.path import join
import wandb

def train_NN(model,optimizer,data,loss_criterion,n_epochs=1, print_loss = True,set_train_mode = True):
    '''Train a Neural Network'''
    dev = next(model.parameters()).device
    if set_train_mode:
        model.train()
    for epoch in range(n_epochs):
        running_loss = 0
        for image,label in data:
            image,label = image.to(dev, non_blocking=True), label.to(dev, non_blocking=True)
            optimizer.zero_grad()
            output = model(image)
            loss = loss_criterion(output,label)
            loss.backward()
            optimizer.step()
            running_loss += loss
        if print_loss:
            print('Epoch ', epoch+1, ', loss = ', running_loss/len(data))
            
    return running_loss/len(data)

def predicted_class(y_pred):
    '''Returns the predicted class for a given softmax output.'''
    with torch.no_grad():
        if y_pred.shape[-1] == 1:
            y_pred = y_pred.view(-1)
            y_pred = (y_pred>0.5).float()
            
        else:
            y_pred = torch.argmax(y_pred, -1)
    return y_pred

def correct_class(y_pred,y_true):
    '''Returns a bool tensor indicating if each prediction is correct'''
    with torch.no_grad():
        y_pred = predicted_class(y_pred)
        correct = (y_pred==y_true)
    
    return correct

def correct_total(y_pred,y_true):
    '''Returns the number of correct predictions in a batch where dk_mask=0'''
    with torch.no_grad():
        correct = correct_class(y_pred,y_true)
        correct_total = torch.sum(correct)
    return correct_total

def calc_loss_batch(model,loss_criterion,data,set_eval = True):
    '''Calculate the average loss over a dataset.'''
    if set_eval:
        model.eval()
    dev = next(model.parameters()).device
    running_loss = 0
    with torch.no_grad():
        for image,label in data:
            image,label = image.to(dev, non_blocking=True), label.to(dev, non_blocking=True)
            output = model(image)
            loss = loss_criterion(output,label)
            running_loss += loss            
    return running_loss/len(data)

def model_acc(model,data,set_eval = True):
    '''Returns the total accuracy of model in some dataset'''
    if set_eval:
        model.eval()
    with torch.no_grad():
        dev = next(model.parameters()).device
        total = 0
        correct= 0
        for image,label in data:
            image,label = image.to(dev, non_blocking=True), label.to(dev, non_blocking=True)
            output = model(image)
            total += label.size(0)
            correct += correct_total(output,label)
    return (correct*100/total)

def model_acc_and_loss(model,loss_criterion,data, set_eval = True):
    '''Calculate the average loss and the accuracy over a dataset.'''
    if set_eval:
        model.eval()
    dev = next(model.parameters()).device
    running_loss = 0
    total = 0
    correct= 0
    with torch.no_grad():
        for image,label in data:
            image,label = image.to(dev, non_blocking=True), label.to(dev, non_blocking=True)
            output = model(image)
            loss = loss_criterion(output,label)
            running_loss += loss
            total += label.size(0)
            correct += correct_total(output,label) 
        loss = running_loss/len(data)
        acc = (correct*100/total)
    return acc,loss


def accumulate_results(model,data, set_eval = False):
    '''Accumulate output (of model) and label of a entire dataset.'''

    dev = next(model.parameters()).device
    if set_eval:
        model.eval()

    output_list = torch.Tensor([]).to(dev)
    label_list = torch.Tensor([]).to(dev)
    with torch.no_grad():
        for image,label in data:
            image,label = image.to(dev), label.to(dev)
            output = model(image)

            label_list = torch.cat((label_list,label))
            output_list = torch.cat((output_list,output))
        
    return output_list,label_list.long()

class hist_train():

    '''Accumulate results while training. Every time update_hist() is called, 
    it evaluates the usefull metrics over the dataset data and stores it in a list.'''
    def __init__(self,model,loss_criterion,data,
                risk_dict = {'accuracy': correct_total}, risk_dict_extra = {}):
        
        self.model = model
        self.loss_criterion = loss_criterion
        self.data = data
        self.risk_dict = risk_dict
        self.risk_dict_extra = risk_dict_extra
        self.risk = defaultdict(list)
        
        self.loss_list = []

    def get_risks(self):
        self.model.eval()
        dev = next(self.model.parameters()).device
        running_loss = 0
        total = 0
        risks = dict.fromkeys(self.risk_dict.keys(), 0.0)
        with torch.no_grad():
            for image,label in self.data:
                image,label = image.to(dev, non_blocking=True), label.to(dev, non_blocking=True)
                output = self.model(image)
                loss = self.loss_criterion(output,label)
                running_loss += loss.item()
                total += label.size(0)
                for name, risk_fn in self.risk_dict.items():
                    risks[name] += risk_fn(output,label).item()
            for name, risk in risks.items():
                self.risk[name].append(risk/total)
            self.loss_list.append(running_loss/len(self.data))

    def update_hist(self):
        self.get_risks()
        with torch.no_grad():
            for name, risk_fn in self.risk_dict_extra.items():
                risk = risk_fn(self.model,self.data).item()
                self.risk[name].append(risk)

    def load_hist(self,hist):
        if isinstance(hist,pd.DataFrame):
            for name,at in hist:
                self.__dict__[name] = at.tolist()
        else:
            for name,at in hist.__dict__.items():
                if not isinstance(at,list): continue
                self.__dict__[name] = at
            self.risk = hist.risk

    def clean(self):
        self.model = None
        self.loss_criterion = None
        self.data = None
        self.risk_dict = None

    def to_dataframe(self):
        d = {}
        for name,at in self.__dict__.items():
            #if isinstance(at,dict): d.update()
            if not isinstance(at,list): continue
            else: d[name] = at
        d.update(self.risk)

        return pd.DataFrame(d)

    def save_class(self, name: str,clean = True): 
        #Não vale a pena já salvar tudo como um dicionário em vez de classe?
        if clean: self.clean()
        name = name + '.pk'
        with open(name, "wb") as output_file:
            pickle.dump(self,output_file)

    def save_df(self, name: str, method:str = 'pickle'):
        assert method == 'pickle' or method == 'csv' or method == 'pickle-df'
        df = self.to_dataframe()
        if method == 'pickle' or method == 'pickle-df':
            name = name + '.pk'
            df.to_pickle(name)
        elif method == 'csv':
            name = name + '.csv'
            df.to_csv(name)
    
                

class Manual_Scheduler(torch.optim.lr_scheduler._LRScheduler):
    #for constant decreasing see MultiStepLR
    def __init__(self,optimizer,schedule:dict, last_epoch = -1,verbose = False):
        self.schedule = schedule
        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self):
        if self.last_epoch in self.schedule.keys():
            return [self.schedule[self.last_epoch]]
        else:
            return [group['lr'] for group in self.optimizer.param_groups]

class Trainer():
    '''Class for easily training/fitting a Pytorch's NN model. Creates 2 'hist' classes,
    keeping usefull metrics and values.'''
    def __init__(self,model,optimizer,loss_criterion,training_data = None,validation_data = None,
                    lr_scheduler = None,risk_dict:dict = {'accuracy': correct_total},risk_dict_extra:dict = {}, name:str = None):

        self.model = model
        self.optimizer = optimizer
        self.loss_criterion = loss_criterion
        self.epoch = 0
        self.lr_scheduler = lr_scheduler
        self.name = name
        if training_data is not None:
            self.hist_train = hist_train(model,loss_criterion,training_data, risk_dict = risk_dict,risk_dict_extra = risk_dict_extra)
        if validation_data is not None:
            self.hist_val = hist_train(model,loss_criterion,validation_data, risk_dict = risk_dict, risk_dict_extra = risk_dict_extra)
        self.update_hist()
            
    def fit(self,data = None,n_epochs = 1, live_plot = False,update_hist = True, 
            save_checkpoint = False, PATH = '.', criterion = 'accuracy',resume = False):
        if hasattr(self,'hist_val'):
            if (criterion == 'loss' or criterion is None):
                criterion_val = self.hist_val.loss_list
            else:
                criterion_val = self.hist_val.risk[criterion]

        if live_plot is True and (not (hasattr(self,'hist_train') or hasattr(self,'hist_val'))):
            live_plot = False
        if data is None:
            data = self.hist_train.data

        if not resume:
            n_epochs += self.epoch
            if live_plot != 'print':
                self.__progress_epoch = trange(n_epochs,position=0, leave=True, desc = 'Progress:')
            else: 
                self.__progress_epoch = range(n_epochs)
            if save_checkpoint:
                self.acc = 0
        if live_plot != 'print':
            progress = tqdm(data,position=1, leave=True, desc = 'Epoch progress:')
        else: progress = data
            
        if live_plot != 'print':
            self.__progress_epoch.disable = False
        while self.epoch < n_epochs:
        #for e in self.__progress_epoch:
            if live_plot != 'print':
                
                desc = 'Progress:'
                if hasattr(self,'hist_train'):
                    if criterion in self.hist_train.risk_dict.keys():
                        desc = f'Loss: {self.hist_train.loss_list[-1]:.4f} | Acc_train: {self.hist_train.risk[criterion][-1]:.2f} |' +desc
                    else:
                        desc = f'Loss: {self.hist_train.loss_list[-1]:.4f}|' +desc
                if hasattr(self,'hist_val'):
                    desc = f'Acc_val (max): {criterion_val[-1]:.2f} ({max(criterion_val):.2f}) | ' + desc

                self.__progress_epoch.set_description(desc)
                progress.disable = False
                progress.reset()

            loss = train_NN(self.model,self.optimizer,progress,self.loss_criterion,1, print_loss = False) #model.train applied internally here
            
            self.update_hist(dataset = update_hist)
            self.epoch += 1
            if live_plot != 'print':
                self.__progress_epoch.update()
            else:
                print('Epoch ', self.epoch, ', loss = ', loss.item())
            

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if live_plot is True:
                desc_dict = {}
                if hasattr(self,'hist_train'):
                    desc_dict['Train loss'] = self.hist_train.loss_list
                if hasattr(self,'hist_val'):
                    desc_dict['Validation loss'] = self.hist_val.loss_list
                    desc_dict['MIN Val loss'] = int(np.argmin(self.hist_val.loss_list))
                utils.live_plot(desc_dict,title = f'Loss {type(self.loss_criterion).__name__}',adjust=True)
                display(self.__progress_epoch.container)
            
                
            if save_checkpoint:
                if criterion_val[-1] >= self.acc:
                    self.acc = criterion_val[-1]
                    self.save_state_dict(PATH,self.name+'_checkpoint')

    def update_hist(self, dataset = 'all'):
        '''Updates hist classes.
        Usefull to use before training to keep pre-training values.'''
        # adicionar modo para criar hist caso o dataset tenha sido adicionado posteriormente
        if (dataset == 'all' or dataset == 'train' or dataset == True) and hasattr(self,'hist_train'):
            self.hist_train.update_hist()

        if (dataset == 'all' or dataset == 'val' or dataset == True) and hasattr(self,'hist_val'):
            self.hist_val.update_hist() 

    def save_hist(self,path, name = None, method = 'pickle-df'):
        assert method == 'pickle-class'  or method == 'pickle-df' or method == 'csv'
        if name is None: name = self.model.name

        #salvar todas hist daqui
        if hasattr(self,'hist_train'):
            name_ = name+'train_hist'
            self.hist_train.save_df(join(path,name_), method)
        if hasattr(self,'hist_val'):
            name_ = name+'val_hist'
            self.hist_train.save_df(join(path,name_), method)

    def save_state_dict(self,path, name = None):
        if name is None:
            if self.name is None:
                name = self.model.name
            else:
                name = self.name
        name = name + '.pt'
        torch.save(self.model.state_dict(), join(path,name))

    def save_all(self, path_model, path_hist, name = None):
        self.save_hist(path_hist, name)
        self.save_state_dict(path_model, name)


class Trainer_WandB(Trainer):
    def __init__(self, model, optimizer, loss_criterion, training_data=None, validation_data=None, 
                lr_scheduler=None, risk_dict: dict = { 'accuracy': correct_total }, risk_dict_extra: dict = {}, 
                **kwargs):
        self.wb = wandb.init(reinit = True,**kwargs)
        self.training_data = training_data
        self.validation_data = validation_data
        self.risk_dict = risk_dict
        self.risk_dict_extra = risk_dict_extra
        super().__init__(model, optimizer, loss_criterion, None, None, lr_scheduler, risk_dict, risk_dict_extra)

    def log(self, data, prefix = ''):
        self.model.eval()
        dev = next(self.model.parameters()).device
        running_loss = 0
        total = 0
        risks = dict.fromkeys(self.risk_dict.keys(), 0.0)
        with torch.no_grad():
            for image,label in data:
                image,label = image.to(dev, non_blocking=True), label.to(dev, non_blocking=True)
                output = self.model(image)
                loss = self.loss_criterion(output,label)
                running_loss += loss.item()
                total += label.size(0)
                for name, risk_fn in self.risk_dict.items():
                    risks[name] += risk_fn(output,label).item()
            for name, risk in risks.items():
                self.wb.log({prefix+name:risk/total})
            self.wb.log({prefix+'Loss':running_loss/len(data)})
            for name, risk_fn in self.risk_dict_extra.items():
                risk = risk_fn(self.model,data).item()
                self.wb.log({prefix+name:risk})
    def fit(self,data = None,n_epochs = 1, live_plot = False,
            save_checkpoint = False, PATH = '.', resume = False, **kwargs):
        if resume:
            self.wb = wandb.init(resume = True, **kwargs)
        with self.wb:
            super().fit(self,data,n_epochs, save_checkpoint, PATH, resume,
            live_plot = False,update_hist = True)
    def save_state_dict(self,path, name = None):
        if name is None:
            if self.wb.name is None:
                name = self.model.name
            else:
                name = self.wb.name
        name = name + '.pt'
        torch.save(self.model.state_dict(), join(path,name))

    def update_hist(self,update_hist = True):
        '''Updates hist classes.
        Usefull to use before training to keep pre-training values.'''
        # adicionar modo para criar hist caso o dataset tenha sido adicionado posteriormente
        if self.training_data is not None and update_hist:
            self.log(self.training_data, 'Training ')

        if self.validation_data is not None and update_hist:
            self.log(self.validation_data, 'Validation ')
        
if __name__ == '__main__':
    model =torch.nn.Sequential(torch.nn.Linear(10,10))
    opt = torch.optim.SGD(model.parameters(), lr = 0.1)
    schedule = {10:0.01,20:0.001,30:0.005}
    scheduler = Manual_Scheduler(opt,schedule)
    for e in range(35):
        print(f"epoch = {e}, lr = {opt.param_groups[0]['lr']}")
        scheduler.step(e)
