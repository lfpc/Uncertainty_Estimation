import torch
import numpy as np
import NN_utils as utils
from collections import defaultdict
import pickle
import pandas as pd
from tqdm.notebook import tqdm,trange
from IPython.display import display

def train_NN(model,optimizer,data,loss_criterion,n_epochs=1, print_loss = True,set_train_mode = True):
    '''Train a Neural Network'''
    dev = next(model.parameters()).device
    if set_train_mode:
        model.train()
    for epoch in range(n_epochs):
        running_loss = 0
        for image,label in data:
            image,label = image.to(dev), label.to(dev)
            optimizer.zero_grad()
            output = model(image)
            loss = loss_criterion(output,label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if print_loss:
            print('Epoch ', epoch+1, ', loss = ', running_loss/len(data))
            
    return running_loss/len(data)

def predicted_class(y_pred):
    '''Returns the predicted class for a given softmax output.'''
    if y_pred.shape[-1] == 1:
        y_pred = y_pred.view(-1)
        y_pred = (y_pred>0.5).float()
        
    else:
        y_pred = torch.max(y_pred, 1)[1]
    return y_pred

def correct_class(y_pred,y_true):
    '''Returns a bool tensor indicating if each prediction is correct'''

    y_pred = predicted_class(y_pred)
    correct = (y_pred==y_true)
    
    return correct

def correct_total(y_pred,y_true):
    '''Returns the number of correct predictions in a batch where dk_mask=0'''
    correct = correct_class(y_pred,y_true)
    correct_total = torch.sum(correct).item()
    return correct_total

def calc_loss_batch(model,loss_criterion,data):
    '''Calculate the average loss over a dataset.'''
    dev = next(model.parameters()).device
    running_loss = 0
    for image,label in data:
        image,label = image.to(dev), label.to(dev)
        output = model(image)
        loss = loss_criterion(output,label)
        running_loss += loss            
    return running_loss/len(data)

def model_acc(model,data):
    '''Returns the total accuracy of model in some dataset'''
    with torch.no_grad():
        dev = next(model.parameters()).device
        total = 0
        correct= 0
        for image,label in data:
            image,label = image.to(dev), label.to(dev)
            output = model(image)
            total += label.size(0)
            correct += correct_total(output,label)
    return (correct*100/total)

def model_acc_and_loss(model,loss_criterion,data):
    '''Calculate the average loss and the accuracy over a dataset.'''
    dev = next(model.parameters()).device
    running_loss = 0
    total = 0
    correct= 0
    for image,label in data:
        image,label = image.to(dev), label.to(dev)
        output = model(image)
        loss = loss_criterion(output,label)
        running_loss += loss.item()
        total += label.size(0)
        correct += correct_total(output,label) 
    loss = running_loss/len(data)
    acc = (correct*100/total)
    return acc,loss


def accumulate_results(model,data):
    '''Accumulate output (of model) and label of a entire dataset.'''

    dev = next(model.parameters()).device

    output_list = torch.Tensor([]).to(dev)
    label_list = torch.Tensor([]).to(dev)

    for image,label in data:
        with torch.no_grad():
            image,label = image.to(dev), label.to(dev)
            output = model(image)

            label_list = torch.cat((label_list,label))
            output_list = torch.cat((output_list,output))
        
    return output_list,label_list.long()

class hist_train():

    '''Accumulate results while training. Every time update_hist() is called, 
    it evaluates the usefull metrics over the dataset data and stores it in a list.'''
    def __init__(self,model,loss_criterion,data,risk_dict = None):
        
        self.model = model
        self.loss_criterion = loss_criterion
        self.data = data
        self.risk_dict = risk_dict
        self.risk = defaultdict(list)
        
        self.acc_list = []
        self.loss_list = []


    def update_hist(self):
        '''Update acc_list's and loss_list.
        If coverage is defined (different than 1), updates acc_c lists'''
        self.model.eval()
        with torch.no_grad():
            acc, loss = model_acc_and_loss(self.model,self.loss_criterion,self.data)
            self.acc_list.append(acc)
            self.loss_list.append(loss)
            if self.risk_dict is not None:
                for name, risk_fn in self.risk_dict.items():
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
    
                

def update_optim_lr(optimizer,rate):
    optimizer.param_groups[0]['lr'] /= rate

class Trainer():
    '''Class for easily training/fitting a Pytorch's NN model. Creates 2 'hist' classes,
    keeping usefull metrics and values.'''
    def __init__(self,model,optimizer,loss_criterion,training_data = None,validation_data = None,
                    update_lr = (0,1),risk_dict = None):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_criterion
        self.epoch = 0
        if isinstance(update_lr,tuple):
            self.update_lr_epochs = update_lr[0]
            self.update_lr_rate = update_lr[1]
        elif isinstance(update_lr,dict): #construir
            self.update_lr = update_lr

        if training_data is not None:
            self.hist_train = hist_train(model,loss_criterion,training_data, risk_dict = risk_dict)
        if validation_data is not None:
            self.hist_val = hist_train(model,loss_criterion,validation_data, risk_dict = risk_dict)
        self.update_hist()
            
    def fit(self,data = None,n_epochs = 1, live_plot = True,update_hist = True, save_checkpoint = False):
        if data is None:
            data = self.training_data
        if not live_plot == 'print':
            progress_epoch = trange(n_epochs,position=0, leave=True, desc = 'Progress:')
            progress = tqdm(data,position=1, leave=True, desc = 'Epoch progress:')
        else: 
            progress_epoch = range(n_epochs)
            progress = data
        if save_checkpoint:
            acc = 0
        for e in progress_epoch:
            if not live_plot == 'print':
                desc = 'Progress:'
                if hasattr(self,'hist_train'):
                    desc = f'Loss: {self.hist_train.loss_list[-1]:.4f} | Acc_train: {self.hist_train.acc_list[-1]:.2f} |' +desc
                if hasattr(self,'hist_val'):
                    desc = f'Acc_val: {self.hist_val.acc_list[-1]:.2f} | ' + desc

                progress_epoch.set_description(desc)
                progress.disable = False
                progress.reset()

            loss = train_NN(self.model,self.optimizer,progress,self.loss_fn,1, print_loss = False) #model.train applied internally here
            self.update_hist(dataset = update_hist)
            self.epoch += 1
            if (self.update_lr_epochs>0) and (self.epoch%self.update_lr_epochs == 0):
                update_optim_lr(self.optimizer,self.update_lr_rate)
            if live_plot is True:
                desc_dict = {}
                if hasattr(self,'hist_train'):
                    desc_dict['Train loss'] = self.hist_train.loss_list
                if hasattr(self,'hist_val'):
                    desc_dict['Validation loss'] = self.hist_val.loss_list
                utils.live_plot(desc_dict)
                display(progress_epoch.container)
            elif live_plot == 'print':
                print('Epoch ', self.epoch, ', loss = ', loss)
            if save_checkpoint:
                if self.hist_val.acc_list[-1] >= acc:
                    acc = self.hist_val.acc_list[-1]
                    self.model.save_state_dict('.',self.model.name+'checkpoint')

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
        pass
        #salvar todas hist daqui
        if hasattr(self,'hist_train'):
            self.hist_train.save_df(path+r'/'+name+'train_hist', method)
        if hasattr(self,'hist_val'):
            self.hist_val.save_df(path+r'/'+name+'val_hist', method)

    def save_all(self, path_model, path_hist, name = None):
        self.save_hist(path_hist, name)
        self.model.save_state_dict(path_model, name)
        
