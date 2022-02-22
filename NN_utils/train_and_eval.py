import torch
import numpy as np
import copy
import NN_utils as utils
import uncertainty.quantifications as unc
import uncertainty.comparison as unc_comp

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

def calc_loss_batch(model,loss_criterion,data):
    '''Calculate the average loss over a dataset.'''
    dev = next(model.parameters()).device
    running_loss = 0
    for image,label in data:
        image,label = image.to(dev), label.to(dev)
        output = model(image)
        loss = loss_criterion(output,label)
        running_loss += loss.item()            
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

def model_acc(model,data):
    '''Returns the total accuracy of model in some dataset'''
    model.eval()
    dev = next(model.parameters()).device
    total = 0
    correct= 0
    for image,label in data:
        image,label = image.to(dev), label.to(dev)
        output = model(image)
        total += label.size(0)
        correct += correct_total(output,label)
    return (correct*100/total)

def accumulate_results(model,data):
    '''Accumulate output (of model) and label of a entire dataset.'''

    dev = next(model.parameters()).device

    output_list = torch.Tensor([]).to(dev)
    label_list = torch.Tensor([]).to(dev)

    for image,label in data:

        image,label = image.to(dev), label.to(dev)
        output = model(image)

        label_list = torch.cat((label_list,label))
        output_list = torch.cat((output_list,output))
        
    return output_list,label_list.long()

class hist_train():

    '''Accumulate results while training. Every time update_hist() is called, 
    it evaluates the usefull metrics over the dataset data and stores it in a list.'''
    def __init__(self,model,loss_criterion,data, c = 1.0):
        
        self.model = model
        self.loss_criterion = loss_criterion
        self.data = data
        self.c = c #coverage
        
        self.acc_list = []
        self.loss_list = []
        if c<1:
            #acc_c represents accuracy when the c most uncertain samples are ignored
            self.acc_c_mcp = [] 
            self.acc_c_entropy = []

    
    def update_hist(self):
        '''Update acc_list's and loss_list.
        If coverage is defined (different than 1), updates acc_c lists'''
        
        dev = next(self.model.parameters()).device
        self.model.eval()
            
        with torch.no_grad():
            #y_pred and label are accumulated for all dataset so that accuracy by coverage can by calculated
            y_pred,label = accumulate_results(self.model,self.data)
            
            loss = self.loss_criterion(y_pred,label).item()
            acc = correct_total(y_pred,label)/label.size(0) #accuracy
            self.acc_list.append(acc)
            self.loss_list.append(loss)
            
            if self.c<1:
                #acc_c represents accuracy when the c most uncertain samples are ignored
                mcp = unc.MCP_unc(y_pred) #maximum softmax value
                ent = unc.entropy(y_pred) #entropy of softmax
                self.acc_c_mcp.append(unc_comp.acc_coverage(y_pred,label,mcp,1-self.c))
                self.acc_c_entropy.append(unc_comp.acc_coverage(y_pred,label,ent,1-self.c))

            
class Trainer():
    '''Class for easily training/fitting a Pytorch's NN model. Creates 2 'hist' classes,
    keeping usefull metrics and values.'''
    def __init__(self,model,optimizer,loss_criterion,training_data,validation_data = None, c=1.0):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_criterion
        self.epoch = 0
        
        self.hist_train = hist_train(model,loss_criterion,training_data, c=c)
        if validation_data is not None:
            self.hist_val = hist_train(model,loss_criterion,validation_data,c=c)
            

    def fit(self,data,n_epochs):
        for e in range(1,n_epochs+1):
            self.epoch += 1
            loss = train_NN(self.model,self.optimizer,data,self.loss_fn,1, print_loss = False) #model.train applied internally here
            print('Epoch ', self.epoch, ', loss = ', loss)

            self.hist_train.update_hist()
            try: self.hist_val.update_hist() #with try/except in case there is no validation hist class
            except: pass
            
    def update_hist(self):
        '''Updates hist classes.
        Usefull to use before training to keep pre-training values.'''
        self.hist_train.update_hist()
        try: self.hist_val.update_hist() #with try/except in case there is no validation hist class
        except: pass