import torch
from uncertainty import train_NN_with_g
import numpy as np
import copy
from NN_utils import ignore_layers

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
    with torch.no_grad():
        model.eval()
        dev = next(model.parameters()).device

        g_list = torch.Tensor([])
        output_list = torch.Tensor([])
        label_list = torch.Tensor([])

        for image,label in data:
            image,label = image.to(dev), label.to(dev)

            output = model(image).cpu()
            g = model.get_g().view(-1).cpu()

            label_list = torch.cat((label_list,label.cpu()))
            output_list = torch.cat((output_list,output))
            g_list = torch.cat((g_list,g))
        
    return label_list.long(),output_list,g_list

def model_metrics(model,loss_criterion,data):
    model.eval()
    loss_criterion.reduction = 'sum'
    dev = next(model.parameters()).device
    total = 0
    correct= 0
    g_i = 0
    bce = 0
    with torch.no_grad():
        for image,label in data:
            image,label = image.to(dev), label.to(dev)
            output = model(image)
            g = model.get_g()
            total += label.size(0)
            correct += correct_total(output,label)
            g_i += torch.sum(g).item()
            bce += loss_criterion(output,label).item()
    return (correct/total),g_i/total, bce/total

class Trainer(torch.nn.Module):
    def __init__(self,model,optimizer,loss_criterion, print_loss = True):
        # adaptar keep hist para definir oq manter\n",
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_criterion
        self.print_loss = print_loss

        self.g_list = []
        self.acc_list = []
        self.bce_list = []
        self.bce_iter = []
        self.acc_c_g = []
        self.acc_c_mcp = []
        
    def fit(self,data,n_epochs):
        for epoch in range(1,n_epochs+1):
            self.train_NN_with_g(data,1,set_train_mode = True)
            self.update_hist(data)
            self.loss_fn.update_L0(self.bce_list[-1])
            #self.bce_iter = []

    def fit_x(self,data,n_epochs,ignored_layers = ['fc_g_layer']):
        loss_criterion = copy.copy(self.loss_fn.criterion)
        loss_criterion.reduction = 'mean'
        ignore_layers(self.model,ignored_layers,reset = True)
        for epoch in range(1,n_epochs+1):
            train_NN(self.model,self.optimizer,data,loss_criterion,1, self.print_loss, set_train_mode = True)
            self.update_hist(data)
            ignore_layers(self.model,ignored_layers,reset = True)

    def fit_g(self,data,n_epochs,ignored_layers = ['conv_layer','fc_layer','fc_x_layer']):
        
        _,_,bce = model_metrics(self.model,nn.NLLLoss(),data)
        self.loss_fn.update_L0(bce)#(self.bce_list[-1])
        ignore_layers(self.model,ignored_layers, reset = True)
        for epoch in range(1,n_epochs+1):
            self.train_NN_with_g(data,1, set_train_mode = False)
            self.update_hist(data)
            ignore_layers(self.model,ignored_layers, reset = True)
        
            
    def train_NN_with_g(self,data,n_epochs, set_train_mode = True):
            '''Train a NN that has a g layer'''
            dev = next(self.model.parameters()).device
            if set_train_mode:
                self.model.train()
            for epoch in range(n_epochs):
                running_loss = 0
                for image,label in data:
                    image,label = image.to(dev), label.to(dev)
                    self.optimizer.zero_grad()
                    output = self.model(image)
                    g = self.model.get_g()
                    loss = self.loss_fn(output,g,label)
                    self.bce_iter.append(torch.mean(self.loss_fn.criterion(output,label)).item())
                    self.loss_fn.update_L0(np.mean(self.bce_iter[-2080:]))
                    #_, _, bce =  model_metrics(self.model,self.loss_fn.criterion,data)
                    #self.loss_fn.update_L0(bce)
                    
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                if self.print_loss:
                    print('Epoch ', epoch+1, ', loss = ', running_loss/len(data))
                    
    def update_hist(self,data):
        with torch.no_grad():
            loss_criterion = copy.copy(self.loss_fn.criterion)
            loss_criterion.reduction = 'mean'
            #acc, g, bce =  model_metrics(self.model,loss_criterion,data)
            #self.g_list.append(g)
            #self.acc_list.append(acc)
            #self.bce_list.append(bce)

            label, output, g = accumulate_results(self.model,data)
            bce = loss_criterion(output,label).item()
            output = torch.exp(output)
            mcp = unc.MCP_unc(output)

            self.acc_c_g.append(unc_comp.acc_coverage(output,label,1-g,0.2))
            self.acc_c_mcp.append(unc_comp.acc_coverage(output,label,mcp,0.2))

            self.g_list.append(torch.mean(g).item())
            acc = correct_total(output,label)/label.size(0)
            self.acc_list.append(acc)
            self.bce_list.append(bce)