import torch
from uncertainty import train_NN_with_g
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



class Trainer(nn.Module):
    def __init__(self,model,optimizer,loss_criterion, print_loss = True,keep_hist = True):
        # adaptar keep hist para definir oq manter\n",
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_criterion
        self.print_loss = print_loss
        self.keep_hist = keep_hist

        self.g_list = []
        self.acc_list = []
        self.bce_list = []

    def fit(self,data,n_epochs):
        for epoch in range(1,n_epochs+1):
            train_NN_with_g(self.model,self.optimizer,data,self.loss_fn,1, self.print_loss, set_train_mode = True)
            if self.keep_hist:
                self.update_hist(data)

    def fit_x(self,data,n_epochs,ignored_layers = ['fc_g_layer']):
        loss_criterion = copy.copy(self.loss_fn.criterion)
        loss_criterion.reduction = 'mean'
        ignore_layers(model,ignored_layers,reset = True)
        for epoch in range(1,n_epochs+1):
            train_NN(self.model,self.optimizer,data,loss_criterion,1, self.print_loss, set_train_mode = True)
            if self.keep_hist:
                self.update_hist(data)

    def fit_g(self,data,n_epochs,ignored_layers = ['conv_layer','fc_x_layer']):
        try:
            self.loss_fn.update_L0(self.bce_list[-1])
        except:
            self.loss_fn.update_L0(np.log(10))
            ignore_layers(model,ignored_layers, reset = True)
        for epoch in range(1,n_epochs+1):
            train_NN_with_g(self.model,self.optimizer,data,self.loss_fn,1, self.print_loss, set_train_mode = True)
            if self.keep_hist:
                self.update_hist(data)

            self.loss_fn.update_L0(self.bce_list[-1])

    def update_hist(self,data):
        acc, g, bce =  model_metrics(self.model,data)
        self.g_list.append(g)
        self.acc_list.append(acc)
        self.bce_list.append(bce)
