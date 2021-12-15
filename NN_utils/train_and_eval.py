import torch
def train_NN(model,optimizer,data,loss_criterion,n_epochs=1, print_loss = True):
    '''Train a Neural Network'''
    dev = next(model.parameters()).device
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
