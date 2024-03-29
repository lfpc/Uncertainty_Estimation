import torch
import torch.nn as nn
import torch.nn.functional as F
import NN_utils.train_and_eval as TE

from tqdm.notebook import tqdm,trange

class MIMOModel(nn.Module):
    def __init__(self,model,num_classes, n_ensembles: int = 3, name = 'MIMO',softmax = 'log', *args):
        super(MIMOModel, self).__init__()
        self.model = model(num_classes = num_classes * n_ensembles, softmax = False, *args)
        self.name = name
        self.softmax = softmax
        self.n_ensembles = n_ensembles
        self.classifier_layer = self.model.classifier_layer
        self.model.classifier_layer = nn.Identity()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_shape_list = list(input_tensor.size())  # (n_ensembles,batch_size,C,H,W)
        n_ensembles, batch_size = input_shape_list[0], input_shape_list[1]
        
        assert n_ensembles == self.n_ensembles

        input_tensor = input_tensor.view([n_ensembles * batch_size] + input_shape_list[2:])
        output = self.model(input_tensor)
        output = output.view(n_ensembles, batch_size, -1)
        output = self.classifier_layer(output)
        output = output.view(n_ensembles, batch_size, n_ensembles, -1)
        output = torch.diagonal(output, offset=0, dim1=0, dim2=2).permute(2, 0, 1)
        if self.softmax == 'log':
            output = F.log_softmax(output,dim=-1)
        elif self.softmax:
            output = F.softmax(output,dim = -1)
        return output
        
    def save_state_dict(self,path, name = None):
        if name is None: name = self.name
        torch.save(self.state_dict(), path + r'/' + name + '.pt')


class Trainer_MIMO(TE.Trainer):
    def __init__(self, model, optimizer, loss_criterion, training_data=None, validation_data=None, lr_scheduler = None, risk_dict=None):
        super().__init__(model, optimizer, loss_criterion, None, None, lr_scheduler=lr_scheduler, risk_dict = risk_dict)
        
        self.test_dataloader = validation_data 
        self.val_acc = []
        self.loss = []
        self.validate(plot = False)

    def fit(self,train_dataloader,n_epochs:int = 1, checkpoint:bool = True, PATH:str = '.'):
        train_dataloaders = [
        train_dataloader for _ in range(self.model.n_ensembles)]

        self.model.train()
        dev = next(self.model.parameters()).device
        progress_epoch = trange(n_epochs,position=0, leave=True, desc = 'Progress:')
        maxacc = 0 
        for epoch in progress_epoch:
            desc = 'Progress:'
            desc = f'Loss: {self.loss[-1]:.4f} |' +desc

            desc = f'Acc_val: {self.val_acc[-1]:.2f} | ' + desc

            progress_epoch.set_description(desc)
            self.epoch += 1
            
            for datum in zip(*train_dataloaders):
                model_inputs = torch.stack([data[0] for data in datum]).to(dev)
                targets = torch.stack([data[1] for data in datum]).to(dev)

                self.optimizer.zero_grad()
                outputs = self.model(model_inputs)
                loss = self.loss_fn(outputs.transpose(2, 1), targets)
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(self.epoch)
            self.validate(False)
            #save mimo model is possible? or save base model?
            if checkpoint:
                if self.val_acc[-1] >= maxacc:
                    maxacc = self.val_acc[-1]
                    self.model.save_state_dict(PATH,self.model.name+'_checkpoint')
            self.model.train()

    def validate(self,plot:bool = True):
        self.model.eval()
        test_loss = 0
        correct = 0
        dev = next(self.model.parameters()).device
        with torch.no_grad():
            for data in self.test_dataloader:
                model_inputs = torch.stack([data[0]] * self.model.n_ensembles).to(dev)
                target = data[1].to(dev)
                outputs = self.model(model_inputs)
                output = torch.mean(outputs, axis=0)
                test_loss += self.loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_dataloader)
        self.loss.append(test_loss)
        acc = 100.0 * correct / len(self.test_dataloader.dataset)
        self.val_acc.append(acc)
        if plot:
            print(f"[Valid] Average loss: {test_loss:.4f} \t Accuracy:{acc:2.2f}%")
        return acc,test_loss

class MIMO_Ensemble(MIMOModel):
    def __init__(self, model, num_classes, n_ensembles: int = 3, name='MIMO', softmax=True, *args):
        super().__init__(model, num_classes, n_ensembles, name, softmax, *args)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = [x for _ in range(self.n_ensembles)]
        x = torch.stack(x)
        return super().forward(x)