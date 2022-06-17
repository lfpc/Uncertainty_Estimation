import torch
import torch.nn as nn
import torch.nn.functional as F
import NN_utils.train_and_eval as TE

from tqdm.notebook import tqdm,trange

class MIMOModel(nn.Module):
    def __init__(self,model,num_classes, ensemble_num: int = 3, name = 'MIMO', *args):
        super(MIMOModel, self).__init__()
        self.model = model(num_classes = num_classes * ensemble_num, name = name, *args).cuda()
        self.ensemble_num = ensemble_num
        self.classifier_layer = self.model.classifier_layer[0]
        self.model.classifier_layer = nn.Identity()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_shape_list = list(input_tensor.size())  # (ensemble_num,batch_size,1,28,28)
        ensemble_num, batch_size = input_shape_list[0], input_shape_list[1]
        assert ensemble_num == self.ensemble_num

        input_tensor = input_tensor.view([ensemble_num * batch_size] + input_shape_list[2:])
        output = self.model(input_tensor)
        output = output.view(ensemble_num, batch_size, -1)
        output = self.classifier_layer(output)
        output = output.view(ensemble_num, batch_size, ensemble_num, -1)
        output = torch.diagonal(output, offset=0, dim1=0, dim2=2).permute(2, 0, 1)
        output = F.log_softmax(output, dim=-1)
        return output

class Trainer_MIMO(TE.Trainer):
    def __init__(self, model, optimizer, loss_criterion, training_data=None, validation_data=None, update_lr=(0,1), risk_dict=None):
        super().__init__(model, optimizer, loss_criterion, None, None, update_lr, risk_dict)
        
        self.test_dataloader = validation_data 
        self.val_acc = []
        self.loss = []
        self.validate(plot = False)

    def fit(self,train_dataloader,n_epochs = 1):
        train_dataloaders = [
        train_dataloader
        for _ in range(self.model.ensemble_num)
    ]
        self.model.train()
        progress_epoch = trange(n_epochs,position=0, leave=True, desc = 'Progress:')
        progress = tqdm(zip(train_dataloaders),position=1, leave=True, desc = 'Epoch progress:')
        for epoch in progress_epoch:

            desc = f'Loss: {self.loss[-1]:.4f} |' +desc

            desc = f'Acc_val: {self.val_acc[-1]:.2f} | ' + desc

            progress_epoch.set_description(desc)
            self.epoch += 1
            progress.disable = False
            progress.reset()

            for datum in progress:
                model_inputs = torch.stack([data[0] for data in datum]).cuda()
                targets = torch.stack([data[1] for data in datum]).cuda()

                self.optimizer.zero_grad()
                outputs = self.model(model_inputs)
                loss = self.loss_criterion(outputs.transpose(2, 1), targets)
                loss.backward()
                self.optimizer.step()

            if (self.update_lr_epochs>0) and (self.epoch%self.update_lr_epochs == 0):
                TE.update_optim_lr(self.optimizer,self.update_lr_rate)
            self.validate(plot = False)

    def validate(self,plot = True):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data in self.test_dataloader:
                model_inputs = torch.stack([data[0]] * self.config.ensemble_num).to(self.device)
                target = data[1].to(self.device)

                outputs = self.model(model_inputs)
                output = torch.mean(outputs, axis=0)
                test_loss += self.loss_criterion(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_dataloader.dataset)
        self.loss.append(test_loss)
        acc = 100.0 * correct / len(self.test_dataloader.dataset)
        self.acc.append(acc)
        if plot:
            print(f"[Valid] Average loss: {test_loss:.4f} \t Accuracy:{acc:2.2f}%")

