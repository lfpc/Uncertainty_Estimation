def train_NN_with_g(model,optimizer,data,loss_criterion,n_epochs=1, print_loss = True):
    '''Train a NN that has a g layer'''
    dev = next(model.parameters()).device
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