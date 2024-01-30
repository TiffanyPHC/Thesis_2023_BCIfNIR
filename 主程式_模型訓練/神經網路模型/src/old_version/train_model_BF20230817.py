import torch
import torch.nn as nn
import wandb

def test_EEG_kernel(train_loader, test_loader, model, optimizer, epoch, learning_rate, weight_decay, wandb_import=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    if wandb_import:
        wandb.watch(model)
        
    criterion = nn.CrossEntropyLoss()
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.999), eps=1e-3, weight_decay=weight_decay)
    total_epoch = epoch
    accfinal=0
    test_correct = 0
    test_total = 0

    # print('epoch, train, test')
    print('epoch, train, test, train_loss, test_loss')
    for epoch in range(total_epoch):  # loop over the dataset multiple times
        model.train()
        train_loss=0
        test_loss=0    
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        test_correct = 0
        test_total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                # calculate outputs by running images through the network
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss = test_loss + loss.item()
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                _, l = torch.max(labels, 1)
                test_total += labels.size(0)
                test_correct += (predicted == l).sum().item()


        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for data in train_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                train_loss = train_loss + loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, l = torch.max(labels, 1)
                train_total += labels.size(0)
                train_correct += (predicted == l).sum().item()
        # print(f'{epoch}, {100 * train_correct / train_total:.2f}%, {100 * test_correct / test_total:.2f}%')
        print(f'{epoch}, {100 * train_correct / train_total:.2f}%, {100 * test_correct / test_total:.2f}%, {train_loss/train_total:.5f}, {test_loss/test_total:.5f}')
        if wandb_import:
            metrics = {
                "train_loss": train_loss/train_total, 
                "train_accuracy": 100 * train_correct / train_total,
                "test_loss": test_loss/test_total, 
                "test_accuracy": 100 * test_correct / test_total}
            wandb.log(metrics)
            if accfinal<(test_correct/ test_total) :
                accfinal=test_correct/ test_total
    
    if wandb_import:
        wandb.log({"best test Accuracy": accfinal*100})
        wandb.finish()
        
    return model, criterion, optimizer
  
def fine_turning_model(FT_loader, test_loader, epoch, model, criterion, optimizer):
  
  total_epoch = epoch
  accfinal=0
  test_correct = 0
  test_total = 0
  
  # print('epoch, train, test')
  print('epoch, train, test, train_loss, test_loss')
  for epoch in range(total_epoch):  # loop over the dataset multiple times
    model.train()
    train_loss=0
    test_loss=0    
    for i, data in enumerate(train_loader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
      inputs = inputs.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

    model.eval()
    test_correct = 0
    test_total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
      for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss = test_loss + loss.item()
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        _, l = torch.max(labels, 1)
        test_total += labels.size(0)
        test_correct += (predicted == l).sum().item()


    train_correct = 0
    train_total = 0
    with torch.no_grad():
      for data in train_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss = train_loss + loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, l = torch.max(labels, 1)
        train_total += labels.size(0)
        train_correct += (predicted == l).sum().item()
    # print(f'{epoch}, {100 * train_correct / train_total:.2f}%, {100 * test_correct / test_total:.2f}%')
    print(f'{epoch}, {100 * train_correct / train_total:.2f}%, {100 * test_correct / test_total:.2f}%, {train_loss/train_total:.5f}, {test_loss/test_total:.5f}')
  #   metrics = {
  #     "train_loss": train_loss/train_total, 
  #     "train_accuracy": 100 * train_correct / train_total,
  #     "test_loss": test_loss/test_total, 
  #     "test_accuracy": 100 * test_correct / test_total}
  #   wandb.log(metrics)
  #   if accfinal<(test_correct/ test_total) :
  #     accfinal=test_correct/ test_total
  # wandb.log({"best test Accuracy": accfinal*100})
  # wandb.finish()
  return model, criterion, optimizer