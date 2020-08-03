import torch
import torch.nn as nn

def training(model, train_loader, test_loader, n_epochs, criterion, optimizer, scheduler=None, device='cpu'):
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    for epoch in range(1, n_epochs+1):
        train_loss = train_acc = 0
        test_loss = test_acc = 0

        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item() * inputs.size(0)
            train_acc += torch.sum(torch.max(outputs, dim=1)[1] == labels).item()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                test_acc += torch.sum(torch.max(outputs, dim=1)[1] == labels).item()

        test_loss /= len(test_loader.dataset)
        test_acc /= len(test_loader.dataset)

        train_loss_list += [train_loss]
        train_acc_list += [train_acc]
        test_loss_list += [test_loss]
        test_acc_list += [test_acc]

        template = 'Epoch({}/{}) loss: {:.3f}, acc: {:.3f}, test_loss: {:.3f}, test_acc: {:.3f}'
        print(template.format(
            epoch, n_epochs,
            train_loss, train_acc,
            test_loss, test_acc
        ))
    
    hist = {
        'train_loss': train_loss_list,
        'train_acc': train_acc_list,
        'test_acc': test_loss_list,
        'test_acc': test_acc_list,
    }

    return hist

