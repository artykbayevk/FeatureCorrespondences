import torch
from torch import nn
import torch.nn.functional as F

import torch.nn.functional
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.float().squeeze()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).squeeze()
        stop  =1
        loss = nn.BCEWithLogitsLoss()
        loss = loss(input=output, target=target)
        # loss = nn.BCEWithLogitsLoss(output, target)
        stop =1
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = target.float().squeeze()
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            loss = nn.BCEWithLogitsLoss()
            sigm = torch.sigmoid(output)
            predicted_vals = sigm > 0.5
            pred = (predicted_vals==1).float()

            test_loss+= loss(input=output, target=target).item()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))