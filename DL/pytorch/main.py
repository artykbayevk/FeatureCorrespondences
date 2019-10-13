import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from DL.pytorch.dataLoader import train_loader, test_loader, val_loader, input_size
from DL.pytorch.model import DLModel
from DL.pytorch.inference import train, test

plt.ion()

#%% main function
n_epochs = 100
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = DLModel(input_size=input_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, n_epochs+1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, val_loader)
stop = 1
test(model, device, test_loader)
