import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

import torch
import torch.optim as optim

from DL.dataLoader import train_loader, test_loader, val_loader, input_size
from DL.model import DLModel
from DL.inference import train, test

# warnings.filterwarnings("ignore")
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

test(model, device, test_loader)
