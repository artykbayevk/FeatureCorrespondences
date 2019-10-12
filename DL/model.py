import torch.nn as nn
import torch.nn.functional as F

class DLModel(nn.Module):
    def __init__(self, input_size):
        super(DLModel, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=100, out_features=50),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=50, out_features=25),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=25, out_features=1)
        )

    def forward(self, x):
        # x = F.normalize(x.float(), p=2, dim=1)
        x = x.float()
        x = self.linear(x)
        return x
