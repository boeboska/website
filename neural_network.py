import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 12, 3) # in channels, out channels , kernel size
        self.conv2 = nn.Conv2d(12, 24, 3)
        self.conv3 = nn.Conv2d(24, 36, 3)

        self.dropout = nn.Dropout(0.1)

        self.pool_2 = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(24336, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, 15)


    def forward(self, x):
        x = self.pool_2(F.relu(self.conv1(x)))
        x = self.pool_2(F.relu(self.conv2(x)))
        x = self.pool_2(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)

        x  = self.dropout(x)
        x = F.relu(self.fc1(x))

        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x