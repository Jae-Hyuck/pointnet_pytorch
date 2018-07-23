import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """
    Input: BxCxN
    Return: Transformation matrix of size BxCxC
    """
    def __init__(self, ch, n_points):
        super(Transformer, self).__init__()

        self.ch = ch

        self.conv1 = nn.Conv1d(ch, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(1024)

        self.mp1 = nn.MaxPool1d(n_points)

        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn5 = nn.BatchNorm1d(256)

        # At training, the ouput transformer is initialized as identity matrix.
        self.fc3_weight = torch.nn.Parameter(data=torch.zeros(ch*ch, 256))
        self.fc3_bias = torch.nn.Parameter(data=torch.eye(ch).reshape(-1))

    def forward(self, x):
        # BxCxN

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Bx1024xN

        x = self.mp1(x)
        x = torch.squeeze(x, dim=2)

        # Bx1024

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = F.linear(x, self.fc3_weight, self.fc3_bias)
        x = x.reshape(-1, self.ch, self.ch)

        # BxCxC
        return x
