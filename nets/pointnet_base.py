import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Transformer


class PointNetBase(nn.Module):
    """
    Input: Bx3xN
    Return: Global_feature of size Bx1024
    """
    def __init__(self, ch=3, n_points=1024):
        super(PointNetBase, self).__init__()

        self.input_transformer = Transformer(ch=ch, n_points=n_points)

        self.conv1 = nn.Conv1d(ch, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(64)

        self.feat_transformer = Transformer(ch=64, n_points=n_points)

        self.conv3 = nn.Conv1d(64, 64, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, 1, bias=False)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 1024, 1, bias=False)
        self.bn5 = nn.BatchNorm1d(1024)

        self.mp1 = nn.MaxPool1d(n_points)

    def forward(self, x):

        # input transform
        T1 = self.input_transformer(x)
        x = torch.bmm(T1, x)

        # Bx3xN
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # feature transform
        T2 = self.feat_transformer(x)
        x = torch.bmm(T2, x)

        # Bx64xN
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        point_feature = x

        # Bx1024xN
        x = self.mp1(x)
        global_feature = torch.squeeze(x, dim=2)

        # Bx1024
        return global_feature, point_feature, T2
