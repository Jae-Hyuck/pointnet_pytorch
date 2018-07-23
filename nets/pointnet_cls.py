import torch.nn as nn

from .pointnet_base import PointNetBase


class PointNetCls(nn.Module):
    """
    Input: Bx3xN
    Return: Classification logits of size Bx40
    """
    def __init__(self):
        super(PointNetCls, self).__init__()

        self.base_net = PointNetBase()

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 40))

        # Init weights.
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Bx3xN

        x, _, T2 = self.base_net(x)

        # Bx1024

        x = self.classifier(x)

        # Bx40
        return x, T2
