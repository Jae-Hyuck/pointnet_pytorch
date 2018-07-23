import torch
import torch.nn as nn

from .pointnet_base import PointNetBase


class PointNetSeg(nn.Module):
    """
    Input: Bx9xN
    Return: Segmentation logits of size Bx13xN
    """
    def __init__(self):
        super(PointNetSeg, self).__init__()

        self.base_net = PointNetBase(ch=9, n_points=4096)

        self.global_embed_net = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.seg_net = nn.Sequential(
            nn.Conv1d(1152, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(256, 13, 1, bias=False),
        )

        # Init weights.
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: Bx9xN
        n_points = x.shape[2]

        global_feature, point_feature, T2 = self.base_net(x)
        # global_feature: Bx1024
        # point_feature: Bx1024xN

        global_feature = self.global_embed_net(global_feature)
        # global_feature: Bx128
        global_feature = global_feature.unsqueeze(-1).repeat(1, 1, n_points)
        # global_feature: Bx128xN
        concat_feature = torch.cat([point_feature, global_feature], dim=1)
        # concat_feature: Bx1152xN

        y = self.seg_net(concat_feature)
        # y: Bx13xN

        return y, T2
