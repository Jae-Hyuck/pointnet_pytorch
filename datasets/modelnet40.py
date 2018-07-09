import os

from torch.utils.data import Dataset
import numpy as np

from . import provider

ROOT_DIR = os.path.join(os.path.dirname(__file__), '../')
ROOT_DIR = os.path.abspath(ROOT_DIR)


class ModelNet40(Dataset):
    def __init__(self, mode, n_points):
        super(ModelNet40, self).__init__()

        self.mode = mode

        if mode == 'train':
            self.pts_set, self.labels_set = provider.load_merged_data(
                os.path.join(ROOT_DIR, 'datasets/data/modelnet40_ply_hdf5_2048/train_files.txt'))
        elif mode == 'test':
            self.pts_set, self.labels_set = provider.load_merged_data(
                os.path.join(ROOT_DIR, 'datasets/data/modelnet40_ply_hdf5_2048/test_files.txt'))

        self.pts_set = self.pts_set[:, :n_points, :]
        self.num_data = self.pts_set.shape[0]

    # Load each shape
    def __getitem__(self, idx):

        pts = self.pts_set[idx, :, :]  # n_points x 3
        labels = self.labels_set[idx]  # scalar

        if self.mode == 'train':
            pts = provider.rotate_point_cloud(pts)
            pts = provider.jitter_point_cloud(pts)

        # sample return
        pts = np.transpose(pts)
        sample = {'pts': pts, 'labels': labels}

        return sample

    def __len__(self):
        return self.num_data
