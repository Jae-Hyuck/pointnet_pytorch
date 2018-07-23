import os

from torch.utils.data import Dataset
import numpy as np
import h5py

from . import provider

ROOT_DIR = os.path.join(os.path.dirname(__file__), '../')
ROOT_DIR = os.path.abspath(ROOT_DIR)


def _load_data(filelist_txt):

    files = [line.rstrip() for line in open(filelist_txt)]
    files = [os.path.join(ROOT_DIR, 'datasets', a) for a in files]

    data_list = []
    label_list = []
    for f in files:
        h5_data = h5py.File(f)
        data_list.append(h5_data['data'][:])
        label_list.append(h5_data['label'][:])

    return np.vstack(data_list), np.vstack(label_list).squeeze()


class ModelNet40(Dataset):
    def __init__(self, mode, n_points):
        super(ModelNet40, self).__init__()

        self.mode = mode

        if mode == 'train':
            self.pts_set, self.labels_set = _load_data(
                os.path.join(ROOT_DIR, 'datasets/data/modelnet40_ply_hdf5_2048/train_files.txt'))
        elif mode == 'test':
            self.pts_set, self.labels_set = _load_data(
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
