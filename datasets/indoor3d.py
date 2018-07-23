import os

from torch.utils.data import Dataset
import numpy as np
import h5py

ROOT_DIR = os.path.join(os.path.dirname(__file__), '../')
ROOT_DIR = os.path.abspath(ROOT_DIR)


def _load_data(filelist_txt):
    files = [line.rstrip() for line in open(filelist_txt)]
    files = [os.path.join(ROOT_DIR, 'datasets', 'data', a) for a in files]

    data_list = []
    label_list = []
    for f in files:
        h5_data = h5py.File(f)
        data_list.append(h5_data['data'][:])
        label_list.append(h5_data['label'][:])

    data_batches = np.concatenate(data_list, 0)  # 23585 x 4096 x 9
    label_batches = np.concatenate(label_list, 0)  # 23585 x 4096

    return data_batches, label_batches


def _split_idxs(roomname_txt):
    room_names = [line.rstrip() for line in open(roomname_txt)]
    train_idxs = []
    test_idxs = []
    for i, room_name in enumerate(room_names):
        if 'Area_6' in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)

    return train_idxs, test_idxs


class Indoor3D(Dataset):
    def __init__(self, mode, n_points=4096):
        super(Indoor3D, self).__init__()

        data_batches, label_batches = _load_data(
            os.path.join(ROOT_DIR, 'datasets/data/indoor3d_sem_seg_hdf5_data/all_files.txt'))

        train_idxs, test_idxs = _split_idxs(
            os.path.join(ROOT_DIR, 'datasets/data/indoor3d_sem_seg_hdf5_data/room_filelist.txt'))

        if mode == 'train':
            self.pts_set = data_batches[train_idxs, ...]
            self.labels_set = label_batches[train_idxs, ...]
        elif mode == 'test':
            self.pts_set = data_batches[test_idxs, ...]
            self.labels_set = label_batches[test_idxs, ...]

        self.pts_set = self.pts_set[:, :n_points, :]
        self.num_data = self.pts_set.shape[0]

    # Load each shape
    def __getitem__(self, idx):

        pts = self.pts_set[idx, :, :]  # n_points x 9
        labels = self.labels_set[idx]  # scalar

        # sample return
        pts = np.transpose(pts)
        sample = {'pts': pts, 'labels': labels}

        return sample

    def __len__(self):
        return self.num_data
