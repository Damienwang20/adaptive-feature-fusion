from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import os

class UCRArchive2018(Dataset):
    def __init__(self, root, subset, train=True, normlize=True, transform=None, varbose=True, downsample=False):
        self.root = root
        self.downsample = downsample
        if train:
            self.dataset = np.loadtxt(os.path.join(self.root, subset, subset+"_TRAIN.tsv"))
        else:
            self.dataset = np.loadtxt(os.path.join(self.root, subset, subset+"_TEST.tsv"))

        self.transform = transform
        # unique_class = np.sort(np.unique(self.dataset[:, 0]))
        # dict_i_class = {cl:i for i, cl in enumerate(unique_class)}
        # for i in range(len(self.dataset)):
        #     self.dataset[i, 0] = dict_i_class[self.dataset[i, 0]]

        # self.dataset = torch.FloatTensor(self.dataset)
        self.X = self.dataset[:, 1:]
        self.y = self.dataset[:, 0]
        self.y = LabelEncoder().fit_transform(self.y)

        self.seq_len = self.X.shape[1]
        if self.downsample:
            self.reduce_len = int(self.seq_len / 3)
            
        self.n_cls = len(np.unique(self.y))
        self.size = self.X.shape[0]
        self.subset = subset

        if normlize:
            X_mean = self.X.mean()
            X_std = self.X.std()
            self.X = (self.X - X_mean) / (X_std + 1e-8)

        if varbose:
            mode = 'train' if train else 'test'
            print('='*20, mode, '='*20 )
            print('Dataset Name      :', subset)
            print('Dataset Size      :', self.size)
            print('Number of Class   :', self.n_cls)
            print('Time Series Length:', self.seq_len)
            print()

    def __getitem__(self, index):
        data = self.X[index]
        label = self.y[index]

        # data augmentation
        data = self.transform[0](data)
        for tsfm in self.transform[1:-1]:
            if np.random.random()<0.5:
                data = tsfm(data)
        data = self.transform[-1](data)

        # to torch tensor
        data = torch.FloatTensor(data)
        if self.downsample:
            data = torch.nn.functional.interpolate(data[None, :], size=int(self.seq_len / 3), mode='linear',
                                                   align_corners=False)
            data = data[0, ...]
        label = torch.tensor(label).long()
        return data, label

    def __len__(self):
        return self.size

    def __str__(self):
        data_info = f"""
                          Dataset Name       : {self.subset}
                          Dataset Size       : {self.size}
                          Number of Class    : {self.n_cls}
                          Time Series Length : {self.seq_len}
                     """
        return data_info

    @property
    def class_weigth(self):
        classes = np.unique(self.y.numpy())
        le = LabelEncoder()
        y_ind = le.fit_transform(self.y.numpy().ravel())
        recip_freq = len(self.y) / (len(le.classes_) *
                                    np.bincount(y_ind).astype(np.float64))
        class_weight = recip_freq[le.transform(classes)]
        class_weight = torch.FloatTensor(class_weight)
        return class_weight.cuda() if torch.cuda.is_available() else class_weight


def UCRArchive2018Loader(data_dir, subset, batch_size, transform=None, shuffle=True, train=True, downsample=False):
        dataset = UCRArchive2018(data_dir, subset, train=train, transform=transform, normlize=True, varbose=True, downsample=downsample)
        # batch_size = min(batch_size, dataset.size // 10)
        batch_size = batch_size
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader, dataset