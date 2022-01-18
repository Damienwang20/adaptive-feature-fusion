from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np

def load_uci_har(root, subset, train=True):
    test = '' if train else '_test' 
    fa = open(root+"/"+subset+"/all_data"+ test + ".csv")
    fb = open(root+"/"+subset+"/answers" + test + ".csv")
    data = np.loadtxt(fname = fa, delimiter = ',')
    labels = np.loadtxt(fname = fb, delimiter = ',')
    fa.close(); fb.close();
    return data, labels

def transform_uci_har(data, labels):
    def norm(x):
        temp = x.T - np.mean(x.T, axis = 0)
        #x = x / np.std(x, axis = 1)
        return temp.T
    
    num_input_channels = 6
    segment_size = 128
    n_classes = 6
    
    for i in range(num_input_channels):
        x = data[:, i * segment_size : (i + 1) * segment_size]
        data[:, i * segment_size : (i + 1) * segment_size] = norm(x)

    size = data.shape[0]

    data = np.reshape(data, [data.shape[0], segment_size * num_input_channels])
    labels = np.reshape(labels, [labels.shape[0], n_classes])
    labels_unary = np.argmax(labels, axis=1)
    
    data = data.reshape(-1, segment_size, num_input_channels).transpose((0, 2, 1))
    return data, labels_unary

class UCI_HAR(Dataset):
    def __init__(self, root, subset, train=True, varbose=True):
        self.root = root
        X, y = load_uci_har(root, subset, train=train)
        self.X, self.y = transform_uci_har(X, y)

        self.seq_len = self.X.shape[2]
        self.dim = self.X.shape[1]
        self.size = self.X.shape[0]
        self.n_cls = len(np.unique(self.y))
        self.subset = subset
        
        if varbose:
            mode = 'train' if train else 'test'
            print('='*20, mode, '='*20 )
            print('Dataset Name      :', 'HAR')
            print('Dataset Size      :', self.size)
            print('Number of Class   :', self.n_cls)
            print('Time Series Diment:', self.dim)
            print('Time Series Length:', self.seq_len)
            print()

    def __getitem__(self, index):
        data = self.X[index]
        label = self.y[index]
        
        # to torch tensor
        data = torch.FloatTensor(data)
        label = torch.tensor(label).long()
        return data, label

    def __len__(self):
        return self.size

    def __str__(self):
        data_info = f"""
                          Dataset Name       : {self.subset}
                          Dataset Size       : {self.size}
                          Number of Class    : {self.n_cls}
                          Time Series Diment : {self.dim}
                          Time Series Length : {self.seq_len}
                     """
        return data_info

def UCIHarLoader(data_dir, subset, batch_size, transform=None, shuffle=True, train=True):
    dataset = UCI_HAR(data_dir, subset, train=train, varbose=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, dataset