from numpy.lib.twodim_base import mask_indices
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
import pickle as cp
import os


def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')

def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)


    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = list(filter(lambda i : i != 1,dim))
    return strided.reshape(dim)


# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113
NB_SENSOR_CHANNELS_WITH_FILTERING = 149

# Hardcoded number of classes in the gesture recognition problem
NUM_CLASSES = 18

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 24

# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = int(SLIDING_WINDOW_LENGTH/2)
SLIDING_WINDOW_STEP_SHORT = SLIDING_WINDOW_STEP


def load_dataset(filename):
    f = open(filename, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    # print(" ..from file {}".format(filename))
    # print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
    data_x, data_y = data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)
    return data_x, data_y


class Opportunity(Dataset):
    def __init__(self, root, subset, segment_size=24, train=True, varbose=True):
        
        X_train, y_train, X_test, y_test = load_dataset(os.path.join(root, subset, 'Opportunity.data'))
        if train:
            X, y = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
        else:
            X, y = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP_SHORT)
        
        self.X = X.transpose((0, 2, 1))
        self.y = y

        self.seq_len = self.X.shape[2]
        self.dim = self.X.shape[1]
        self.size = self.X.shape[0]
        self.n_cls = len(np.unique(self.y))
        self.root = root
        self.subset = subset
        
        if varbose:
            mode = 'train' if train else 'test'
            print('='*20, mode, '='*20 )
            print('Dataset Name      :', self.subset)
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

def OpportunityLoader(data_dir, subset, batch_size, segment_size=24, transform=None, shuffle=True, train=True):
    dataset = Opportunity(data_dir, subset, segment_size, train=train, varbose=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, dataset

if __name__ == '__main__':
    data_dir = r'F:\Code_Python\MyProject\Data\UCI'
    subset = 'Opportunity'
    load, data = OpportunityLoader(data_dir, subset, 128, train=True, shuffle=True)
    x, y = iter(load).next()
    print(x.shape, y.shape)

    load, data = OpportunityLoader(data_dir, subset, 128, train=False, shuffle=False)
    x, y = iter(load).next()
    print(x.shape, y.shape)