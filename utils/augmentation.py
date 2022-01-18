import numpy as np

# channel adjust
def expand_dim(x):
    """change input shape [seq] to [1, seq, 1](batch, seq, channel)
    """
    return x[np.newaxis, :, np.newaxis]

def to_torch(x):
    """change [batch, seq, channel] to [channel, seq]
    """
    return np.transpose(x, (0, 2, 1)).squeeze(0)