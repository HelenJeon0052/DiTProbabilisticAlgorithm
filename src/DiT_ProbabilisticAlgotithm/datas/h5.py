import h5py
import numpy as np
import json


import torch
from torch.utils.data import Dataset

# ------------------------------
# GCP
# ------------------------------
try:
    import h5py
except ImportError:
    raise ImportError('Please update your installation of h5py')

# ------------------------------
# h5py utils
# ------------------------------

def _normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    """
    preview only
    """
    if x.dtype == np.uint8:
        return x

    x = x.astype(np.float32)
    x = x - x.min()
    x_max = x.max()
    if x_max > 0:
        x = x / x.max

    return (x*255.0).clip(0, 255).astype(np.uint8)


def _to_ncdlw (x: np.ndarray) -> np.ndarray:
    """
    (C, D, L, W) : accepted
    (D, L, W, C) : permute
    (D, L, W) : add C (=1)
    (C, L, W) : as (C, 1, L, W)
    """

    if x.ndim == 3:
        x = x[None, ...]
    if x.ndim == 4:
        if x.shape[-1] in (1, 3, 4) and x.shape[0] not in (1, 3, 4):
            x = np.transpose(x, (3, 0, 1, 2))
        if x.shape[0] in (1, 3, 4) and x.shape[1] > 8 and x.shape[2] > 8 and x.shape[3] <= 8:
            pass
    return x

class H5Dataset(Dataset):
    def __init__(self, h5_path, splits_json, split='train', x_key=None, y_key=None):
        self.h5_path = h5_path
        with open(splits_json, 'r') as f:
            splits = json.load(f)
        meta = splits.get('meta', {})

        self.indices = np.array(splits[split], dtype=np.int64)
        self.x_key = x_key or meta.get('x_key', 'x')
        self.y_key = y_key or meta.get('y_key', 'y')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        with h5py.File(self.h5_path, 'r') as h5:
            x = np.asarray(h5[self.x_key][idx])
            y = None

            if self.y_key in h5:
                y = int(np.asarray(h5[self.y_key][idx]).reshape(-1)[0])

        x = _to_ncdlw(x).astype(np.float32)

        if x.max() > 2.0:
            x = x / 255.0
        x = torch.from_numpy(x)

        if y is None:
            return x, -1

        return x, y
