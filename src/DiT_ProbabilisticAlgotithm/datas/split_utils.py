import h5py
import numpy as np



from pathlib import Path

import torch
from torch.utils.data import Dataset


from typing import Dict, Optional, List

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

def _find_first_key(h5: h5py.File, candidates: List[str]) ->  Optional[str]:
    for k in candidates:
        if k in h5:
            return k
    return None

def make_split_h5(
        h5_path: Path,
        out_dir: Path,
        x_key: Optional[str] = None,
        y_key: Optional[str] = None,
        n_train: int = 262_144,
        n_val: int = 32_768,
        n_test: int = 32_768,
        seed: int = 0,
) -> Dict[str, int]:
    """
    x || X || images : (N, L, W, C)
    y || Y || labels : (N,), (N, 1)
    """

    set_seed(seed)
    ensure_dir(out_dir)

    with h5py.File(h5_path, 'r') as h5:
        if x_key is None:
            x_key = _find_first_key(h5,['x','X','imgs', 'images','data'])
        if y_key is None:
            y_key = _find_first_key(h5, ['y', 'Y', 'labels', 'label', 'targets'])

        if x_key is None:
            raise KeyError(f'could not find {list(h5.keys())}')
        if y_key is None:
            print(f'labels required; proceeding with no labels')

        N = h5[x_key].shape[0]

        need = n_train + n_val + n_test

        if N < need:
            raise RuntimeError(f'enough samples in h5 required: Found: {N} / need {need}')

        idx = np.arange(N)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:n_train + n_val + n_test]

        splits = {
            'meta':{
                'h5_path': str(h5_path),
                'x_key': x_key,
                'y_key': y_key,
                'N_total': int(N),
                'seed': int(seed),
                'n_train': int(n_train),
                'n_val': int(n_val),
                'n_test': int(n_test),
            },
            'train': train_idx.tolist(),
            'val': val_idx.tolist(),
            'test': test_idx.tolist(),
        }
        out_json = out_dir / 'splits.json'
        with open(out_json, 'w') as f:
            json.dump(splits, f, indent=3)

        return {'train': n_train, 'val': n_val, 'test': n_test}

def _normalize_to_unit8(x: np.ndarray) -> np.ndarray:
    """
    preview only
    """
    if x.dtype == np.unit8:
        return x

    x = x.astype(np.float32)
    x_min = x.min()
    x = x - x_min
    x_max = x.max()
    if x_max > 0:
        x = x / x.max

    return (x*255.0).clip(0, 255).astype(np.unit8)


def export_preview_npz(
        h5_path: Path,
        splits_json: Path,
        out_dir: Path,
        split : str = 'train',
        n: int = 64
) -> Path:
    """
    export a small previe subset to out_dir/previe_<split>.npz
    includes:
        images : (n, L, W, C)
        labels, indices : (n,)
    """
    ensure_dir(out_dir)
    with open(splits_json, 'r') as f:
        splits = json.load(f)
    meta = splits['meta']
    x_key = meta['x_key']
    y_key = meta.get('y_key', None)

    indices = np.array(splits[split], dtype=np.int64)
    if len(indices) < n:
        n = len(indices)
    pick = indices[:n]

    with h5py.File(h5_path, 'r') as h5:
        X = h5[x_key][pick]
        Y = None
        if y_key is not None and y_key in h5:
            Y = h5[y_key][pick]
            Y = np.asarray(Y).reshape(-1)

    X = np.asarray(X)
    X_u8 = _normalize_to_unit8(X)

    out_path = out_dir / f'preview_{split}.npz'
    if Y is None:
        np.savez_compressed(out_path, images=X_u8, indices=pick)
    else:
        np.savez_compressed(out_path, images=X_u8, labels=Y, indices=pick)

    return out_path


# -----------------------------------
# dataset builder for PCam
# -----------------------------------

def build_split(
        src_dir: Path,
        out_dir: Path,
        n_train: int = 262_144,
        n_val: int = 32_768,
        n_test: int = 32_768,
        seed: int = 0,
        copy: bool = False,
) -> Dict[str, int]:
    """
    out_dir
        train/
        val/
        test/
    index json with file lists
    """
    set_seed(seed)
    ensure_dir(out_dir)
    files = list_files(src_dir)

    if len(files) < (n_train+n_val+n_test):
        raise RuntimeError(f'enough files required, {len(files)} / {n_train+n_val+n_test}')

    random.shuffle(files)
    train_files = files[:n_train]
    val_files = files[n_train:n_train+n_val]
    test_files = files[n_train+n_val:n_train+n_val+n_test]

    splits = {
        'train' : train_files,
        'val': val_files,
        'test': test_files,
    }

    meta = {}

    for split, flist in splits.items():
        split_dir = out_dir / split
        ensure_dir(split_dir)
        meta[split] = len(flist)

        if copy:
            for f in flist:
                dst = split_dir / f.name
                if not dst.exists():
                    shutil.copy(f, dst)

    idx = {
        'src_dir': str(src_dir),
        'out_dir': str(out_dir),
        'seed': seed,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'copy': copy,
        'splits' : {
            'train': [str(p) for p in train_files],
            'val': [str(p) for p in val_files],
            'test': [str(p) for p in test_files],
        }
    }
    with open(out_dir / 'index.json', 'w') as f:
        json.dump(idx, f, indent=3)

    return meta