from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from typing import Dict, List, Optional, Tuple

import torch

import os
import sys
import math
import random
import shutil

import numpy as np






# ------------------------------
# utils
# ------------------------------

def set_seed(seed: int = 0):
    randon.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministric = True

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_files(root: Path, exts: Tuple[str, ...]=('.png','.jpg','.jpeg', '.tif', '.tiff')) ->  List[Path]:
    files = []
    for ext in exts:
        files.extend(root.rglob(f"({ext}"))
    return files





def make_mu_schedule(kind, iters, device):
    if kind == 'log':
        return torch.logspace(-1, 2, steps=iters, device=device)
    if kind == 'linear':
        return torch.linspace(0.1, 100.0, steps=iters, device=device)

    raise ValueError(f'Unknown schedule: {kind}')