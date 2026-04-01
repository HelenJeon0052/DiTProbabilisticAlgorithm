import torch
import torch.nn
import math

import os
import csv
import time
import json
import random
import argparse
from torch.utils.data import DataLoader

from DiT_ProbabilisticAlgotithm.models.diffusion import DiffusionSchedule, logit_shift_xt_vs_x0hat
from DiT_ProbabilisticAlgotithm.data.h5 import H5Dataset
from DiT_ProbabilisticAlgotithm.utils.utils import ensure_dir, set_seed


# ------------------------------
# utils
# ------------------------------

def set_seed(seed: int = 0):
    randon.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ------------------------------
# Wrapper
# ------------------------------

class EpsWrapper:
    def __init__(self, net, is_dit: bool):
        super().__init__()
        self.net = net
        self.is_dit = is_dit

    def forward(self, x_t, t, y=None):
        if self.is_dit:
            return self.net(x_t, t, y)

        return self.net(x_t, t)


class Classifiers3D(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, paddig=1), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        l = self.net(x).flatten(1)
        return self.fc(l)


# ------------------------------
# csv utility
# ------------------------------

def append_csv(path, row: dict):
    exists = os.path.exists(path)

    with open(path, 'a', newline='') as f:
        w = cvs.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)

# ------------------------------
# CLI
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--make_split', action='store_true', help='Build split index (copy is optional)')
    ap.add_argument('--src', type=str, default='', help='Source directory containing images')
    ap.add_argument('--out', type=str, default='', help='output directory for split/index')
    ap.add_argument('--copy', action='store_true', help='copy files into train/val/test')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sample_demo', action='store_true', help='run ddpm sampling with dummy model')
    ap.add_argument('--schedule', type=str, default='cosine', choices=['linear', 'cosine'])
    ap.add_argument('--T', type=int, default=50)
    ap.add_argument('--B', type=int, default=2)
    ap.add_argument('--C', type=int, default=3)
    ap.add_argument('--L', type=int, default=96)
    ap.add_argument('--W', type=int, default=96)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    ap.add_argument('--make_split_h5', action='store_true', help='Vuild split index with h5')
    ap.add_argument('--h5', type=str, default='', help='path to pcam HDF5')
    # ap.add_argument('--out', type=str, default='./rough_pcam_h5',help='output directory of splits.json')
    ap.add_argument('--x_key', type=str, default=None)
    ap.add_argument('--y_key', type=str, default=None)
    ap.add_argument('--export_preview', action='store_true')
    ap.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--results', type=str, default='results.csv')
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--backbone', type=str, choices=['dit3d', 'vit3d'], default='dit3d')
    ap.add_argument('--in_channels', type=int, default=1)

    args = ap.parse_args()







    set_seed(args.seed) # torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ds = H5Dataset(args.h5, args.split, split='val', x_key=args.x_key, y_key=args.y_key)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    out_dir = Path(args.out)
    ensure_dir(out_dir)


    if args.backbone == 'dit3d':
        net = DiT3D(in_channels=args.in_channels, dim=384, depth=6, heads=8, patch=(2, 4, 4))
        eps_model = EpsWrapper(net, is_dit=True)
    else:
        net = ViT3D(in_channels=args.in_channels, dim=384, patch=(2, 4, 4))
        eps_model = EpsWrapper(net, is_dit=False)

    eps_model = eps.model.to(device).eval()

    clf = Classifiers3D(in_channels=args.in_channels).to(device).eval()

    sched = DiffusionSchedule(T=args.T, kind=args.schedule)
    pre = sched.precompute(device=device)

    mals_list, filp_list, bvr_list = [], [], []

    t0 = time.time()

    for step, (x0, y) in enumerate(dl):
        if step >= args.steps:
            break
        x0 = x0.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True) if isinstance(y, torch.Tensor) else None

        t = torch.randint(0, args.T, (x0.size(0),), device=device, dtype=torch.long)

        m = logit_shift_xt_vs_x0hat(
            clf = clf,
            eps_model = eps_model,
            x0=x0,
            t=t,
            y=y,
            tau=args.tau,
            pre=pre,
        )

        mals_list.append(m['mals'])
        filp_list.append(m['flip_rate'])
        bvr_list.append(m['bvr'])

    dt = time.time() - t0

    row = {
        'seed': args.seed,
        'backbone': args.backbone,
        'schedule':args.schedule,
        'T': args.T,
        'tau': args.tau,
        'mals': float(sum(mals_list)/len(mals_list)),
        'filp_rate': float(sum(filp_list)/len(filp_list)),
        'bvr': float(sum(bvr_list)/len(bvr_list)),
        'steps_eval': args.steps,
        'time': dt,
        'device': device,
    }

    append_csv(args.results, row)
    print('wrote:', row)

if __name__ == '__main__':
    main()


