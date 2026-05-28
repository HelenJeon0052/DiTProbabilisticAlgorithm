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

import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import LambdaLR






# ------------------------------
# utils
# ------------------------------

def set_seed(seed: int = 0):
    random.seed(seed)
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

def sample_sigma(kind, n, sigma_min, sigma_max, device, *, sigma_center, rel_width, mix_ratio, eps = 1e-12, sort_dec = False, verbose = True):
    """
    training
        args:
            kind:
            -"log uniform" : log-uniform over [sigma_min, sigma_max]
            -"band" : focused near sigma-center
            -"mix" : mix of log-uniform and band
            sigma_center : center of band or mix auxiliary sampling
        min : sigma_min_infer = 1 / sqrt(mu_max)
        max : sigma_max_infer = 1 / sqrt(mu_min)
    """
    
    
    if kind not in {"log_uniform", "band", "mix"}:
        raise ValueError(f"invalid sampling kind, {kind}")
    if sigma_min <=0 or sigma_max <= 0:
        raise ValueError(f"sigma min or sigma max must be larger than 1, got sigma_min = {sigma_min} | sigma_max = {sigma_max}")
    if sigma_min >= sigma_max:
        raise ValueError(f"sigma min must be smaller than sigma max")
    
    sigma_center = float(sigma_center)
    rel_width = float(rel_width)
    mix_ratio = float(mix_ratio)

    def _log_uniform(m):
        u = torch.rand(m, device=device)
        log_min = torch.log(torch.tensor(sigma_min + eps, device = device))
        log_max = torch.log(torch.tensor(sigma_max + eps, device = device))

        return torch.exp(log_min + (log_max - log_min) * u)

    def _band(m):
        u = torch.rand(m, device=device)
        # v = 2.0 * u - 1.0
        # v = v * torch.abs(v)
        # sigma = sigma.clamp(min = sigma_center * (1.0 - rel_width), max = sigma_center * (1.0 + rel_width))
        sigma = u * (sigma_center * rel_width) + sigma_center
        sigma = sigma.clamp(min=sigma_min, max = sigma_max)

        return sigma

    if kind == "log_uniform":
        sigma = _log_uniform(n)
    elif kind == "band":
        sigma = _band(n)
    elif kind == "mix":
        n_log = int(round(n * mix_ratio))
        n_band = n - n_log
        sigma_log = _log_uniform(n_log) if n_log > 0 else torch.empty(0, device = device)
        sigma_band = _band(n_band) if n_band > 0 else torch.empty(0, device = device)
        
        sigma = torch.cat([sigma_log, sigma_band], dim = 0)
        print(f"n_log: {n_log} | n_band: {n_band} | sigma_log : {sigma_log.shape} | sigma_band: {sigma_band.shape}")
    else:
        raise ValueError(f"invalid sampling kind, {kind}")
    
    # preventing distribution collapse >> noise injection
    sigma = sigma * torch.exp(1e-3 * torch.randn_like(sigma))
    sigma = sigma.clamp(min = sigma_min, max = sigma_max)

    if sort_dec:
        sigma = torch.sort (sigma, descending = True).values
    
    print(f"sigma : {sigma.shape} | {type(sigma)}")

    # sanity check
    if verbose:
        plt.hist(sigma.cpu().numpy(), bins = 100)
        plt.show()
        
    return sigma

def make_mu_schedules(kind, iters, sigma_min, sigma_max, device, dec_sigma, eps: float = 1e-12, use_center_band: bool=True):
    assert kind in ["geometric", "cosine", "linear"], f'not valid kind of scheduler {kind}'

    
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError(f"sigma min or sigma max must be larger than 1, got sigma_min = {sigma_min} | sigma_max = {sigma_max}")
    if sigma_min >= sigma_max:
        raise ValueError(f"sigma min must be smaller than sigma max")


    if kind == "geometric":
      sigma = torch.exp(torch.linspace(torch.log(torch.tensor(sigma_max, device=device)), torch.log(torch.tensor(sigma_min, device=device)), steps = iters, device = device))

    elif kind == "cosine":
        v = torch.linspace(0.0, 1.0, steps = iters, device = device)
        
        w = 0.5 * (1.0 + torch.cos(torch.pi * v))
        sigma = sigma_min + (sigma_max - sigma_min) * w

    elif kind ==  "linear":
        sigma = torch.linspace(sigma_max, sigma_min, steps = iters, device = device)

    if not dec_sigma:
        sigma = torch.flip(sigma, dims=[0])
    mu = 1.0 / (sigma * sigma + eps)
    return mu, sigma

def infer_sample_sigma(mus, sigma):
    sigma_min_infer, sigma_max_infer = 0.0, 0.0

    sigma_vals = [1.0 / math.sqrt(float(mu)) for mu in mus]
    sigma_min_infer = min(sigma_vals)
    sigma_max_infer = max(sigma_vals)

    
    sigma_t_min = sigma - sigma_min_infer
    sigma_t_max = sigma - sigma_max_infer
    print(f"sigma_t_min: {sigma_t_min}, sigma_t_max: {sigma_t_max}")


    sigma_min = 0.5 * sigma_min_infer
    sigma_max = 1.5 * sigma_max_infer

    print(f"sigma_min_infer: {sigma_min_infer}, sigma_max_infer: {sigma_max_infer}, sigma_min: {sigma_min}, sigma_max: {sigma_max}")
    print("infer sigma range:", sigma_min_infer, sigma_max_infer)
    
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

def normalize_sigma(sigma, g):

    if not torch.is_tensor(sigma):
      sigma = torch.tensor(sigma, device=g.device, dtype=g.dtype)
    if sigma.ndim == 0:
      sigma = sigma[None]
    if sigma.shape[0] == 1 and g.shape[0] > 1:
      sigma = sigma.repeat(g.shape[0])
    if sigma.ndim == 2 and sigma.shape[1] == 1:
      sigma = sigma[:, 0]
    return sigma

def make_gradient_denoiser(x0, t, *, grad_fn, noise: torch.Tensor | None, eps: float = 1e-12):
    """
    x0: [B, C, H, W]
    grad_fn:
            Function mapping [B, 3, H, W] -> [B, 6, H, W]
    g0:
            Clean gradient tensor [B, 6, H, W]
        g_noisy:
            Noisy gradient tensor [B, 6, H, W]
    """

    if x0.ndim != 4:
        raise ValueError(f"x0 shape mismatch, got = {x0.shape}")
    if x0.shape[1] != 3:
        raise ValueError(f"expected x0 channel  3, got = {x0.shape[1]}")
    
    if t.ndim != 1:
        raise ValueError(f"t shape mismatch, got = {t.shape}")
    if t.shape[0] != x0.shape[0]:
        print(f"t.shape: {t.shape}")
        raise ValueError(f"expected batch of t and x0 must be identical, got {t.shape[0]} | {x0.shape[0]}")
    

    g = grad_fn(x0)

    if noise is None:
        noise = torch.randn_like(g)
    else:
        if noise.shape != g.shape:
            raise ValueError(f"expected noise shape must be identical with shape of g, got= {noise.shape} | g = {g.shape}")

    sigma_view = t.to(device = x0.device, dtype = x0.dtype).clamp_min(eps)
    sigma_view = sigma_view.view(-1, 1, 1, 1)

    g_noisy = g + sigma_view * noise

    return g, g_noisy, noise, sigma_view