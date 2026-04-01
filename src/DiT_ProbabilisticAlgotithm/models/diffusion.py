# for Dit3D comparison
# setting for: h5py
# eps-pred parameterization : model(x_t, t) > eps_hat (x_t)



from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import math
import json
import random
import shutil
import argparse

import numpy as np

from DiT_ProbabilisticAlgotithm.utils.utils import ensure_dir, set_seed

# -----------------------------
# Diffusion Config
# -----------------------------
@dataclass
class DiffusionConfig:
    T : int = 200
    beta_start : float = 1e-3
    beta_end : float = 2e-2


# -----------------------------
# Diffusion Schedule
# -----------------------------

@dataclass
class DiffusionSchedule:
    def __init__(self, cfg: DiffusionConfig):
        self.cfg = cfg
        self.T = cfg.T
        self.beta_start = cfg.beta_start
        self.beta_end = cfg.beta_end
        self.kind = 'cosine' # 'linear'

    def betas(self, device) -> torch.Tensor:
        if self.kind == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.T, device=device)

        elif self.kind == 'cosine':
            steps = self.T
            s = .008
            t = torch.linspace(0, steps, steps + 1, device=device) /steps
            f = torch.cos(((t + s) / (1 + s)) * math.pi /2) ** 2
            alpha_bar = f / f[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])

            return betas.clamp(1e-8, 0.999)

        else:
            raise ValueError(f'unknown schedule: {self.kind}')

    """
    inverse loop : pre > stale
    def precompute(self, device) -> Dict[str, torch.Tensor]:
        betas = self.betas(device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

        alpha_bar_prev = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]], dim=0)
        posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        posterior_var = posterior_var.clamp(min=1e-20)

        return dict(
            betas=betas,
            alphas=alphas,
            alpha_bar=alpha_bar,
            sqrt_alpha_bar=sqrt_alpha_bar,
            sqrt_one_minus_alpha_bar=sqrt_one_minus_alpha_bar,
            alpha_bar_prev=alpha_bar_prev,
            posterior_var=posterior_var,
        )
    """

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) ->  torch.Tensor:
    B = t.shape[0]
    out = a.gather(0, t) # (B,)
    return out.view(B, *([1] * (len(x_shape) - 1)))

# -------------------------------
# Diffusion Class
# -------------------------------

class GaussianDiffusion(nn.Module):
    def __init__(self, cfg: DiffusionConfig, schedule: DiffusionSchedule, device):
        super().__init__()
        self.cfg = cfg

        betas = schedule.betas(device)
        alphas = 1.0 - betas

        eps = 1e-5

        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]], dim=0)



        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)

        self.register_buffer('alpha_bar_prev', alpha_bar_prev)
        self.register_buffer('sqrt_alpha_bar', sqrt_alpha_bar)
        self.register_buffer('sqrt_one_minus_alpha_bar', sqrt_one_minus_alpha_bar)

        posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        self.register_buffer('posterior_variance', posterior_variance.clamp(min=eps))

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        B = t.shape[0]
        out = a.gather(0, t)

        return out.view(B, *([1] * (len(x_shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise=None) -> torch.Tensor:
        """
        :return:
            x_t: sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise
        """

        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alpha_bar, t, x0.shape)
        sqrt_mab = self._extract(self.sqrt_one_minus_alpha_bar,t, x0.shape)

        xt = sqrt_ab * x0 + sqrt_mab * noise

        return xt, noise

    def p_losses(self, model, x0, t):
        xt, eps = self.q_sample(x0, t)
        eps_hat = model(xt, t)

        return F.mse_loss(eps_hat, eps)


    @torch.no_grad()
    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps_hat) -> torch.Tensor:
        """
        :return:
        x0_hat = (x_t - sqrt(1-alpha_bar)*eps) / sqrt(alpha_bar)
        """

        sqrt_ab = self._extract(self.sqrt_alpha_bar, t, x_t.shape)
        sqrt_mab = self._extract(self.sqrt_one_minus_alpha_bar, t, x_t.shape)

        x0 = (x_t - sqrt_mab * eps_hat) / sqrt_ab

        return x0

    @torch.no_grad()
    def ddim_step(self, model, x_t, t, t_prev, eta=0.0):
        eps_hat = model(x_t, t)
        x0 = self.predict_x0_from_eps(x_t, t, eps_hat)

        ab = self._extract(self.alpha_bar, t, x_t.shape)
        ab_prev = self._extract(self.alpha_bar_prev, t_prev, x_t.shape)

        # sigma per DDIM
        # DDIM(η>0): stochastic sampling
        # DDIM(η=0): HQS / inverse loop prior step
        sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab)) * torch.sqrt(1 - ab / ab_prev)
        noise = torch.randn_like(x_t)

        mean = torch.sqrt(ab_prev) * x0 + torch.sqrt(1 - ab_prev - sigma**2) * eps_hat
        mask = (t != 0).float().view(x_t.shape[0], *([1] * (x_t.ndim - 1)))

        x_prev = mean + mask * sigma * noise

        return x_prev, x0

    @torch.no_grad()
    def p_sample_ddpm(
            self,
            x_t: torch.Tensor,
            t: torch.Tensor,
            model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        ddpm reverse step : x_t-1 ~ N(mu, var)
        """

        betas_t = self._extract(self.betas, t, x_t.shape)
        alphas_t = self._extract(self.alphas, t, x_t.shape)
        alpha_bar_t = self._extract(self.alpha_bar, t, x_t.shape)
        alpha_bar_prev = self._extract(self.alpha_bar_prev, t, x_t.shape)
        var = self._extract(self.posterior_variance, t, x_t.shape)

        eps_hat = model(x_t, t)
        x0_hat = (x_t - torch.sqrt(1.0 - alpha_bar_t)*eps_hat) / (torch.sqrt(alpha_bar_t) + 1e-12)

        # posterior mean
        coef1 = torch.sqrt(alpha_bar_prev) * betas_t / (1.0 - alpha_bar_t + 1e-12)
        coef2 = torch.sqrt(alphas_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t + 1e-12)
        mu = coef1 * x0_hat + coef2 * x_t

        # if t == 0, return mean (no noise)
        noise = torch.randn_like(x_t)
        nonzero = (t != 0).float().view(-1, *([1] * (x_t.dim() - 1)))

        return mu + nonzero * torch.sqrt(var) * noise

    @torch.no_grad()
    def sample_ddpm(
            self,
            shape: Tuple[int, ...],
            model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            device: torch.device,
    ) ->  torch.Tensor:
        """
        ddpm sampling loop from x_T ~ N(0, I)
        """

        x = torch.randn(shape, device=device)

        for ti in reversed(range(self.cfg.T)):
            t = torch.full((shape[0],), ti, device=device, dtype=torch.long)
            x = self.p_sample_ddpm(x, t, model)

        return x

"""diffusion = GaussianDiffusion
sample = diffusion.sample_ddpm(model, (1, 4, 64, 64), device)

print(
    sample.min().item(),
    sample.max().item(),
    sample.mean().item(),
    sample.std().item()
)"""
