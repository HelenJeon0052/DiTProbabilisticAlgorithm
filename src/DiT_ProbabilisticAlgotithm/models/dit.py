# ----------------------------------------
# References:
# Dit : https://github.com/facebookresearch/DiT/blob/main/models.py

# based on : 2D latent
# further approach : DiT3D with CT or MRI data
# ----------------------------------------


from __future__ import annotations

# import argparse

import dataclasses
import math
import os

import random
import time
from typing import Optional, Tuple
from dataclasses import dataclass


import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import types

from torch.utils.data import Dataset, DataLoader



# ------------------------------
# utils
# ------------------------------

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def modulate(x, shift, scale):
    return x * (1+scale.unsqueeze(1))+shift.unsqueeze(1)

def seed_everything(seed: int =42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for  p in model.parameters() if p.requires_grad)

# ----------------------------------------
# Embedders
# ----------------------------------------

class TimestepEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    @staticmethod
    def sinusoidal_embedding(t, dim, max_period=10000):
        # t: (B,) long
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)

        return embedding

    def forward(self, t):
        t_freq = self.sinusoidal_embedding(t, self.dim)
        t_emb = self.mlp(t_freq)

        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes+use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids = None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = (force_drop_ids == 1)

        labels = torch.where(drop_ids, self.num_classes, labels) # torch.tensor(self.num_classes. device=labels.device)

        return labels

    def forward(self, labels, train, force_drop_ids=None):

        if (train and self.dropout_prob > 0) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)

        return embeddings

# ------------------------------
# synthetic volume dataset
# ------------------------------

class SyntheticVolumes(Dataset):
    """
    return image
    """

    def __init__(self, n_samples: int = 2048, size: Tuple[int, int, int] = (64, 64, 64), n_blobs_range: Tuple[int, int] = (1, 4), radius_range: Tuple[float, float] = (6.0, 13.0), seed: int = 123) -> None:
        super().__init__()

        self.n_samples = n_samples
        self.size = size
        self.n_blobs_range = n_blobs_range
        self.radius_range = radius_range
        self.seed = seed
        self.rng = random.Random(seed)

        D, L, W = size

        zs = torch.linspace(-1.0, 1.0, D)
        ys = torch.linspace(-1.0, 1.0, L)
        xs = torch.linspace(-1.0, 1.0, W)

        zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing='ij')
        self.grid = torch.stack([zz, yy, xx], dim = -1)  #(D, L, W, 3)

    def __len__(self) -> int:
        return self.n_samples
    def _rand_uniform(self, a: float, b: float) -> float:
        return a + (b-a) * self.rng.random()
    def __getitem__(self, idx: int) -> torch.Tensor:
        D, L, W = self.size
        vol = torch.zeros((D, L, W), dtype = torch.float32)

        n_blobs = self.rng.randint(self.n_blobs_range[0],self.n_blobs_range[1])
        # print(f'n_blobs: {n_blobs}')

        for _ in range(n_blobs):
            cz = self._rand_uniform(-.6, .6)
            cy = self._rand_uniform(-.6, .6)
            cx = self._rand_uniform(-.6, .6)

            radius = self._rand_uniform(self.radius_range[0], self.radius_range[1])
            avg_axis = (D+L+W) / 3.0

            r_norm = radius * (2.0 / avg_axis)
            center = torch.tensor([cz, cy, cx], dtype = torch.float32)
            dist = torch.norm(self.grid - center, dim=-1)
            blob = (dist <= r_norm).float()
            vol = torch.maximum(vol. blob)

        vol = vol + .05 * torch.randn_like(vol)
        vol = torch.clamp(vol, 0.0, 1.0)
        vol = vol ** .8

        # print(f'vol_**.8:{vol.shape}')

        # normalize to [-1. 1] for diffusion
        vol = vol * 2.0 - 1.0

        return vol.squeuze(0)  # (1, D, L, W)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 100 ) -> torch.Tensor:
    """
    sinusoidal timestep embedding
    """

    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)

    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((t.shape[0], 1), device=t.device)], dim=1)

    return emb

class TimeMLP(nn.Module):
    def __init__(self, dim: int, l: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, l),
            nn.SiLU(),
            nn.Linear(l, dim)
        )
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 3.0, dropout: float = .01) -> None:
        super().__init__()
        l = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, l)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(l, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) ->  torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)

        x = self.dropout(x)

        return x



class AdaLayerNorm(nn.Module):

    def __init__(self, dim, cond_dim):
        super().__init__()

        self.dim = dim
        self.cond_dim = cond_dim

        self.to_params = nn.Sequential(
            nn.Linear(cond_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.zeros_(self.to_params[-1].weight)
        nn.init.zeros_(self.to_params[-1].bias)

    def forward(self, cond):


        params = self.to_params(cond)
        (shift_msa, gate_msa, scale_msa, gate_mlp, shift_mlp, scale_mlp) = params.chunk(6, dim=-1)

        return (shift_msa, gate_msa, scale_msa, gate_mlp, shift_mlp, scale_mlp)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.d_h = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.d_h).permute(2, 0, 3, 1, 4) # (3,B,h,N,dh)

        q, k, v = qkv[0], qkv[1], qkv[2]   # (B,h,N,dh)

        # F.scaled_dot_product_attention applies 1/sqrt(d_k) scaling
        # optimized kernels(FlashAttention)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        # out = out.transpose(1, 2).contiguous().view(B, N, D)

        return self.proj(out)


class FinalLayer(nn.Module):
    """
    Continued final head:
        AdaLayerNorm > adpative shift / scale from cond > zero_init linear
    """

    def __init__(self, dim, cond_dim, out_dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.to_shift_scale = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * dim, bias=True)
        )

        nn.init.zeros_(self.to_shift_scale[-1].weight)
        nn.init.zeros_(self.to_shift_scale[-1].bias)

        self.proj = nn.Linear(dim, out_dim, bias=True)

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond) -> torch.Tensor:
        shift, scale = self.to_shift_scale(cond).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.proj(x)

        return x


class DiTBlock(nn.Module):
    """6D modulation"""
    def __init__(self, dim, cond_dim, num_heads = 8, mlp_ratio = 4.0, dropout: float = 0.08):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(dim = dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.dim = dim
        self.cond_dim = cond_dim

        self.adaLN = AdaLayerNorm(dim = self.dim, cond_dim = self.cond_dim)

    def forward(self, x, cond):
        # x : (B, N, dim), cond: (B, cond_dim)

        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.adaLN(cond)

        # Attention
        l = self.norm1(x)
        l = modulate(l, shift_msa, scale_msa)

        x = x + gate_msa.unsqueeze(1) * self.attn(l)
        print(f'first_x: {x.shape}')

        # mlp
        l = self.norm2(x)
        l = modulate(l, shift_mlp, scale_mlp)

        x = x + gate_mlp.unsqueeze(1) * self.mlp(l)
        print(f'second_x: {x.shape}')

        return x

class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch: int) ->  None:
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        print(f"[patch embed] x: {x.shape}")
        x = self.proj(x)
        print(f"[patch embed] xx: {x.shape}")
        B, E, Lp, Wp = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous() # [Batch, Num_Patches, Embed_Dim]
        print(f"[patch embed] tokens: {tokens.shape}")

        return tokens, (Lp, Wp)


class UnpatchifyEmbed(nn.Module):
    """
    tokens > (B, out_ch, Lp, Wp)
    """

    def __init__(self, embed_dim, out_ch, patch) -> None:
        super().__init__()

        self.patch = patch
        self.proj = nn.ConvTranspose2d(embed_dim, out_ch, kernel_size=patch, stride=patch)

    def manual_unpatchify(self, tokens, grid) -> torch.Tensor:
        B, N, D = tokens.shape
        Lp, Wp = grid
        p = self.cfg.patch
        C = self.cfg.out_channels

        if N != Lp * Wp:
            raise ValueError(f"N must equal Gh*Gw, got N={N}, grid={grid}")

        expected_D = C * p * p
        if D != expected_D:
            raise ValueError(f"D must be C*p*p={expected_D}, got D={D}")

        x = tokens.reshape(B, Lp, Wp, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.reshape(B, C, Lp * p, Wp * p)

        return x

    def forward(self, tokens, grid) -> torch.Tensor:
        B, N, D = tokens.shape
        Lp, Wp = grid

        x = tokens.transpose(1, 2).reshape(B, D, Lp, Wp).contiguous() # D = out_channels * patch * patch , [B, C, H, W]

        return self.proj(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# ------------------------------
# DiT Model : noise predictor
# ------------------------------

@dataclass
class DiT2DConfig:
    in_channels: int = 6
    out_channels: int = 6
    img_size : int = 32
    patch: int = 4
    embed_dim : int = 256
    depth : int = 6
    num_heads : int = 8
    mlp_ratio : float = 3.0
    dropout : float = 0.0
    time_in : int = 2
    time_embed_dim : int = 256
    center_output: bool = True

class DiT2D(nn.Module):
    """
    -patchify input img(2d) into tokens
    -add positional embedding
    -condition transformer blocks on t-emb
    -unpatchify tokens into predicted noise img (2d)
    -objective : ddpm denoising
    """

    def __init__(self, cfg: DiT2DConfig):
        super().__init__()

        self.cfg = cfg
        # print(f"DiT2D Config: {dataclasses.asdict(cfg)}")
        self.patch = PatchEmbed(cfg.in_channels, cfg.embed_dim, cfg.patch)
        self.unpatchify = UnpatchifyEmbed(cfg.embed_dim, cfg.out_channels, cfg.patch)

        self.time_emb = TimestepEmbedder(cfg.time_embed_dim)
        self.cond_proj = nn.Linear(cfg.time_embed_dim, cfg.time_embed_dim)

        num_tokens = (cfg.img_size // cfg.patch) ** 2
        self.postn = nn.Parameter(torch.zeros(1, num_tokens, cfg.embed_dim))
        # nn.init.trunc_normal_(self.postn, std=0.02)

        self.in_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.blocks = nn.ModuleList([
            DiTBlock(cfg.embed_dim, cfg.embed_dim, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout)
            for _ in range(cfg.depth)
        ])

        self.out_norm = nn.LayerNorm(cfg.embed_dim, elementwise_affine=False, eps=1e-6)
        self.out = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=True)

        self.final = FinalLayer(
            dim = cfg.embed_dim,
            cond_dim = cfg.embed_dim,
            out_dim = cfg.embed_dim
        )

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        self._init_weights()

    def _init_weights(self) -> None:
        if hasattr(self.unpatchify, "proj"):
            if getattr(self.unpatchify.proj, "bias", None) is not None:
                nn.init.zeros_(self.unpatchify.proj.bias)

    def set_pos_embed(self, pos_embed) -> None:
        self.postn = pos_embed


    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        print(f"input x: {x.shape} | t: {t.shape}")
        tokens, grid = self.patch(x)
        print(f"tokens : {tokens.shape} | grid: {grid}")
        if self.postn is not None:
            tokens = tokens + self.postn
        tokens = self.in_proj(tokens)

        cond = self.cond_proj(self.time_emb(t))

        for blk in self.blocks:
            tokens = blk(tokens, cond)

        tokens = self.final(self.out_norm(tokens), cond)
        eps = self.unpatchify(tokens, grid)

        if self.cfg.center_output:
            # [B, 1, 1, 1] >> [B, C, 1, 1]
            eps = eps - eps.mean(dim=(1, 2, 3), keepdim = True)
        return eps
