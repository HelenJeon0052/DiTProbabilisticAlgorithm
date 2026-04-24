import os, csv, time, math, random
import numpy as np


import gc
import itertools
import traceback

import torch
from prodigyopt import Prodigy
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

import torch.optim as optim
from typing import Any, Dict, Iterable, Tuple
from tqdm import tqdm
from dataclasses import dataclass

from DiT_ProbabilisticAlgotithm.models.dit import DiT2D, DiT2DConfig
from DiT_ProbabilisticAlgotithm.utils.make_dit_builder import make_dit_builder
from DiT_ProbabilisticAlgotithm.utils.metric_utils import normalize_sigma
from DiT_ProbabilisticAlgotithm.pnp.pnpstarter import hqs_solve
from DiT_ProbabilisticAlgotithm.datas.pcam_starter import show_batch, PCamDataset, build_pcam_loader
from DiT_ProbabilisticAlgotithm.ops.blur import make_gaussian_kernel, blur2d, mse, psnr, estimate_optimal_norm, synth_blur_noise

from DiT_ProbabilisticAlgotithm.utils.utils import set_seed, make_mu_schedules, sample_sigma, infer_sample_sigma, get_cosine_schedule_with_warmup
from DiT_ProbabilisticAlgotithm.models.diffusion import DiffusionConfig, GaussianDiffusion, DiffusionSchedule



def append_csv(path: str, fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_hd = not os.path.exists(path)   


def make_A_AT(k, sigma_blur, AT_mode:str, device):    
    
    kernel = make_gaussian_kernel(k=k, sigma=sigma_blur, device=device, dtype=torch.float32)
    kernel_flip = torch.flip(kernel, dims=[-1, -2])

    # Operators A=I
    A = lambda x: blur2d(x, kernel)

    if AT_mode == 'blur':
        AT = lambda y: blur2d(y, kernel)
    elif AT_mode == 'inverse':
        AT = lambda y: blur2d(y, kernel_flip)
    else:
        raise ValueError(f'Unknown AT_mode: {AT_mode}')
    
    return A, AT
        


def _extract_batch_tensor(batch):
    if isinstance(batch, dict):
        if "image" in batch:
            return batch["image"]
        raise KeyError(f"unsupported batch keys: {list(batch.keys())}")


    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch

def build_diffusion_instance(device, cfg = DiffusionConfig | None):
    if cfg is None:
        cfg = DiffusionConfig()
    schedule = DiffusionSchedule(cfg=cfg)

    diffusion_instance = GaussianDiffusion(cfg=cfg, schedule=schedule, device=device)
    return diffusion_instance

def get_cfg_key(
    *,
    seed,
    # denoiser_name = d_name,
    sigma_obs,
    k,
    sigma_blur,
    sigma_data,
    mu_kind,
    sigma_min,
    sigma_max,
    AT_mode,
    hqs_iters,
    cg_iters,
):
    return (
        int(seed),
        float(sigma_obs),
        int(k),
        float(sigma_blur),
        float(sigma_data),
        str(mu_kind),
        float(sigma_min),
        float(sigma_max),
        str(AT_mode),
        int(hqs_iters),
        int(cg_iters),
    )

@torch.no_grad()
def calibration_k(sigma_max: float, t_target: int, diffusion_instance, device):
  alpha_bar = diffusion_instance.alpha_bar.to(device).float()
  sc = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=0.0))

  sigma_schedule = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=0.0))

  sc_c = float(sc[t_target] / sigma_max)

  return alpha_bar, sc_c, sigma_schedule

class DenoiserWrapper:
  def __init__(self, model, diffusion_instance, opt_norm = None, eps=1e-10, log=True):
    self.model = model
    self.diffusion_instance = diffusion_instance
    self.opt_norm = opt_norm
    self.eps = eps
    self.log = log

  @torch.no_grad()
  def _from_diffusion(self, device):
    return calibration_k(sigma_max=0.02, t_target=23, diffusion_instance=self.diffusion_instance, device=device)

  @torch.no_grad()
  def to_t(self, sigma_v, sigma_schedule, device):
    
    sigma_processed = torch.as_tensor(sigma_v, device=device).float()
    if sigma_processed.ndim == 0:
      sigma_processed = sigma_processed[None]

    dist = (sigma_processed[:, None] - sigma_schedule[None, :]).abs()
    t_indices = dist.argmin(dim=1).long()

    return t_indices

  @torch.no_grad()
  def __call__(self, g, sigma, opt_norm):
    B = g.shape[0]
    device = g.device

    alpha_bar_tensor, sc_c, sigma_schedule = self._from_diffusion(device=device)
    print("sigma_schedule min/max:", sigma_schedule.min().item(), sigma_schedule.max().item())

    sigma = normalize_sigma(sigma, g)

    if torch.is_tensor(sigma):
      sigma_v = sigma.to(device).float()
      if sigma_v.ndim == 0:
        sigma_v = sigma_v.expand(B)
    else:
      sigma_v = torch.full((B,), float(sigma), device = device)

    if opt_norm is not None:
      sigma_v = (sc_c * sigma_v) / (float(self.opt_norm) + self.eps)
    else:
      sigma_v = sc_c * sigma_v

    t = self.to_t(sigma_v, sigma_schedule, device = device)

    if self.log:
      print("g:", tuple(g.shape),
            "sigma_v:", sigma_v,
            "t_indices range:", int(t.min()), "to", int(t.max()),
            "schedule start/end:", float(sigma_schedule[0]), float(sigma_schedule[-1])
      )

    model_out = self.model(g, t)
    
    return model_out



@torch.no_grad()
def eval_uncertainty_metrics(
    x_gt, x_hat_clamp, k, y, A, risk_fn, strong_fn=None, std = True, debug = True,
):
    u_low, u_medium, u_strong = uncertainty_scores(x_hat_clamp, y, A, track, k = k, strong_eps = 0.05, strong_fn = strong_fn)

    risk = risk_fn(x_hat_clamp, x_gt)

    # eval
    in_outs = {
        'u_low': u_low, 'u_medium': u_medium, 'u_strong': u_strong, 'risk': risk,
    }
    
    outs = evaluate_uncertainty(in_outs, std=True, verbose=True)
    

    sanity_low_std = sanity_check(uncertainty_data=u_low, risk=risk, std=std)
    sanity_medium_std = sanity_check(uncertainty_data=u_medium, risk=risk, std=std)
    sanity_strong_std = sanity_check(uncertainty_data=u_strong, risk=risk, std=std)
    
    
    return {
        "aurc_low": float(outs["u_low"]["aurc"]),
        "aurc_medium": float(outs["u_medium"]["aurc"]),
        "aurc_strong": float(outs["u_strong"]["aurc"]),
        "risk_mean": float(risk.float().mean().item()),
        "sanity_low_coverage": float(sanity_low_std["coverage_target"]),
        "sanity_low_target": float(sanity_medium_std["risk_at_target"]),
        "sanity_medium_coverage": float(sanity_strong_std["coverage_target"]),
        "sanity_medium_target": float(sanity_low_std["risk_at_target"]),
        "sanity_strong_coverage": float(sanity_medium_std["coverage_target"]),
        "sanity_strong_target": float(sanity_strong_std["risk_at_target"]),
    }


def eval_ssim(x_gt, x_hat_clamp, y, A, AT, model, sigma, k):
    
    if y is not None:
        mean_mc, samples_mc, track_mc = pnp_mc_dropout(
            x_hat_clamp = x_hat_clamp, y=y, A=A, AT=AT, model=model, sigma = sigma, k = k, debug = True
        )
    else:
       raise ValueError(f"evaluation step requires y data")
    

    
    
    mean_mc_clamp = mean_mc.clamp(0, 1)
    x_gt_clamp = x_gt.clamp(0, 1)

    ssim_det = ssim_per_sample(mean_mc_clamp, x_gt_clamp).clamp(0.0, 1.0)
    risk_det = ssim_per_sample(1.0 -ssim_det).clamp_min(0.0)
    ssim_mc = ssim_per_sample(mean_mc_clamp, x_gt_clamp).clamp(0.0, 1.0)
    risk_mc = (1.0 - ssim_mc).clamp_min(0.0)
    

    return {
        "mean_mc": mean_mc,
        "samples_mc": samples_mc,
        "track_mc": track_mc,
        "ssim_det": ssim_det,
        "ssim_mc": ssim_mc,
        "risk_det": risk_det,
        "risk_mc": risk_mc,
        "ssim_det_mean": float(ssim_det.mean().item()),
        "ssim_mc_mean": float(ssim_mc.mean().item()),
        "risk_det_mean": float(risk_det.mean().item()),
        "risk_mc_mean": float(risk_mc.mean().item()),
        "ssim_gain_mean": float((ssim_mc - ssim_det).mean().item())
    } 

def train_diffusion(
    *, model, diffusion_instance, dataloader, optimizer, scheduler, sigma_min, sigma_max, device, num_epochs: int, verbose = True
):
    
    model.train()
    
    
    if verbose:
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_mean = param.grad.mean().item()
                    grad_max = param.grad.abs().max().item()
                    print(f"{name: <40} | Mean: {grad_mean: .8f} | Max Abs: {grad_max: .8f}")
                else:
                    print(f"{name: <40} | NO GRADIENT (None)")

    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_steps = 0

        for batch in dataloader:
            x0 = _extract_batch_tensor(batch).to(device)

            batch_size = x0.shape[0]

            sigma_batch = sample_sigma(
                kind="mix",
                n=batch_size,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                device=device,
                sigma_center=1e-2,
                rel_width=0.3,
                mix_ratio=0.25,
            )

            t = torch.randint(low = 0, high = diffusion_instance.num_timestpes, size = (x0.shape[0],), device = device).long()


            optimizer.zero_grad(set_to_none = True)
            loss = diffusion_instance.p_losses(model, x0, t = t)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            epoch_loss += float(loss.detach().item())
            n_steps += 1

        lr = optimizer.param_groups[0]["lr"]
        mean_loss = epoch_loss / max(n_steps, 1)
        print(f"[train] epoch = {epoch:03d} loss = {mean_loss:.6f} lr = {lr:.8f}")
    
    model.eval()
    return model

@torch.no_grad()
def run_one(
    *,
    device,
    x_gt,
    A, AT, denoiser_fn,
    sigma_obs:float,
    mu_kind: str,
    hqs_iters:int,
    cg_iters: int,
    sigma_min: float,
    sigma_max: float,
    sigma_data: float | None = None,
):
    y = A(x_gt) + sigma_obs * torch.randn_like(x_gt)

    mu, sigma = make_mu_schedules(kind = mu_kind, iters = cg_iters, sigma_min = sigma_min, sigma_max = sigma_max, device = x_gt.device, dec_sigma = False)
    infer_sample_sigma(mu, sigma)
    
    z0 = AT(y)

    if sigma_data is None:
        simga_data = sigma_obs

    t_0 = time.time()
    x_hat, track = hqs_solve(A = A, AT = AT, y= y, z0 = z0, denoisers = denoiser_fn, mu_schedule = mu, iters = hps_iters, cg_steps = cg_iters, sigma_data = sigma_data, denoiser_kwargs = {"log": True})
    t_hqs = time.time() - t_0

    x_hat_clamp = x_hat.clamp(0, 1)
    g_last = grad2d_rgb(x_hat)
    sigma_last = (1.0 / torch.sqrt(mu_schedule[-1])).item()
    opt_norm = estimate_optimal_norm(A, AT, x_gt, device, iters = hqs_iters)
    z_last = denoiser_fn["blur"](g_last, sigma_last, opt_norm)
    
    # assert_shapes (x_hat, g_last, z_last)
    if g_last.ndim != 4:
        raise RuntimeError(f"g must be [B,C,H,W], got {tuple(g_last.shape)}")
    if z_last.shape != g_last.shape:
        raise RuntimeError(f"denoiser output must match,  g={tuple(g_last.shape)} z={tuple(z_last.shape)}")
    if x_hat.ndim != 4:
        raise RuntimeError(f"x_hat must be [B,3,H,W], got {tuple(x_hat.shape)}")
    
    out = {
        "track": track,
        "g_last": g_last,
        "z_last": z_last,
        "sigma_last": sigma_last,
        "mu_schedule": mu_schedule,
        "x_hat_clamp": x_hat_clamp,
        "opt_norm": float(opt_norm),
        "finite": int(torch.isfinite(x_hat).all().item()),
        "time": float(t_hqs),
        "mse_x": mse(x_hat_clamp, x_gt),
        "psnr": psnr(x_hat_clamp, x_gt),
        "mse_Ax_y": mse(A(x_hat_clamp), y),
        "x_hat_mean": float(x_hat_clamp.mean().item()),
        "x_hat_min": float(x_hat_clamp.min().item()),
        "x_hat_max": float(x_hat_clamp.max().item()),
    }

    


    return out

# -----------------------------
# grid runner
# -----------------------------
@dataclass
class GridConfig:
    num_epochs_list: Tuple[int, ...] = (2, 3, )
    seeds: Tuple[int, ...] = (0, 1, 2)
    sigma_obs_list: Tuple[float, ...]=(0.01, 0.03, 0.05)
    sigma_blur_list: Tuple[float, ...] = (0.0, 0.7, 1.2)
    sigma_data_list: Tuple[float, ...] = (0.01, 0.03, 0.05)
    sigma_min_list: Tuple[float, ...] = (5e-4 , 1e-3, 2e-3)
    sigma_max_list: Tuple[float, ...] = (2e-2 , 5e-2, 1e-1)
    mu_kinds: Tuple[str, ...] = ("cosine", "linear", "geometric")
    hqs_iters_list : Tuple[int, ...] = (5, 10, 20)
    cg_iters_list : Tuple[int, ...] = (5, 10, 20)
    k_list : Tuple[int, ...] = (3, 5, 8)
    AT_modes: Tuple[str, ...] = ('blur', 'inverse')
    max_batches: int = 1
    strong_eps: float = 0.05
    std_aurc = True
    use_mc_risk = True


def grid_search(
    *,
    out_csv: str,
    out_pt: str,
    device,
    train_dataloader,
    val_dataloader,
    model,
    diffusion_instance,
    optimizer,
    scheduler,
    denoiser_fn,
    grid: GridConfig,
):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_pt) or ".", exist_ok=True)

    fieldnames = [
        "seed", "sigma_obs", "k","sigma_blur", "sigma_data", "mu_kind", "AT_mode", "hqs_iters", "cg_iters", "finite", "time", "mse_x", "psnr", "mse_Ax_y", "aurc_low", "aurc_medium", "aurc_strong", "risk_mean", "batch", "sigma_last", "mu_schedule", "opt_norm", "track", "sanity_low_coverage", "sanity_low_target", "sanity_medium_coverage", "sanity_medium_target", "sanity_strong_coverage", "sanity_strong_target", "mean_mc", "samples_mc", "track_mc", "ssim_det", "ssim_mc", "risk_det", "risk_mc", "ssim_det_mean", "ssim_mc_mean", "risk_det_mean", "risk_mc_mean", "ssim_gain_mean", "sigma_min_list", "sigma_max_list"
    ]
    
    
    set_params = itertools.product(
        grid.sigma_min_list,
        grid.sigma_max_list,
        grid.num_epochs_list,
    )

    print(grid.num_epochs_list)
    print([type(x) for x in grid.num_epochs_list])

    print(f"\n [train] train diffusion")

    for sigma_min, sigma_max, num_epochs in tqdm(list(set_params), desc = f"train sets"):        
            
            # train
            train_diffusion(
                model = model, diffusion_instance = diffusion_instance, dataloader = train_dataloader, optimizer = optimizer, scheduler = scheduler, sigma_min = sigma_min, sigma_max = sigma_max, device = device, num_epochs = num_epochs, verbose = True
            )
            
            val_batch = next(iter(val_dataloader))    
            
            x_gt = _extract_batch_tensor(val_batch)
            x_gt = x_gt.to(device)
            
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            
                for seed in grid.seeds:
                    set_seed(seed)

                    print(f"\n[Seed: {seed}] Training HQS optimizations")
                    # ----------------- run_one -------------------------
                    run_params = itertools.product(
                        grid.sigma_obs_list,
                        grid.k_list,
                        grid.sigma_blur_list,
                        grid.sigma_data_list,
                        grid.sigma_min_list,
                        grid.sigma_max_list,
                        grid.mu_kinds,
                        grid.AT_modes,
                        grid.hqs_iters_list,
                        grid.cg_iters_list,
                    )

                    total = (
                        len(grid.sigma_obs_list)
                        * len(grid.k_list)
                        * len(grid.sigma_blur_list)
                        * len(grid.sigma_data_list)
                        * len(grid.sigma_min_list)
                        * len(grid.sigma_max_list)
                        * len(grid.mu_kinds)
                        * len(grid.AT_modes)
                        * len(grid.hqs_iters_list)
                        * len(grid.cg_iters_list)
                    )

                    for (sigma_obs, k, sigma_blur, sigma_data, mu_kind, AT_mode, hqs_iters, cg_iters) in tqdm(run_params, total = total, desc = f"run setting {seed}"):
                    
                        
                        

                        cfg_key = get_cfg_key(
                            seed = seed,
                            # denoiser_name = d_name,
                            sigma_obs = sigma_obs,
                            k = k,
                            sigma_blur = sigma_blur,
                            sigma_data = sigma_data,
                            mu_kind = mu_kind,
                            sigma_min = sigma_min,
                            sigma_max = sigma_max,
                            AT_mode = AT_mode,
                            hqs_iters = hqs_iters,
                            cg_iters = cg_iters
                        )
                        
                        A, AT = make_A_AT(k=k, sigma_blur=sigma_blur, AT_mode=AT_mode)
                        
                        start = time.perf_counter()
                        
                        t_metrics =  run_one(
                            device = device,
                            x_gt = x_gt,
                            A = A,
                            AT = AT,
                            denoiser_fn = denoiser_fn,
                            sigma_obs = sigma_obs,
                            mu_kind = mu_kind,
                            sigma_min = sigma_min,
                            sigma_max = sigma_max,
                            # sigma_blur = sigma_blur,
                            hqs_iters = hqs_iters,
                            cg_iters = cg_iters,
                            sigma_data = sigma_data,
                        )
                                
                        elapsed = time.perf_counter() - start


                        required = ["x_hat_clamp", "y", "track"]
                        missing = [k_ for k_ in required if k_ not in t_metrics]

                        if missing:
                            print(f"missing key: {missing}")
                            # raise KeyError(f"missing key: {missing}")
                    
                        # ------------- Caching scalar values -----------------

                        r_metrics =  eval_uncertainty_metrics(
                                x_gt = x_gt,
                                x_hat_clamp = t_metrics["x_hat_clamp"],
                                k = k,
                                y = t_metrics["y"],
                                A = A,
                                track = t_metrics["track"],
                                cg_iters = cg_iters,
                                strong_eps = grid.strong_eps,
                                std = grid.std_aurc,
                                debug = True
                            )

                        s_metrics = eval_ssim(
                                x_gt = x_gt,
                                x_hat_clamp = t_metrics["x_hat_clamp"],
                                y = t_metrics["y"],
                                A = A,
                                AT = AT,
                                model = model,
                                sigma = sigma_data,
                                k = k,
                            )

                        pt_f = {
                                "seed": int(t["seed"]),
                                "config": {
                                    "sigma_obs": sigma_obs,
                                    "k": k,
                                    "mu_kind": str(mu_kind),
                                    "sigma_blur": float(sigma_blur),
                                    "sigma_min": float(sigma_min),
                                    "sigma_max": float(sigma_max),
                                    "AT_mode": str(AT_mode),
                                    "hqs_iters": int(hqs_iters),
                                    "cg_iters": int(cg_iters),
                                },
                                "t_metrics": t_metrics,
                                "r_metrics": r_metrics,
                                "s_metrics": s_metrics,
                        }
                        
                        torch.save(
                            pt_f,
                            out_pt
                        )
                            
                        row = {
                                "seed": int(t_metrics["seed"]),
                                "sigma_obs": sigma_obs,
                                "k": k,
                                "mu_kind": str(mu_kind),
                                "sigma_blur": sigma_blur,
                                "sigma_min": sigma_min,
                                "sigma_max": sigma_max,
                                "AT_mode": AT_mode,
                                "hqs_iters": int(hqs_iters),
                                "cg_iters": int(cg_iters),
                                "batch" : int(batch_idx),
                                "sigma_last": float(t_metrics["sigma_last"]),
                                "mu_schedule": t_metrics["mu_schedule"],
                                "opt_norm": float(t_metrics["opt_norm"]),
                                "finite": int(t_metrics["finite"]),
                                "time": float(elapsed),
                                "mse_x": float(t_metrics["mse_x"]),
                                "psnr": float(t_metrics["psnr"]),
                                "mse_Ax_y": float(t_metrics["mse_Ax_y"]),
                                "aurc_low": float(r_metrics["aurc_low"]),
                                "aurc_medium": float(r_metrics["aurc_medium"]),
                                "aurc_strong": float(r_metrics["aurc_strong"]),
                                "risk_mean": float(r_metrics["risk_mean"].float().mean().item()),
                                "sanity_low_coverage": float(r_metrics["sanity_low_coverage"]),
                                "sanity_low_target": float(r_metrics["sanity_low_target"]),
                                "sanity_medium_coverage": float(r_metrics["sanity_medium_coverage"]),
                                "sanity_medium_target": float(r_metrics["sanity_medium_target"]),
                                "sanity_strong_coverage": float(r_metrics["sanity_strong_coverage"]),
                                "sanity_strong_target": float(r_metrics["sanity_strong_target"]),
                                "ssim_det_mean": float(s_metrics["ssim_det_mean"]),
                                "ssim_mc_mean": float(s_metrics["ssim_mc_mean"]),
                                "risk_det_mean": float(s_metrics["risk_det_mean"]),
                                "risk_mc_mean": float(s_metrics["risk_mc_mean"]),
                                "ssim_gain_mean": float(s_metrics["ssim_gain_mean"]),
                                "artifact_path": out_pt,
                        }

                        writer.writerow(row)
                        f.flush()
                        print(f'row expanded: {row}')

                        if "x_hat_clamp" in t_metrics:
                            del t_metrics["x_hat_clamp"]
                        if "y" in t_metrics:
                            del t_metrics["y"]
                            
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(f"\n[Seed: {seed}] done HQS optimizations")


def train_anp_pnp():
    
    print("Starting ANP-PNP Tests...")

    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("available:", torch.cuda.is_available())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ 
    # y=A(x)+noise
    # PCam synthetic inverse. 
    A, AT = make_A_AT(mode='blur', k=5, sigma_blur=1.2)
    """
    
    train_dataloader, val_dataloader = build_pcam_loader(
        data_root=PCAM_ROOT,
        img_size=224,
        batch_size=64,
        num_workers=2,
        download=False,
    )

    show_batch(train_dataloader)



    diffusion_instance = build_diffusion_instance(device = device, cfg = DiffusionConfig)

    # g: [B,2,H,W] for grayscale or [B,6,H,W] for RGB.
    # DiT in_channels == out_channels == g_channels
    dit_config = DiT2DConfig(
        in_channels=6,  
        out_channels=6,  # DiT model outputs noise of same shape
        img_size=32,  # Spatial dimensions of the gradients
        patch=4,  
        embed_dim=256,  
        depth=6,  
        num_heads=4,  
        mlp_ratio=3.0,  
        dropout=0.0,  
        time_embed_dim=256  
    )
    try:
        build_model = make_dit_builder(DiT2D, dit_config)
        model, cfg = build_model(patch=4)
        print(f"model cfg: {cfg}")
        model = model.to(device)

        denoiser_wrapper_instance = DenoiserWrapper(model, diffusion_instance, opt_norm = None, log = True)
    except Exception as e:
        print(f'dit: {e}')
        tr=traceback.print_exc()
        if tr:
            print(tr)
        

    # Denoiser function G_denoiser for anp_pnp_hqs.
    # It receives complex gradients `g_complex` and a noise level `sigma`.
    # It should return denoised complex gradients.
    def build_denoiser(denoiser_wrapper_instance):
        def G_denoiser(g, sigma, opt, log = True, **kwargs):

            if log:
                print(f"g = {g.shape}, sigma = {sigma.shape}")
            # Run the wrapper
            denoiser_wrapper_instance.log = log

            return denoiser_wrapper_instance(g, sigma)

        return {
            'blur': G_denoiser,
        }
    
    denoisers = build_denoiser(denoiser_wrapper_instance)

    # optimizer = optim.AdamW(model.parameters(), lr = 1e-4, weight_decay=1e-2)
    optimizer = Prodigy(model.parameters(), lr = 1e-4, weight_decay = 1e-2, betas = (0.9, 0.999), safeguard_warmup = True, use_bias_correction = True)
    
    for i, g in enumerate(optimizer.param_groups):
            print(f"group {i} param_count =", len(g["params"]))
            print(f"group {i} lr = {g['lr']}")
            print(f"group{i} actual learing rate = {(g['lr']) * (g.get('d', 1.0))}")

    scheduler = get_cosine_schedule_with_warmup(optimizer = optimizer, num_warmup_steps = 25, num_training_steps = 50)
        
    # Test HQS
    try:
        print("Testing anp_pnp_hqs...")
        grid_search(
            out_csv = "./results/csv/ablation_pcam_hqs.csv",
            out_pt = "./results/metrics.pt",
            device = device,
            train_dataloader = train_dataloader,
            val_dataloader = val_dataloader,
            model = model,
            diffusion_instance = diffusion_instance,
            optimizer = optimizer,
            scheduler = scheduler,
            denoiser_fn = denoisers,
            grid = GridConfig,
        )
        print("Testing is done")
        
    except Exception as e:
        print(f"HQS Failed: {e}")
        
        tr = traceback.print_exc()
        print(tr)



if __name__ == "__main__":
    train_anp_pnp()

