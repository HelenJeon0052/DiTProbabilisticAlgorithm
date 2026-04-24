import torch
import torch.nn.functional as F


import torch.nn as nn

from DiT_ProbabilisticAlgotithm.ops.blur import synth_blur_noise, psnr
from DiT_ProbabilisticAlgotithm.pnp.grad import grad2d_rgb
from DiT_ProbabilisticAlgotithm.pnp.pnpstarter import hqs_solve
from DiT_ProbabilisticAlgotithm.eval.mc_dropout import enable_mc_dropout
from DiT_ProbabilisticAlgotithm.eval.coverage_risk import coverage_risk, coverage_risk_std, uncertainty_scores
from DiT_ProbabilisticAlgotithm.utils.utils import make_mu_schedule

def sigma_to_batch(sigma, g):
    b = g.shape[0]
    """
    if isinstance(sigma, (float, int)):
        return torch.full((b,), float(sigma), device=g.device, dtype=g.dtype)
    """
    sigma = torch.as_tensor(sigma, device=g.device, dtype=g.dtype)
    if sigma.ndim == 0:
        sigma = sigma.expand(b)
    if sigma.ndim == 1 and sigma.shape[0] == 1:
        sigma = sigma.expand(b)
    return sigma


def pnp_mc_dropout(x_hat_clamp, y, A, AT, model, sigma: float, k: int, clamp_fn = None, dc_weight: float = 1.0, debug: bool = True):

    enable_mc_dropout(model)

    xs = []


    for _ in range(k):
        residual = A(x_hat_clamp) - y
        x_hat_out = x_hat_clamp - dc_weight * AT(residual)

        xs.append(x_hat_clamp)

    xs = torch.stack(xs, dim=0)
    xs_mean = xs.mean(dim=0)
    xs_var = xs.var(dim=0, unbiased=False)



    return xs_mean, xs, xs_var

@torch.no_grad()
def ssim_per_sample(x: torch.Tensor, y: torch.Tensor, win_size = 11, sigma = 1.5, data_range=1.0, K1=0.01, K2=0.03, eps=1e-10):
    """
     - x, y: [B, C, L, W]
     return: [B]
    """
    if x.shape != y.shape:
        raise ValueError(f'x, y shape mismatch')
    if x.ndim != 4:
        raise ValueError(f'expected [b, C, L, W], | but got {tuple(x.shape)}')
    
    B, C, L, W = x.shape
    device, dtype = x.device, x.dtype
    
    padding = win_size // 2
    coords = torch.arange(win_size, device=device, dtype=dtype) - (win_size - 1) / 2.0
    gr = torch.exp(-(coords**2) / (2*sigma**2))
    gr_0 = gr / gr.sum()
    w = (gr_0[:, None] * gr_0[None, :]).unsqueeze(0).unsqueeze(0)
    print(f'w:{w.shape}')
    
    w = w.repeat(C, 1, 1, 1)
    
    def filt(z):
        return F.conv2d(F.pad(z, (padding, padding, padding, padding), mode='reflect'), w, groups=C)
    
    mu_x = filt(x)
    mu_y = filt(y)
    mu_x2 = (mu_x) **2
    mu_y2 = (mu_y) **2
    mu_xy = mu_x * mu_y
    
    sig_x2 = filt(x*x) - mu_x2
    sig_y2 = filt(y*y) - mu_y2
    sig_xy = filt(x*y) - mu_xy
    
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    
    num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
    ssim_map = num / (den+eps)
    
    print(f'ssim_map: {ssim_map.mean(dim=(2, 3)).mean(dim=1)}')
    
    return ssim_map.mean(dim=(2, 3)).mean(dim=1)


    
    









# coverage-risk
def expected_aurc_first(risk: torch.Tensor) -> float:
    rk = risk.detach().float().cpu()
    N = rk.numel()
    if N == 0:
        return 0.0
    rbar = rk.mean().item()
    print(f'rbar: {rbar}')
    return rbar * (1.0 - 1.0 / max(N, 1))

def expected_aurc_random_std(risk: torch.Tensor) -> float:
    rk = risk.detach().float().cpu()
    if rk.numel() == 0:
        return 0.0
    print(f'rk.numel(): {rk.numel()}')
    return rk.mean().item()


def aurc_with_correction(uncertainty: torch.Tensor, risk: torch.Tensor, *, std: bool = True, verbose:bool = False):
    """
     - AURC
     - E[AURC_random]
     - EAURC = AURC - E[AURC_random]
     - nAURC = EAURC / E[AURC_random]
    """
    
    if std:
        cov, cr, aurc = coverage_risk_std(uncertainty, risk, verbose=verbose)
        e_aurc = expected_aurc_random_std(risk)
    else:
        cov, cr, aurc = coverage_risk(uncertainty, risk)
        e_aurc = expected_aurc_first(risk)
        
    eaurc_0 = aurc - e_aurc
    naurc_0 = eaurc_0 / (e_aurc + 1e-10)
    
    out = {
        'coverage':cov,
        'cum_risk': cr,
        'aurc': float(aurc),
        'e_aurc_random': float(e_aurc),
        'eaurc': float(eaurc_0),
        'naurc': float(naurc_0),
        'std': bool(std),
    }
    
    if verbose:
        print(f'[std={std}] | AURC={aurc:.6g} | E_rand={e_aurc:.6g} | EAURC={eaurc_0:.6g} | NAURC={naurc_0:.6g}')
    
    return out

def sanity_check(uncertainty: torch.Tensor, risk: torch.Tensor, *, std: bool= True, target_coverage: float = 0.15):
    
    uncertainty = torch.as_tensor(uncertainty).reshape(-1).float()
    risk = torch.as_tensor(risk).reshape(-1).float()
    
    if uncertainty.numel() != risk.numel():
        raise ValueError(f"mismatch size: uncertainty = {uncertainty.numel()}, risk = {risk.numel()}")
    if not torch.isfinite(uncertainty).all():
        raise ValueError(f"uncertainty having non-finite value")
    if not torch.isfinite(risk).all():
        raise ValueError(f"risk having non-finite value")
    
    out = aurc_with_correction(uncertainty, risk, std=std, verbose=False)
    
    cr = out["cum_risk"].detach().clone().to(torch.float32) # torch.tensor(out['cum_risk'], dtype=torch.float32)
    coverage = out["coverage"].detach().clone().to(torch.float32)
    
    if std:
        idx = int(torch.argmin((coverage - target_coverage).abs()).item())
    else:
        idx = max(0, int(target_coverage * (len(coverage) - 1)))
        
    summary = {
        "risk_full": float(cr[-1].item()),
        "risk_at_target": float(cr[idx].item()),
        "coverage_target": float(coverage[idx].item()),
        "improves": bool(cr[idx] <= cr[-1]),
    }
    
    print(f'[std={std}] | risk@full={cr[-1].item():.6g} | risk@~15%={cr[idx].item():.6g}')
    print(f'compare risk@full={cr[-1].item():.6g} and risk@~15%={cr[idx].item():.6g}')
    
    return summary    


def evaluate_uncertainty(res:dict, *, std: bool = True, verbose: bool = True):
    """
    expect : res['u_low'], res['u_medium'], res['u_strong']
    returns : dict with per-score curves + metrics
    """
    
    indicators = ["u_low", "u_medium", "u_strong", "risk"]
    missing = [k for k in indicators if k not in res]
    if missing:
        raise KeyError(f"missing keys in res: {missing}")
    
    risk = torch.as_tensor(res["risk"]).reshape(-1).float()
    if not torch.isfinite(risk).all():
        raise ValueError(f"risk having non-finite value")
        
    
    outs = {}
    
    for key in ["u_low", "u_medium", "u_strong"]:
        score = torch.as_tensor(res[key]).reshape(-1).float()

        if score.numel() != risk.numel():
            raise ValueError(f"mismatch of score and risk, score = {score.numel()} | risk = {risk.numel()}")
        if not torch.isfinite(score).all():
            raise ValueError(f"score having non-finite value")
        
        outs[key] = aurc_with_correction(score, risk, std=std, verbose=False)
        if verbose:
            print(
                f"{key:8s} | AURC={outs[key]['aurc']:.6g}"
                f"E_rand= {outs[key]['e_aurc_random']:.6g}"
                f"EAURC={outs[key]['eaurc']:.6g}, nAURC = {outs[key]['naurc']:.6g}"
            )
    
    if verbose:
        rank_aurc = sorted([(k, v['aurc']) for k, v in outs.items()], key=lambda x:x[1])
        rank_eaurc = sorted([(k, v['eaurc']) for k, v in outs.items()], key=lambda x:x[1])
        print('AURC ranking (lower):', rank_aurc)
        print('EAURC ranking (lower, 0=random):', rank_eaurc)
        
    return outs