import torch
from DiT_ProbabilisticAlgotithm.models.diffusion import GaussianDiffusion
from DiT_ProbabilisticAlgotithm.pnp.cg import cg_solve


from DiT_ProbabilisticAlgotithm.pnp.grad import grad2d_rgb, div2d_rgb

from typing import Any, Dict, List, Optional, Tuple

@torch.no_grad()
def hqs_solve(
        A, AT, y, z0, denoisers, mu_schedule, opt_norm, iters, cg_steps, sigma_data, AT_extra=None, out_ch=None, denoiser_kwargs: Optional[Dict[str, Any]] = None, return_track: bool = True, track_every:int = 1, track_full_state: bool = False, track_on_cpu: bool = True, return_history = True
):
    """
    :param A: linear operator
    :param AT: linear operator
    :param y: measurement
    :param z0: initial estimate
    :track: list[Dict], optional
    """
    if denoiser_kwargs is None:
        denoiser_kwargs = {}
    
    if z0 is None:
        if AT_extra is None:
            x = AT(y)
        else:
            x = AT_extra(y, out_ch)
    else:
        x = z0.clone()

    track: List[Dict[str, Any]] = []
    
    
    if mu_schedule is None:
        mu_schedule = torch.logspace(-1, 2, steps=cg_steps, device=y.device)

    if mu_schedule.numel() != iters:
        raise ValueError(f"mu schedule length must be equivalent to iters({iters}), got {mu_schedule.numel()}")

    def apply_AT(inp: torch.Tensor) -> torch.Tensor:
        if AT_extra is None:
            return AT(inp)
        return AT_extra(inp, out_ch)
    
    def ATA(v:torch.Tensor) -> torch.Tensor:
        if AT_extra is None:
            return AT(A(v))
        return apply_AT(A(v))

    def is_store(t: torch.Tensor):
        t = t.detach()
        return t.cpu() if track_on_cpu else t
    
    ATy = apply_AT(y)

    for k in range(iters):

        mu = mu_schedule[k]

        if mu <= 0.0 or not isinstance(mu.item(), float):
            raise ValueError(f"mu must be a float and positive value, {mu}")

        
        x_prev = x.detach().clone()
        
        # z-step
        g = grad2d_rgb(x)
        sigma_k = float((1.0 / torch.sqrt(mu)))
        z = denoisers["blur"](g, sigma_k, opt_norm = opt_norm, **denoiser_kwargs)

        def linop(v):
            # μdiv(∇v)
            return (ATA(v) / (sigma_data **2)) + (mu * div2d_rgb(grad2d_rgb(v)))
        # μdiv(z)
        rhs = ATy / (sigma_data **2) + mu * div2d_rgb(z)
        # g = grad(x) with shape [B, 6, H, W]
        x = cg_solve(linop, rhs, x0=x, iters=cg_steps)

        dx = x - x_prev

        if return_track and (k % track_every == 0 or k == iters -1):
            entry = {
                "iter": k,
                "mu": float(mu.item()),
                "x_mean": float(x.mean().item()),
                "x_min": float(x.min().item()),
                "x_max": float(x.max().item()),
                "g_norm": float(g.norm().item()),
                "z_norm": float(z.norm().item()),
                "dx_norm": float(dx.norm().item()),
                "sigma_k": float(sigma_k.item()),
                "data_residual_norm": float((A(x) - y).norm().item()),
            }

            if track_full_state:
                entry["x"] = is_store(x)
                entry["g"] = is_store(g)
                entry["z"] = is_store(z)

            track.append(entry)

    g = grad2d_rgb(x)
    if g.shape[0] != x.shape[0]:
        raise ValueError(f"g shape and x shape mismatch: g.shape = {g.shape} | x.shape = {x.shape}")
        
    if return_track:
        return x, track

    # denoised gradient field of the same shape
    return x
