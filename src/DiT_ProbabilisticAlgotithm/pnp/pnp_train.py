import torch
import torch.nn.functional as F

from DiT_ProbabilisticAlgotithm.ops.blur import synth_blur_noise, psnr
from DiT_ProbabilisticAlgotithm.pnp.grad import grad2d_rgb
from DiT_ProbabilisticAlgotithm.pnp.pnpstarter import hqs_solve, normalize_sigma
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

def pnp_mc_dropout(y, A, AT, model, K=8, debug=False):

    enable_mc_dropout(model)

    xs = []

    for _ in range(K):
        def G_denoiser(g, sigma):
            sigma = sigma_to_batch(sigma, g)
            sigma = normalize_sigma(sigma, g)

            return model(g, sigma)

        # Test HQS
        try:
            print("Testing anp_pnp_hqs...")
            x_hat, _track = hqs_solve(y, A, AT, G_denoiser, iters=5, cg_iters=5)
            print(f"HQS Success! Output shape: {x_hat.shape}")
            print(
                    f'finite:{torch.isfinite(x_hat).all().item()}, mean: {x_hat.abs().mean().item()}, min/max: {x_hat.min().item()}/{x_hat.max().item()}')
        except Exception as e:
            if debug:  
                import traceback
                traceback.print_exc()
            x_hat = torch.full_like(y, float('nan'))
            print(f"HQS Failed: {e}")

        xs.append(x_hat)

    xs = torch.stack(xs, dim=0)
    mean = xs.mean(0)
    var = xs.var(0)

    u_strong = var.mean(dim=(1, 2, 3)) # [B]



    return mean, u_strong, xs

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
    
def assert_shapes(x_hat, g, z):
    if g.ndim != 4:
        raise RuntimeError(f"g must be [B,C,H,W], got {tuple(g.shape)}")
    if z.shape != g.shape:
        raise RuntimeError(f"denoiser output z must match g. g={tuple(g.shape)} z={tuple(z.shape)}")
    if x_hat.ndim != 4:
        raise RuntimeError(f"x_hat must be [B,3,H,W], got {tuple(x_hat.shape)}")
    
@torch.no_grad()
def eval_on_setting(
        x,
        blur_sigma,
        noise_sigma,
        model,
        iters,
        mu_kind,
        risk_fn,
        track,
        strong_K=5,
        debug = True
):

    y, blur = synth_blur_noise(x, blur_sigma=blur_sigma, noise_sigma=noise_sigma)
    A, AT = blur.A, blur.AT
    if debug:
        print(f'blur: {blur.A} | {blur.AT}')

    mu = make_mu_schedule(mu_kind, iters, x.device)


    # low / medium from single deterministic/ema run
    model.eval()

    def G_denoiser(g, sigma):
        sigma = sigma_to_batch(sigma, g)
        sigma = normalize_sigma(sigma, g)

        return model(g, sigma)

    out = hqs_solve(
        y, A, AT, G_denoiser, iters=iters, sigma_data=max(noise_sigma, 1e-6), mu_schedule=mu, cg_iters=20
    )
    
    if isinstance(out, tuple):
        x_hat, track = out
        fail_shape = None
        fail_shape = x_hat.shape
        
        if fail_shape is None:
            print(f'fail_shape: {fail_shape}')
            raise ValueError('solver shape mismatch')
    else:
        x_hat, track = out, None
    
    x_hat = x_hat.clamp(0, 1)
    print("x_hat:", tuple(x_hat.shape))

    g_last = grad2d_rgb(x_hat)
    sigma_last = (1.0 / torch.sqrt(mu[-1])).item()
    z_last = G_denoiser(g_last, sigma_last)
    assert_shapes(x_hat, g_last, z_last)
    # assert z_last.shape == g_last.shape, f'z_last.shape == g_last.shape'
    
    u_low, u_medium, u_strong = uncertainty_scores(x_hat, y, A, g_last, z_last, track, strong_K = strong_K, strong_eps = 0.05, strong_fn=None)
    
    ssim_det = ssim_per_sample(x_hat, x).clamp(0.0, 1.0)
    
    """
    mean_mc, u_strong, xs = pnp_mc_dropout(
        y=y, blur=blur, model=model, mu_schedule=mu, iters=iters, sigma_data=max(noise_sigma, 1e-6), cg_iters=20, K=K_strong
    )

    risk_mc = (1.0 - ssim_per_sample(mean_mc.clamp(0, 1), x)).clamp_min(0.0)
    ssim_mc = ssim_per_sample(mean_mc.clamp(0, 1), x).clamp(0.0, 1.0)"""
    
    if risk_fn is None:
        risk = (1.0 - ssim_det).clamp_min(0.0)
    else:
        risk=risk_fn(x_hat, x)

    return {
        'risk': risk,
        'u_low': u_low, 'u_medium': u_medium, 'u_strong': u_strong,
        'psnr': psnr(x_hat, x), 'ssim': ssim_det.mean().item(),
    }

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
        'com_risk': cr,
        'aurc': float(aurc),
        'e_aurc_random': float(e_aurc),
        'eaurc': float(eaurc_0),
        'naurc': float(naurc_0),
        std: bool(std),
    }
    
    if verbose:
        print(f'[std={std}] | AURC={aurc:.6g} | E_rand={e_aurc:.6g} | EAURC={eaurc_0:.6g} | NAURC={naurc_0:.6g}')
    
    return out

def sanity_check(uncertainty: torch.Tensor, risk: torch.Tensor, *, std: bool= True):
    out = aurc_with_correction(uncertainty, risk, std=std, verbose=False)
    
    cr = torch.tensor(out['cum_risk'])
    cov = torch.tensor(out['coverage'])
    
    if std:
        target = .15
        idx = int(torch.argmin((cov - target).abs()).item())
    else:
        idx = max(0, int(0.1 * (len(cov) - 1)))
        
    print(f'[std={std}] | risk@full={cr[-1].item():.6g} | risk@~15%={cr[idx].item():.6g}')
    print(f'compare risk@full={cr[-1].item():.6g} and risk@~15%={cr[idx].item():.6g}')
    
    
def evaluate_uncertainty(res:dict, *, std: bool = True, verbose: bool = True):
    """
    expect : res['u_low'], res['u_medium'], res['u_strong']
    returns : dict with per-score curves + metrics
    """
    
    risk = res['risk']
    outs = {}
    
    for key in ['u_low', 'u_medium', 'u_strong']:
        outs[key] = aurc_with_correction(res[key], risk, std=std, verbose=False)
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