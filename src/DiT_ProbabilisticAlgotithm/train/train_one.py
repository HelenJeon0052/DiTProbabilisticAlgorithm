import os, csv, time, math, random
import numpy as np



import torch
from torch.utils.data import DataLoader
import traceback

from typing import List, Dict, Tuple
from dataclasses import dataclass

from torchvision import transforms

from DiT_ProbabilisticAlgotithm.models.dit import DiT2D, DiT2DConfig
from DiT_ProbabilisticAlgotithm.utils.make_dit_builder import make_dit_builder
from DiT_ProbabilisticAlgotithm.pnp.pnpstarter import hqs_solve, normalize_sigma
from DiT_ProbabilisticAlgotithm.pnp.pnp_train import evaluate_uncertainty, sanity_check, eval_on_setting, sigma_to_batch
from DiT_ProbabilisticAlgotithm.pnp.grad import grad2d_rgb
from DiT_ProbabilisticAlgotithm.data.pcam_starter import show_batch, PCamDataset
from DiT_ProbabilisticAlgotithm.ops.blur import make_gaussian_kernel, blur2d, mse, psnr
from DiT_ProbabilisticAlgotithm.eval.coverage_risk import uncertainty_scores
from DiT_ProbabilisticAlgotithm.utils.utils import set_seed



def make_A_AT(AT_mode, k, sigma_blur):
    # Operators A=I
    A = lambda x: blur2d(x, k=k, sigma=sigma_blur) if 'k' in blur2d.__code__.co_varnames else blur2d(x)
    
    kernel = make_gaussian_kernel(k=5, sigma=1.2, dtype=torch.float32)
    kernel_flip = torch.flip(kernel, dims=[-1, -2])


    if AT_mode == 'blur':
        AT = lambda y: blur2d(y, kernel) if 'k' in blur2d.__code__.co_varnames else blur2d(y)
    elif AT_mode == 'inverse':
        AT = lambda y: blur2d(y, kernel_flip)
    else:
        raise ValueError(f'Unknown AT_mode: {AT_mode}')
    
    return A, AT
        

# -----------------------------
# runners
# -----------------------------
@torch.no_grad()
def run_one(
    device,
    x_gt,
    A, AT, denoiser_fn,
    sigma_obs:float,
    hqs_iters:int,
    cg_iters: int,
):
    y = A(x_gt) + sigma_obs * torch.randn_like(x_gt)

    
    t_0 = time.time()
    out = hqs_solve(y, A, AT, denoiser_fn, iters=hqs_iters, cg_iters=cg_iters)
    if isinstance(out, tuple):
        x_hat, track = out
    else:
        x_hat, track = out, None
    t_hqs = time.time() - t_0

    x_hat_clamp = x_hat.clamp(0, 1)

    out = {
        "finite": int(torch.isfinite(x_hat).all().item()),
        "time": t_hqs,
        "x_hat": x_hat,
        "mse_x": mse(x_hat_clamp, x_gt),
        "psnr": psnr(x_hat_clamp, x_gt),
        "mse_Ax_y": mse(A(x_hat_clamp), y),
        "x_hat_mean": x_hat_clamp.mean().item(),
        "x_hat_min": x_hat_clamp.min().item(),
        "x_hat_max": x_hat_clamp.max().item(),
    }

    return out

def _append_row_csv(path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.exists(path)
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    

@dataclass
class GridConfig:
    blur_sigmas: Tuple[float, ...] = (0.0, 0.7, 1.2)
    noise_sigma: Tuple[float, ...] = (0.0, 0.01, 0.03)
    mu_kinds = ('cosine',)
    iters_list : Tuple[int, ...] = (10,)
    cg_iters_list : Tuple[int, ...] = (20,)
    strong_K_list : Tuple[int, ...] = (8,)
    AT_modes=('blur', 'inverse'),
    max_batches: int = 10
    std_aurc = True,
    use_mc_risk = True

# -----------------------------
# grid runner
# -----------------------------
def grid_search(
    *,
    out_csv,
    device,
    dataloader,
    denoiser_bank: dict,
    seeds=(0, 1, 2),
    grid = GridConfig()
):

    fieldnames = [
        "seed","denoiser","blur_sigma","noise_sigma","mu_kind","iters","cg_iters","num_samples","batch","strong_K","AT_mode","psnr_det_mean","ssim_det_mean","aurc_low","aurc_medium","aurc_strong","risk_mean","time"
    ]
        
    for seed in seeds:
        set_seed(seed)
        x_gt, label = next(iter(dataloader))
        x_gt = x_gt.to(device)

        for d_name, d_fn, model_builders in denoiser_bank.items():
            model = model_builders().to(device)
            for blur_sigma in grid.blur_sigmas:
                for noise_sigma in grid.noise_sigmas:
                        for mu_kind in grid.mu_kinds:
                            for iters in grid.iters_list:
                                for AT_mode in grid.AT_modes:
                                    A, AT = make_A_AT(k=5, sigma_blur=blur_sigma, AT_mode=AT_mode)
                                    for cg_iters in grid.cg_iters_list:
                                        for strong_K in grid.strong_K_list:
                                            t_0 = time.time()
                                            
                                            u_low_all, u_medium_all, u_strong_all, risk_all = [], [], [], []
                                            psnr_det_list, ssim_det_list = [], []
                                            
                                            n_samples = 0
                                            for bidx, batch in enumerate(dataloader):
                                                if bidx >= grid.max_batches:
                                                    break
                                                
                                                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                                                x = x.to(device)
                                                
                                                out = eval_on_setting(
                                                        x=x,
                                                        blur_sigma=1.2,
                                                        noise_sigma=.01,
                                                        strong_K = 5,
                                                        mu_kind='cosine',
                                                        model=model,
                                                        risk_fn=None,
                                                        debug=True
                                                )
                                                
                                                u_low_all.append(out['u_low'].detach())
                                                u_medium_all.append(out['u_medium'].detach())
                                                u_strong_all.append(out['u_strong'].detach())
                                                risk_all.append(out['risk'].detach())
                                                
                                                psnr_det_list.append(out['psnr'])
                                                ssim_det_list.append(out['ssim'])
                                                
                                                n_samples += int(x.shape[0])
                                            
                                            if n_samples == 0:
                                                continue
                                            
                                            u_low = torch.cat(u_low_all, dim=0)
                                            u_medium = torch.cat(u_medium_all, dim=0)
                                            u_strong = torch.cat(u_strong_all, dim=0)
                                            risk = torch.cat(risk_all, dim=0)
                                            
                                            in_outs = {'u_low': u_low, 'u_medium': u_medium, 'u_strong': u_strong, 'risk': risk}
                                            outs = evaluate_uncertainty(in_outs, std=grid.std_aurc)
                                            
                                            psnr_det_mean = float(sum(psnr_det_list)/max(1, len(psnr_det_list)))
                                            ssim_det_mean = float(sum(ssim_det_list)/max(1, len(ssim_det_list)))
                                            risk_mean = float(risk.detach().float().mean().item())
                                            
                                            dt = time.time() - t_0
                                            
                                            row = {
                                                "seed": seed,
                                                "denoiser": d_name,
                                                "blur_sigma": blur_sigma,
                                                "noise_sigma": noise_sigma,
                                                "mu_kind": str(mu_kind),
                                                "iters": iters,
                                                "cg_iters": cg_iters,
                                                "num_samples": n_samples,
                                                "batch": x,
                                                "strong_K": strong_K,
                                                "AT_mode": AT_mode,
                                                "psnr_det_mean": psnr_det_mean,
                                                "ssim_det_mean": ssim_det_mean,
                                                "aurc_low": outs['u_low']['aurc'],
                                                "aurc_medium": outs['u_medium']['aurc'],
                                                "aurc_strong": outs['u_strong']['aurc'],
                                                "risk_mean": risk_mean,
                                                "time": float(dt),
                                                **out
                                            }
                                            
                                            _append_row_csv(out_csv, fieldnames, row)
                                            
                                            print(
                                                f'[saved]: seed={seed} | destination={d_name}'
                                                f'[aurc(low/medium/strong)]: low = {row['aurc_low']} | medium = {row['aurc_medium']} | strong = {row['aurc_strong']}'
                                            )


def train_anp_pnp():
    print("Starting ANP-PNP Tests...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    B = 1
    """ 
    # y=A(x)+noise
    # PCam synthetic inverse. 
    A, AT = make_A_AT(mode='blur', k=5, sigma_blur=1.2)
    """

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    dataset = PCamDataset(data_root='./data', split='train', transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available(), persistent_workers=False)

    batch = next(iter(dataloader))
    if isinstance(batch, (list, tuple)) and len(batch) >= 1:
        x = batch[0]
    else:
        x = batch
    
    x = x.to(device)
    
    # g: [B,2,H,W] for grayscale or [B,6,H,W] for RGB.
    # DiT in_channels == out_channels == g_channels
    tmp_g = grad2d_rgb(x[:1].to(device))
    L, W = tmp_g.shape[-2], tmp_g.shape[-1]
    in_ch, out_ch = tmp_g.shape[1], tmp_g.shape[1]
    assert L == W, (L, W)
    dit_config = DiT2DConfig(
        in_channels=in_ch,  
        out_channels=out_ch,  # DiT model outputs noise of same shape
        img_size=L,  # Spatial dimensions of the gradients
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
        try:
            out = build_model()
        except TypeError:
            out = build_model(patch=dit_config.patch)
            
        if isinstance(out, tuple) and len(out) == 2:
            model, cfg = out
        else:
            model, cfg = out, dit_config
        
        model = model.to(device)
        
    except Exception as e:
        print(f'dit: {e}')
        import traceback
        traceback.print_exc()
        raise ValueError('model shape')

    # Denoiser function G_denoiser for anp_pnp_hqs.
    # It receives complex gradients `g_complex` and a noise level `sigma`.
    # It should return denoised complex gradients.
    def G_denoiser(g, sigma):
        print(f'g.shape:', tuple(g.shape), 'sigma:', sigma)
        sigma = sigma_to_batch(sigma, g)
        sigma = normalize_sigma(sigma, g)

        return model(g, sigma)

    denoisers = {
        'blur': G_denoiser,
    }
    
    
    
    # eval
    res = eval_on_setting(
        x=x,
        blur_sigma=1.2,
        noise_sigma=.01,
        strong_K = 5,
        mu_kind='cosine',
        model=model,
        risk_fn=None,
        debug=True
    )
    
    print(
        f'[DET] psnr= {res['psnr']:.5g} | ssim = {res['ssim']:.5g}'
    )
    
    if evaluate_uncertainty is not None:
        in_outs = {
            'u_low': res['u_low'],
            'u_medium': res['u_medium'],
            'u_strong': res['u_strong'],
            'risk': res['risk']
        }
        outs = evaluate_uncertainty(in_outs, std=True, verbose=True)
    
        sanity_check(res['u_low'], res['risk'], std=True)
        sanity_check(res['u_strong'], res['risk'], std=True)
        
        return res, outs

    """grid = GridConfig(
        blur_sigmas = (0.0, 0.7, 1.2),
        noise_sigma = (0.0, 0.01, 0.03),
        mu_kinds = ('cosine',),
        iters_list = (10,),
        cg_iters_list = (20,),
        strong_K_list = (8,),
        max_batches = 10,
        std_aurc = True,
        AT_mode='blur'
        use_mc_risk = True,
    )

    # Test HQS
    try:
        print("Testing anp_pnp_hqs...")
        grid_search(
            out_csv = './runs/ablation_pcam_hqs.csv',
            device=device,
            dataloader=dataloader,
            denoiser_bank=denoisers,
            seeds=(0, 1),
            grid=grid,
        )
        print(f"HQS Success! Results saved to './runs/ablation_pcam_hqs.csv'")
    except Exception as e:
        print(f"HQS Failed: {e}")"""
    
if __name__ == "__main__":
    train_anp_pnp()
    
