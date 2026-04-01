import os, csv, time, math, random
import numpy as np



import torch
from torch.utils.data import DataLoader
import traceback

from torchvision import transforms

from DiT_ProbabilisticAlgotithm.src.models.dit import DiT2D, DiT2DConfig
from DiT_ProbabilisticAlgotithm.src.utils.make_dit_builder import make_dit_builder
from DiT_ProbabilisticAlgotithm.src.pnp.pnpstarter import hqs_solve
from DiT_ProbabilisticAlgotithm.src.data.pcam_starter import show_batch, PCamDataset
from DiT_ProbabilisticAlgotithm.src.ops.blur import make_gaussian_kernel, blur2d, mse, psnr
from DiT_ProbabilisticAlgotithm.src.utils.utils import set_seed


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


def make_A_AT(mode, k, sigma_blur, AT_mode:str, device):
    # Operators A=I
    A = lambda x: blur2d(x, k=k, sigma=sigma_blur) if 'k' in blur2d.__code__.co_varnames else blur2d(x)
    
    kernel = make_gaussian_kernel(k=5, sigma=1.2, device=device, dtype=torch.float32)
    kernel_flip = torch.flip(kernel, dims=[-1, -2])


    if AT_mode == 'blur':
        AT = lambda y: blur2d(y, k=k, sigma=sigma_blur) if 'k' in blur2d.__code__.co_varnames else blur2d(y)
    elif AT_mode == 'inverse':
        AT = lambda y: blur2d(y, kernel_flip, sigma=sigma_blur)
    else:
        raise ValueError(f'Unknown AT_mode: {AT_mode}')
    
    return A, AT
        

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
    x_hat = hqs_solve(y, A, AT, denoiser_fn, iters=hps_iters, cg_iters=cg_iters)
    t_hqs = time.time() - t_0

    x_hat_clamp = x_hat.clamp(0, 1)

    out = {
        "finite": int(torch.isfinite(x_hat).all().item()),
        "time": t_hqs,
        "mse_x": mse(x_hat_clamp, x_gt),
        "psnr": psnr(x_hat_clamp, x_gt),
        "mse_Ax_y": mse(A(s_hat_clamp), y),
        "x_hat_mean": x_hat_clamp.mean().item(),
        "x_hat_min": x_hat_clamp.min().item(),
        "x_hat_max": x_hat_clamp.max().item(),
    }

    return out

# -----------------------------
# grid runner
# -----------------------------
def grid_search(
    out_csv: str,
    device,
    dataloader,
    denoiser_bank: dict,
    seeds=(0, 1, 2),
    sigma_obs_list=(0.01, 0.03, 0.05, 0.08),
    k_list=(3, 5, 7),
    sigma_blur_list=(0.8, 1.2, 1.6),
    AT_modes=('blur', 'inverse'),
    hqs_iters_list=(3, 5, 12),
    cg_iters_list=(5, 10, 30),
):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    fieldnames = [
        "seed","denoiser","sigma_obs","k","sigma_blur", "AT_mode","hqs_iters","cg_iters","finite","time","mse_x","psnr","mse_Ax_y","x_hat_mean","x_hat_min","x_hat_max"
    ]

    write_header = not os.path.exists(out_csv)
    with open(out_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        
        for seed in seeds:
            set_seed(seed)
            x_gt, label = next(iter(dataloader))
            x_gt = x_gt.to(device)

            for d_name, d_fn in denoiser_bank.items():
                for sigma_obs in sigma_obs_list:
                    for k in k_list:
                        for sigma_blur in sigma_blur_list:
                            for AT_mode in AT_modes:
                                A, AT = make_A_AT(k=k, sigma_blur=sigma_blur, AT_mode=AT_mode)
                                for hqs_iters in hqs_iters_list:
                                    for cg_iters in cg_iters_list:
                                        out = run_one(
                                            device=device,
                                            x_gt=x_gt,
                                            A=A,
                                            AT=AT,
                                            denoiser_fn=d_fn,
                                            sigma_obs=sigma_obs,
                                            hqs_iters=hqs_iters,
                                            cg_iters=cg_iters,
                                        )

                                        out_row = {
                                            "seed": seed,
                                            "denoiser": d_name,
                                            "sigma_obs": sigma_obs,
                                            "k": k,
                                            "sigma_blur": sigma_blur,
                                            "AT_mode": AT_mode,
                                            "hqs_iters": hqs_iters,
                                            "cg_iters": cg_iters,
                                            **out
                                        }

                                        writer.writerow(out_row)
                                        f.flush()
                                        print(f'row: {row}')


def train_anp_pnp():
    print("Starting ANP-PNP Tests...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    B, C, H, W = 1, 3, 32, 32
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
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True, num_workers=2, pin_memory=True)

    # show_batch(dataloader) """

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
        build_model = make_dit_builder(DiT2D, dit_config, dit_config)
        model, cfg = build_model(patch=4).to(device)
    except Exception as e:
        print(f'dit: {e}')
        """
        tr=traceback.print_exc()
            print(tr)
        """

    # Denoiser function G_denoiser for anp_pnp_hqs.
    # It receives complex gradients `g_complex` and a noise level `sigma`.
    # It should return denoised complex gradients.
    def G_denoiser(g, sigma):
        print(f'g.shape:', tuple(g.shape), 'sigma:', sigma)
        sigma = normalize_sigma(sigma, g)

        return model(g, sigma)

    denoisers = {
        'blur': G_denoiser,
    }
        

    # Test HQS
    try:
        print("Testing anp_pnp_hqs...")
        grid_search(
            out_csv = './runs/ablation_pcam_hqs.csv',
            device=device,
            dataloader=dataloader,
            denoiser_bank=denoisers,
            seeds=(0, 1),  # Use more seeds for full experiments
        )
        print(f"HQS Success! Results saved to './runs/ablation_pcam_hqs.csv'")
    except Exception as e:
        print(f"HQS Failed: {e}")
        """
	    import traceback
        tr = traceback.print_exc()
        prin(tr)
	    """


if __name__ == "__main__":
    train_anp_pnp()

