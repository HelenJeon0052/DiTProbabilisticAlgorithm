import os, csv, time, math, random, json, hashlib
import numpy as np



import torch
from torch.utils.data import DataLoader
import traceback

from torchvision import transforms

from DiT_ProbabilisticAlgotithm.models.dit import DiT2D, DiT2DConfig
from DiT_ProbabilisticAlgotithm.utils.make_dit_builder import make_dit_builder
from DiT_ProbabilisticAlgotithm.pnp.pnpstarter import hqs_solve, normalize_sigma
from DiT_ProbabilisticAlgotithm.data.pcam_starter import show_batch, PCamDataset
from DiT_ProbabilisticAlgotithm.ops.blur import make_gaussian_kernel, blur2d, mse, psnr
from DiT_ProbabilisticAlgotithm.utils.utils import set_seed


# -----------------------------
# utils
# -----------------------------
def canoical_run_id(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True, separators=(',', ':'))
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:12]

def load_done_ids(csv_path: str) -> set[str]:
    if not os.path.exists(csv_path):
        return set()
    done = set()
    with open(csv_path, 'r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            if 'run_id' in row:
                done.add(row['run_id'])
    return done

# -----------------------------
# adjoint
# ------------------------------
@torch.no_grad()
def adjoint_test(A, AT, shape, device, trials=2, eps=1e-5, seed=2):
    g = torch.Generator(device=device).manual_seed(seed)
    errs = []
    
    for t in range(trials):
        x = torch.randn(*shape, device=device, generator=g)
        y = torch.randn(*shape, device=device, generator=g)
        
        Ax = A(x)
        ATy = AT(y)
        ls = (Ax * y).sum()
        rs = (x * ATy).sum()
        rel = (ls - rs).abs() / (ls.abs() + rs.abs() + eps)
        errs.append(rel.item())
        
    return {
        'rel_err_mean': float(sum(errs) / len(errs)),
        'rel_err_min': float(min(errs)),
        'rel_err_max': float(max(errs)),
    }
        
# -----------------------------
# index-fixed samples
# -----------------------------
def make_fixed_samples(dataset, indices, batch_size=1, num_workers=2):
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# -----------------------------
# noise-fixed builder
# -----------------------------
@torch.no_grad()
def make_fixed_noise_builder(A, x_gt, sigma_obs:float, noise_seed: int):
    g = torch.Generator(device=x_gt.device).manual_seed(noise_seed)
    n = torch.randn_like(x_gt, generator=g)
    y = A(x_gt) + sigma_obs * n
    
    return y

def make_A_AT(AT_mode, k, sigma_blur):
    # Operators A=I
    
    kernel = make_gaussian_kernel(k=k, sigma=sigma_blur, dtype=torch.float32)
    kernel_flip = torch.flip(kernel, dims=[-1, -2])
    
    A = lambda x: blur2d(x, kernel)

    if AT_mode == 'blur':
        AT = lambda y: blur2d(y, kernel)
    if AT_mode == 'inverse':
        AT = lambda y: blur2d(y, kernel_flip)
    else:
        raise ValueError(f'Unknown AT_mode: {AT_mode}')
    
    return A, AT

# -----------------------------
# runners
# -----------------------------

def run_phase(
    phase_name: str,
    out_csv: str,
    device, fixed_dataloader,
    denoiser_bank, make_A_AT_fn,
    grid, seeds=(0,),
    topN:int | None = None,
    sort_key:str = psnr,
    sort_desc:bool = True,
):
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    done_ids = load_done_ids(out_csv)
    
    fieldnames = [
        "phase","run_id", "seed","denoiser","sigma_obs","k","sigma_blur", "AT_mode","hqs_iters","cg_iters","finite","time","mse_x", "mse_Ax_y","psnr","rel_err_mean", "rel_err_min", "rel_err_max",
    ]
    
    write_header = not os.path.exists(out_csv)
    rows_rank = []
    
    with open(out_csv, 'a', newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        
        for cfg in grid:
            for d_name, d_fn in denoiser_bank.items():
                for seed in seeds:
                    run_cfg = dict(phase=phase_name, denoiser=d_name, seed=seed, **cfg)
                    run_id = canoical_run_id(run_cfg)
                    if run_id in done_ids:
                        print(f'Skipping done run_id: {run_id}')
                        continue
                    
                    set_seed(seed)
                    
                    A, AT = make_A_AT_fn(k=cfg["k"], sigma_blur=cfg["sigma_blur"], AT_mode=cfg["AT_mode"], device=device)
                    x_gt0, _ = next(iter(fixed_dataloader))
                    s = tuple(x_gt0.to(device).shape)
                    adjoint = adjoint_test(A, AT, shape=s, device=device, trials=seed, seed=seed)
                    
                    t_0 = time.time()
                    finite_all = True
                    mse_x, psnrs, mse_Ax_ys = [], [], []
                    
                    for i, (x_gt, _) in enumerate(fixed_dataloader):
                        x_gt = x_gt.to(device)
                        
                        # noise fixed per sample index i + seed
                        y = make_fixed_noise_builder(A, x_gt, sigma_obs=cfg['sigma_obs'], noise_seed=10_0)
                        
                        # HQS
                        x_hat = hqs_solve(
                            y, A, AT, d_fn,
                            iters=cfg['hqs_iters'],
                            cg_iters=cfg['cg_iters'],
                        )
                        
                        finite_all = finite_all and bool(torch.isfinite(x_hat).all().item())
                        x_hat = x_hat.clamp(0, 1)
                        
                        mse_x.append(mse(x_hat, x_gt))
                        psnrs.append(psnr(x_hat, x_gt))
                        mse_Ax_ys.append(mse(A(x_hat), y))
                        
                    dt = time.time() - t_0
                        
                    out = {
                        "finite":int(finite_all),
                        "time":dt,
                        "mse_x":float(sum(mse_x) / len(mse_x)),
                        "psnr":float(sum(psnrs) / len(psnrs)),
                        "mse_Ax_y":float(sum(mse_Ax_ys) / len(mse_Ax_ys)),
                    }

                    out_row = {
                        "phase": phase_name,
                        "run_id": run_id,
                        "seed": seed,
                        "denoiser": d_name,
                        **cfg,
                        **out,
                        **adjoint,
                    }
                    
                    w.writerow(out_row)
                    f.flush()
                    done_ids.add(run_id) 
                    rows_rank.append(out_row)
                    
                    print(f'Completed run_id: {run_id} | psnr: {out["psnr"]:.4f}')
                    
    if topN is not None and len(rows_rank) > 0:
        row_sorted = sorted(rows_rank, key=lambda r: r.get(sort_key, float('-inf')), reverse=sort_desc)
        picked = row_sorted[:topN]
        picked_cfgs = []
        for r in picked:
            picked_cfgs.append({
                'sigma_obs': float(r['sigma_obs']),
                'k': int(r['k']),
                'sigma_blur': float(r['sigma_blur']),
                'AT_mode': r['AT_mode'],
                'hqs_iters': int(r['hqs_iters']),
                'cg_iters': int(r['cg_iters']),
            })
        return picked_cfgs
    return None

def train_anp_pnp():
    print("Starting ANP-PNP Tests...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    dataset = PCamDataset(data_root='./data', split='train', transform=transform, download=True)
    
    fixed_inidices_1 = [12, 98]
    fixed_indicess_2 = [12, 98, 301, 777, 1024, 2048, 4096, 5000]

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

    # Denoiser function G_denoiser for anp_pnp_hqs.
    # It receives complex gradients `g_complex` and a noise level `sigma`.
    # It should return denoised complex gradients.
    def G_denoiser(g, sigma):
        print(f'g.shape:', tuple(g.shape), 'sigma:', sigma)
        sigma = normalize_sigma(sigma, g)

        return model(g, sigma)

    denoisers = {
        'blur': G_denoiser, # lambda g, sigma: g : # no normalization
        'inverse': G_denoiser,
    }
    
    grid_p1 = []
    
    for sigma_obs in (0.03, 0.05):
        for k in (3, 5):
            for sigma_blur in (0.8, 1.2):
                for AT_mode in ('blur', 'inverse'):
                    for hqs_iters in (5, 12):
                        for cg_iters in (10, 30):
                            grid_p1.append({
                                'sigma_obs': sigma_obs,
                                'k': k,
                                'sigma_blur': sigma_blur,
                                'AT_mode': AT_mode,
                                'hqs_iters': hqs_iters,
                                'cg_iters': cg_iters,
                            })
    
    top_cfgs = run_phase(
        phase_name='p1',
        out_csv='./runs/ablation_pcam_phase1.csv',
        device=device,
        fixed_dataloader=make_fixed_samples(dataset, fixed_inidices_1, batch_size=1),
        denoiser_bank=denoisers,
        make_A_AT_fn=make_A_AT,
        grid=grid_p1,
        seeds=(0,),  # Use more seeds for full experiments
        topN=10,
        sort_key='psnr',
        sort_desc=True,
    )
    
    if top_cfgs:
        run_phase(
            phase_name='p2',
            out_csv='./runs/ablation_pcam_phase2.csv',
            device=device,
            fixed_dataloader=make_fixed_samples(dataset, fixed_indicess_2, batch_size=1),
            denoiser_bank=denoisers,
            make_A_AT_fn=make_A_AT,
            grid=top_cfgs,
            seeds=(0, 1),
            topN=None,
        )

if __name__ == "__main__":
    train_anp_pnp()

