import torch
import os

from DiT_ProbabilisticAlgotithm.pnp.grad import grad2d_rgb


class EMA:
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = {}
        self._init_from(model)

    @torch.no_grad()
    def _init_from(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()
    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)
    @torch.no_grad()
    def copy_to(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])



def sample_sigma(B, device, kind='log_uniform', sigma_min=1e-3, sigma_max=1e-1):
    if kind == 'log_uniform':
        u = torch.rand(B, device=device)
        return sigma_min * (sigma_max / sigma_min) ** u
    elif kind == 'uniform':
        return sigma_min * (sigma_max / sigma_min) * torch.rand(B, device=device)
    else:
        raise ValueError(f'Unknown kind: {kind}')

def train_gradient_denoiser_dit(
        model, train_loader, val_loader, epochs = 3, lr=2e-3, ema_decay=0.99, sigma_min = 1e-3, sigma_max=1e-1,amp=True, out_dir='./runs_ema', log_every=50
):
    os.makedirs(out_dir, exist_ok=True)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = GradScaler(enabled=amp)
    ema = EMA(model, decay=ema_decay)

    def step_batch(images):
        B = images.shape[0]
        sigma_0 = sample_sigma(B, images.device, 'log_uniform', sigma_min, sigma_max) # [B]
        noise = torch.randn_like(images) * sigma_0.view(B, 1, 1, 1)
        x_noisy = (images + noise).clamp(0, 1)

        g_clean = grad2d_rgb(images)
        g_noisy = grad2d_rgb(x_noisy)

        with autocast(enabled=amp):
            g_hat = model(g_noisy, sigma_0)
            loss = (g_hat - g_clean).abs().mean() # L1
        return loss


