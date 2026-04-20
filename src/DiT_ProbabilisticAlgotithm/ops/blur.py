import torch
import torch.nn.functional as F



K = 5

def make_gaussian_kernel(k=K, sigma=2.0, device='cpu', dtype=torch.float32):
    ax = torch.arange(k, device=device, dtype=dtype) - (k - 1) // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    return kernel


class BlurA:

    def __init__(self, kernel_2d: torch.Tensor):
        self.kernel2d = kernel_2d
        k = self.kernel2d[None, None] # [1, 1, kl, kw]
        self.k = k

    def A(self, x):
        k = self.k.to(x.device, x.dtype)
        k3 = k.repeat(3, 1, 1, 1) # [3, 1, kl, kw]

        return F.conv2d(x, k3, padding='same', groups=3)

    def AT(self, y):
        return self.A(y)

@torch.no_grad()
def synth_blur_noise(x, blur_sigma=2.0, noise_sigma=0.01):
    kernel = make_gaussian_kernel(k=9, sigma=blur_sigma, device=x.device, dtype=x.dtype)
    blur = BlurA(kernel)
    y = blur.A(x)
    y_t = (y + noise_sigma * torch.randn_like(y)).clamp(0, 1)

    return y_t, blur

# -------------------------------
# mse
# -------------------------------
@torch.no_grad()
def mse(a, b):
    return torch.mean((a - b)**2).item()

# -------------------------------
# PSNR / SSIM
# -------------------------------
@torch.no_grad()
def psnr(x_hat, x, data_range=1.0, eps=1e-8):
    mse = (x_hat -x).pow(2).mean(dim=(1, 2, 3)) # F.mse_loss(x_hat, x)
    return (10.0 * torch.log10((data_range**2)/(mse+eps))).mean().item()

def _gaussian_1d(window_size, sigma, device, dtype):
    ax = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) /2
    g = torch.exp(-(ax**2)/(2 * sigma**2))
    return g / g.sum()

def _create_window(window_size, sigma, channels, device, dtype):
    g = _gaussian_1d(window_size, sigma, device, dtype) # [1, 1, w, w]
    w = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0) # [C, 1, w, w]

    return w.repeat(channels, 1, 1, 1)


def blur2d(x, kernel):
    kernel = kernel.to(device=x.device, dtype=x.dtype)
    k = kernel.shape[-1]
    w = kernel.view(1, 1, k, k).repeat(x.shape[1], 1, 1, 1) # [C, 1, k, k]

    return F.conv2d(x, w, padding=k//2, groups=x.shape[1])

@torch.no_grad()
def estimate_optimal_norm(A, AT, x_0, device, iters=10, eps=1e-10):
    """
    estimate ||A||^2 via power iteration on (A^T A)
    return : scalar (float, approximation of norm A)
    """

    x = torch.randn_like(x_0).to(device)
    x = x / (x.norm() + eps)


    for _ in range(iters):
        y = A(x)
        x = AT(y)
        n = x.norm() + eps
        x = x / n
    
    y = A(x)
    opt = y.norm()
    return float(opt)
    