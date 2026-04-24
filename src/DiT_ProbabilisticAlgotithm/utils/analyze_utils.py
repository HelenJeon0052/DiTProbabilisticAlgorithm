import numpy as np
import torch



import matplotlib.pyplot as plt
from collections import defaultdict
# from typing import Optional, Tuple




def analyze_snr(diffusion_instance, device:str ='cpu'):
  
  
  """
  Signal-to-Noise Ratio
  """

  alpha_bar = diffusion_instance.alpha_bar.to(device).float()
  # Noise variance
  sigma_sq = torch.clamp(1.0 - alpha_bar, min=1e-10)

  snr = alpha_bar / sigma_sq
  log_snr = torch.log(snr.clamp(min=1e-8))
  print(f'log_snr: {log_snr.shape}')

  plt.figure(figsize=(10, 5))
  plt.plot(log_snr.cpu().numpy(), label='log(snr)', lw=2)
  plt.axhline(y=0, linestyle='--', alpha=.5, label='snr=1 (epsilon)')
  plt.title('log snr over time-step')
  plt.xlabel('t index')
  plt.ylabel('log(snr)')
  plt.grid(True, which='both', ls='-', alpha=.5)
  plt.legend()
  plt.show()

def check_mu_stability(mu_scheduler):
    mu_val = mu_scheduler.cpu().numpy()

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(np.log1p(mu_val), color='blue')
    plt.title("Log(1 + mu) Schedule")
    plt.ylabel("Log Scale (log(1+mu))")
    plt.xlabel("timestep")

    plt.subplot(1, 2, 2)
    plt.plot(np.diff(np.log1p(mu_val)), color='green')
    plt.title("Rate of Change (Delta Log(1+mu))")
    plt.xlabel("timestep")
    plt.ylabel("delta")

    plt.tight_layout()
    plt.show()


def analyze_t_distribution(denoiser, sigmas, diffusion_instance, num_steps=10000, device='cuda'):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    t_sigmas = torch.linspace(sigmas.min(), sigmas.max(), num_steps).to(device)
    print(f'[t distribution] t_sigmas: {t_sigmas.shape}')
    
    _, _, sigma_schedule = calibration_k(0.02, 23, diffusion_instance, device=device)
    
    d_eps = denoiser.eps
    denom = float(denoiser.opt_norm if denoiser.opt_norm is not None else 1.0) + d_eps
    scaled_sigmas = t_sigmas / denom

    t_indices = denoiser.to_t(scaled_sigmas, sigma_schedule, device=device)

    t_np = t_indices.cpu().numpy()

    used_idx = torch.unique(t_np)
    missing_idx = [i for i in range(num_steps) if i not in used_idx]
    print(f'Unique Indices Used: {len(used_idx)} / {num_steps}')

    if len(missing_idx) > 0:
      print(f'Missing Indices: {missing_idx[:10]}')
    else:
      print('no missing indices')

    print(f"Unique t-indices used: {len(np.unique(t_np))} / {len(sigma_schedule)}")
    print(f"t-index range: {t_np.min()} ~ {t_np.max()}")
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(t_np, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Time-steps (t)')
    plt.xlabel('t index')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(test_sigmas.cpu().numpy(), t_np, color='salmon', lw=2)
    plt.title('Sigma to T Mapping Curve')
    plt.xlabel('Input Sigma')
    plt.ylabel('Mapped t index')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


class LossMonitor:
    def __init__(self, num_bins:int=10, max_t:int=200):
        self.num_bins = num_bins
        self.max_t = max_t
        self.history:defaultdict = defaultdict(list)

    @torch.no_grad()
    def update(self, t_indices: torch.Tensor, loss_values: torch.Tensor) -> None:
        ts = t_indices.detach().cpu().numpy().flatten()
        ls = loss_values.detach().cpu().numpy().flatten()

        for t, l in zip(ts, ls):
            bin_idx = int((t/self.max_t) * self.num_bins)
            bin_idx = min(bin_idx, self.num_bins - 1)
            self.history[bin_idx].append(l)

    def report(self):
        """
        based on txt
        """
        print(f"\n{'='*15} Loss Convergence by T-Bins {'='*15}")
        print(f"{'Bin Range':<15} | {'Avg Loss':<12} | {'Samples':<8}")
        print("-" * 60)

        for i in range(self.num_bins):
            data = self.history[i]
            avg = np.mean(data) if data else 0.0
            count = len(data)

            t_start = int(i * (self.max_t / self.num_bins))
            t_end = int((i + 1) * (self.max_t / self.num_bins))

            bar = "█" * int(min(avg * 100, 30)) if avg > 0 else ""
            print(f"t={t_start:3d}~{t_end:3d} | {avg:12.6f} | {count:8d}  {bar}")

    def plot(self):
        """
        Matplotlib
        """
        bins = range(self.num_bins)
        avg_losses = [np.mean(self.history[i]) if self.history[i] else 0 for i in bins]

        plt.figure(figsize=(10, 5))
        plt.bar(bins, avg_losses, color='#4A90E2', edgecolor='black', alpha=0.8)

        labels = [f"{int(i*(self.max_t/self.num_bins))}" for i in bins]
        plt.xticks(bins, labels)
        plt.xlabel("Starting Timestep (t)")
        plt.ylabel("Average MSE Loss")
        plt.title("Loss Distribution Across Timesteps")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.show()