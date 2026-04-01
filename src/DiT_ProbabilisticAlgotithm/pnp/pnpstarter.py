import torch
from DiT_ProbabilisticAlgotithm.models.diffusion import GaussianDiffusion
from DiT_ProbabilisticAlgotithm.pnp.cg import cg_solve


from DiT_ProbabilisticAlgotithm.pnp.grad import grad2d_rgb, div2d_rgb

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

@torch.no_grad()
def hqs_solve(
        A, AT, y, z0, denoiser_ddim, G_denoiser, beta: float, lam: float, iters: int = 50, cg_steps: int =50, sigma_data=.01, AT_extra=None, out_ch=None,
):
    """
    :param A: linear operator
    :param AT: linear operator
    :param y: measurement
    :param z0: initial estimate
    """

    if AT_extra is None:
        x = AT(y)
    else:
        x = AT_extra(y, out_ch)

    if mu_schedule is None:
        mu_schedule = torch.logspace(-1, 2, steps=cg_steps, device=y.device)


    for k in range(iters):

        mu = mu_schedule[k]

        # z-step
        g = grad2d_rgb(x)
        sigma_k = (1.0 / torch.sqrt(mu)).item()
        z = G_denoiser(g, sigma_k)

        def ATy():
            if AT_extra is None:
                return AT(y)
            return AT_extra(y, out_ch)

        def ATA(v):
            if AT_extra is None:
                return AT(A(v))
            return AT_extra(A(v), out_ch)

        def linop(v):
            return ATA(v) / (sigma_data**2) + mu * div2d_rgb(grad2d_rgb(v))

        rhs = ATy() / (sigma_data**2) + mu * div2d_rgb(z)
        x = cg_solve(linop, rhs, x0=x, iters=cg_steps)

    return x
