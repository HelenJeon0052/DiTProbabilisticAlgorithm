import torch






@torch.no_grad()
def uncertainty_scores(
    x_hat, y, A, g_last, z_last, track, strong_K: int = 5, strong_eps: float = 0.05, strong_fn=None
):
    """
    :return: per-sample uncetainties: low | medium | strong, [B]
     - low : proxy from last primal / dual mismatch
     - medium : last residual terms tracked by solver (dc + cons)
     - strong : if strong_fn == None : sensitivity-to-input-noise proxy around x_hat , else : ensemble variance from strong_fn() stochastic outputs
    """

    # low , [B, C, L, W]
    if z_last.shape != g_last.shape:
        raise ValueError(f'shape mismatch: z_last={tuple(z_last.shape)} and g_last={tuple(g_last.shape)}')
    u_low = (z_last - g_last).pow(2).mean(dim=(1, 2, 3)).sqrt()

    # medium : last residuals
    if 'dc' not in track or 'cons' not in track or len(track['dc']) == 0 or len(track['cons']) == 0:
        raise ValueError('track does not contain dc and cons')
    
    dc_last = track['dc'][-1]
    cons_last = track['cons'][-1]
    u_medium = dc_last + cons_last # [B]

    # strong : ensemble variance
    
    xs = []
    if strong_fn is None:
        # u_strong - sensitivity proxy

        for _ in range(strong_K):
            xj = (x_hat + strong_eps * torch.randn_like(x_hat)).clamp(0, 1)
            xs.append(xj)
            print(f'xs.shape: {xs.append(xj)}')

        xs = torch.stack(xs, dim=0) # [K, B, 3, L, W]
        u_strong = xs.var(dim=0).mean(dim=(1, 2, 3)) # [B]
    else:
        xs = []
        for _ in range(strong_K):
            xj = strong_fn()
            if xj.shape != x_hat.shape:
                raise ValueError(f'strong_fn(): {x_hat.shape} | but {xj.shape}')
            xs.append(xj)

    xs = torch.stack(xs, dim=0)
    u_strong = xs.var(dim=0).mean(dim=(1, 2, 3))

    return u_low, u_medium, u_strong



def coverage_risk(uncertainty, risk, verbose: bool = False):
    """
    uncertainty : [N] higher = more uncertainty (reject H_0)
    risk : [N] higer = riskier
    :return: (coverages, risks_at_coverage, aurc)
    """

    un = uncertainty.detach().float().cpu()
    risk = risk.detach().float().cpu()
    if un.numel() != risk.numel():
        raise ValueError(f'uncertainty and risk must contain same length of value: {un.numel()} - {risk.numel()}')

    idx = torch.argsort(un)
    r_sorted = risk[idx]

    N = r_sorted.numel()
    coverages = torch.arange(1, N+1) / N
    cum_risk = torch.cumsum(r_sorted, dim=0) / torch.arange(1, N+1)


    # AURC : area under risk-coverage curve
    aurc = torch.trapz(cum_risk, coverages).item()

    if verbose:
        print('coverages:', coverages,
           'cum_risk:', cum_risk,
           'aurc:', aurc)

    return coverages.numpy(), cum_risk.numpy(), aurc


def coverage_risk_std(uncertainty, risk, verbose):
    un = uncertainty.detach().float().cpu()
    rk = risk.detach().float().cpu()
    if un.numel() != risk.numel():
        raise ValueError(f'uncertainty and risk must contain same length of value: {un.numel()} - {risk.numel()}')
    
    idx = torch.argsort(un)
    r_sorted = rk[idx]
    N = r_sorted.numel()
    
    cov = torch.arange(1, 1+N) / N
    cr = torch.cumsum(r_sorted, dim=0) / torch.arange(1, 1+N)
    
    cov_0 = torch.cat([torch.tensor([0.0]), cov])
    cr_0 = torch.cat([torch.tensor([0.0]), cr])
    
    aurc = torch.trapz(cr_0, cov_0).item()
    
    if verbose:
        print('coverage std:', cov_0)
        print('cum_risk_std:', cr_0)
        print('aurc_std:', aurc)
    
    return cov_0.numpy(), cr_0.numpy, aurc