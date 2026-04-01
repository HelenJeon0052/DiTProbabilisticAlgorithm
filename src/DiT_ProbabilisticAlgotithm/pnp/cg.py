import torch




@torch.no_grad()
def cg_solve(Aop, b, x0=None, iters=50, tol=1e-6):
    # x-update : solve (A^T A + (beta/2) I) x = A^T y + (beta/2) x
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - Aop(x)
    p = r.clone()
    rsold = (r*r).sum(dim=(1,2,3), keepdim=True)

    for _ in range(iters):
        Ap = Aop(p)
        denom = (p*Ap).sum(dim=(1,2,3), keepdim=True) + 1e-12
        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = (r*r).sum(dim=(1,2,3), keepdim=True)

        if torch.sqrt(rsnew.mean()).item() < tol:
            break

        p = r + (rsnew / (rsold + 1e-12)) * p
        rsold = rsnew

    return x