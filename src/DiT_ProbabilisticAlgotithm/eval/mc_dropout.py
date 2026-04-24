import torch
import torch.nn as nn



from contextlib import contextmanager
from typing import Dict, Optional, Tuple


def _extract_tensor_output(out):

    if torch.is_tensor(out):
        return out


    if isinstance(out, dict):
        keys = ("x_hat_clamp", "denoised", "pred", "out", "logit")
        for key in keys:
            if key in out and torch.is_tensor(out[key]):
                return out[key]
        
        tensor_items = [v for v in out.values() if torch.is_tensor(v)]

        if len(tensor_items) == 1:
            
            print(f"tensor_items == {len(tensor_items)}")
            return tensor_items[0]
        
        raise TypeError(f"model returned dict, but tensors are unclear, {list(out.keys())}")

    if isinstance(out, (tuple, list)):
        tensor_items = [v for v in out if torch.is_tensor(v)]
        if not tensor_items:
            raise TypeError(f" model produced : {type(out).__name__} no tensor")

        return tensor_items[-1]
    raise TypeError(f"unsupported model output type included: {type(out)}")

@contextmanager
def enable_mc_dropout(model: nn.Module):

    prev_states = {m:m.training for m in model.modules()}

    try:
        model.eval()

        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
            
        yield model

    finally:
        for m, state in prev_states.items():
            m.train(state)


@torch.inference_mode()
def mc_dropout_predict(model:nn.Module, denoiser, x, sigma, run_one_fn, run_one_kwargs: dict, return_tracks = False, num_samples=8, reduce_dims: Optional[tuple[int, ...]] = None):
    """
    mean = None
    mm = None
    count = 0
    """
    with enable_mc_dropout(model):

        x_samples = []
        track_samples = []

        for _ in range(num_samples):
            out = run_one_fn(**run_one_kwargs)
            # out = _extract_tensor_output(out)

            x_samples.append(out["x_hat_clamp"].detach().cpu())

            if return_tracks:
                track_samples.append(out.get("track", None))

        x_t = torch.stack(x_samples, dim=0)
        mean_mc = x_t.mean(dim=0)

        var_mc = x_t.var(dim=0, unbiased=False)

    result = {"mc": x_t, "mean_mc": mean_mc, "var_mc": var_mc, "std_mc": torch.sqrt(var_mc).clamp(1, 0)}
    
    return result

    



