import math
from typing import Dict

import torch
import torch.nn.functional as F


def _valid_depth_mask(target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.bool() & torch.isfinite(target)
    if target.numel() > 0 and target.detach().min() >= 0 and target.detach().max() <= 1:
        return valid
    return valid & (target > 0)


def depth_metrics(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, float]:
    if pred.ndim == 3:
        pred = pred[:, None]
    if target.ndim == 3:
        target = target[:, None]
    if valid_mask.ndim == 3:
        valid_mask = valid_mask[:, None]
    valid = _valid_depth_mask(target, valid_mask)
    if not valid.any():
        return {"absrel": math.nan, "rmse": math.nan, "delta1": math.nan}
    p = pred[valid].clamp_min(1e-6)
    t = target[valid].clamp_min(1e-6)
    ratio = torch.maximum(p / t, t / p)
    return {
        "absrel": ((p - t).abs() / t).mean().item(),
        "rmse": torch.sqrt(((p - t) ** 2).mean()).item(),
        "delta1": (ratio < 1.25).float().mean().item(),
    }


def display_depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    min_target: float = 0.03,
) -> Dict[str, float]:
    """Metrics for normalized inverse-depth/display targets in [0, 1].

    AbsRel and ratio metrics are unstable when inverse-depth targets approach
    zero, so those two are reported on an epsilon-masked subset. MAE/RMSE use
    all finite display-valid pixels and are the primary metrics for this mode.
    """
    if pred.ndim == 3:
        pred = pred[:, None]
    if target.ndim == 3:
        target = target[:, None]
    if valid_mask.ndim == 3:
        valid_mask = valid_mask[:, None]

    valid = valid_mask.bool() & torch.isfinite(pred) & torch.isfinite(target)
    if not valid.any():
        return {"mae": math.nan, "rmse": math.nan, "absrel_eps": math.nan, "delta1_eps": math.nan}

    p = pred[valid].float()
    t = target[valid].float()
    mae = (p - t).abs().mean().item()
    rmse = torch.sqrt(((p - t) ** 2).mean()).item()

    ratio_valid = valid & (target > min_target) & (pred > min_target)
    if not ratio_valid.any():
        absrel = math.nan
        delta1 = math.nan
    else:
        p_ratio = pred[ratio_valid].float()
        t_ratio = target[ratio_valid].float()
        ratio = torch.maximum(p_ratio / t_ratio, t_ratio / p_ratio)
        absrel = ((p_ratio - t_ratio).abs() / t_ratio).mean().item()
        delta1 = (ratio < 1.25).float().mean().item()

    return {"mae": mae, "rmse": rmse, "absrel_eps": absrel, "delta1_eps": delta1}


def patch_edge_gradient_error(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    patch_size: tuple[int, int],
) -> float:
    if pred_depth.ndim == 3:
        pred_depth = pred_depth[:, None]
    if target_depth.ndim == 3:
        target_depth = target_depth[:, None]
    if valid_mask.ndim == 3:
        valid_mask = valid_mask[:, None]
    valid_float = valid_mask.float()
    pred_patch = F.adaptive_avg_pool2d(pred_depth, patch_size)
    valid_patch = F.adaptive_avg_pool2d(valid_float, patch_size) > 0.5
    target_sum = F.adaptive_avg_pool2d(target_depth * valid_float, patch_size)
    valid_avg = F.adaptive_avg_pool2d(valid_float, patch_size)
    target_patch = target_sum / valid_avg.clamp_min(1e-6)
    errors = []
    for dy, dx, scale in [(0, 1, 1.0), (1, 0, 1.0), (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0))]:
        ps, pt = _edges(pred_patch, dy, dx)
        ts, tt = _edges(target_patch, dy, dx)
        ms, mt = _edges(valid_patch.float(), dy, dx)
        valid = (ms > 0.5) & (mt > 0.5)
        if valid.any():
            pred_grad = (pt - ps) / scale
            target_grad = (tt - ts) / scale
            errors.append((pred_grad - target_grad).abs()[valid].mean())
    if not errors:
        return math.nan
    return torch.stack(errors).mean().item()


def _axis_slices(length: int, delta: int):
    if delta == 1:
        return slice(0, length - 1), slice(1, length)
    if delta == -1:
        return slice(1, length), slice(0, length - 1)
    return slice(0, length), slice(0, length)


def _edges(x: torch.Tensor, dy: int, dx: int):
    ys, yt = _axis_slices(x.shape[-2], dy)
    xs, xt = _axis_slices(x.shape[-1], dx)
    return x[:, :, ys, xs], x[:, :, yt, xt]
