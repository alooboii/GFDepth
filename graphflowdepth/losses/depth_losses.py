import torch


def valid_depth_mask(target_depth: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.bool() & torch.isfinite(target_depth)
    if target_depth.numel() > 0 and target_depth.detach().min() >= 0 and target_depth.detach().max() <= 1:
        return valid
    return valid & (target_depth > 0)


def masked_l1_depth_loss(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if pred_depth.ndim == 3:
        pred_depth = pred_depth[:, None]
    if target_depth.ndim == 3:
        target_depth = target_depth[:, None]
    if valid_mask.ndim == 3:
        valid_mask = valid_mask[:, None]
    valid = valid_depth_mask(target_depth, valid_mask)
    loss = (pred_depth - target_depth).abs()
    return loss[valid].mean() if valid.any() else loss.sum() * 0.0 + eps * 0.0


def silog_loss(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    variance_focus: float = 0.85,
    eps: float = 1e-6,
) -> torch.Tensor:
    if pred_depth.ndim == 3:
        pred_depth = pred_depth[:, None]
    if target_depth.ndim == 3:
        target_depth = target_depth[:, None]
    if valid_mask.ndim == 3:
        valid_mask = valid_mask[:, None]
    valid = valid_depth_mask(target_depth, valid_mask)
    if not valid.any():
        return pred_depth.sum() * 0.0
    log_diff = torch.log(pred_depth[valid].clamp_min(eps)) - torch.log(target_depth[valid].clamp_min(eps))
    return torch.sqrt((log_diff**2).mean() - variance_focus * (log_diff.mean() ** 2) + eps)
