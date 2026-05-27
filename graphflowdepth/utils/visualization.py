import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch


def save_depth_visualization(
    output_dir: str,
    stem: str,
    image: torch.Tensor,
    gt_depth: torch.Tensor,
    baseline_depth: Optional[torch.Tensor],
    graphflow_depth: torch.Tensor,
    velocity_maps: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    image_np = _denormalize_image(image.detach().cpu())
    gt = gt_depth.squeeze().detach().cpu().float()
    pred = graphflow_depth.squeeze().detach().cpu().float()
    valid = torch.isfinite(gt)
    if gt.min() < 0 or gt.max() > 1:
        valid = valid & (gt > 0)

    panels = [
        ("rgb", image_np),
        ("gt", _fixed_depth_for_display(gt)),
        ("graphflow", _fixed_depth_for_display(pred)),
        ("graphflow_error", _fixed_error_for_display((pred - gt).abs())),
    ]
    if baseline_depth is not None:
        base = baseline_depth.squeeze().detach().cpu().float()
        base_aligned = _align_scale_shift(base, gt, valid)
        panels.insert(2, ("baseline_aligned", _fixed_depth_for_display(base_aligned)))
        panels.insert(4, ("baseline_error", _fixed_error_for_display((base_aligned - gt).abs())))
    if velocity_maps:
        for name, value in velocity_maps.items():
            magnitude = value.detach().pow(2).sum(dim=0).sqrt().cpu()
            panels.append((f"vel_{name}", _normalize_for_display(magnitude)))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), squeeze=False)
    for ax, (title, value) in zip(axes[0], panels):
        ax.set_title(title)
        ax.axis("off")
        if title == "rgb":
            ax.imshow(value)
        else:
            ax.imshow(value, cmap="magma")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{stem}.png"), dpi=160)
    plt.close(fig)


def _denormalize_image(image: torch.Tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image * std + mean).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def _align_scale_shift(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    p = pred.squeeze().float()
    t = target.squeeze().float()
    m = mask.squeeze().bool()
    p_valid = p[m]
    t_valid = t[m]
    if p_valid.numel() == 0:
        return p
    p_mean = p_valid.mean()
    t_mean = t_valid.mean()
    cov = ((p_valid - p_mean) * (t_valid - t_mean)).mean()
    var = ((p_valid - p_mean) ** 2).mean().clamp_min(1e-6)
    scale = cov / var
    shift = t_mean - scale * p_mean
    return scale * p + shift


def _fixed_depth_for_display(depth: torch.Tensor) -> torch.Tensor:
    return depth.squeeze().float().clamp(0, 1)


def _fixed_error_for_display(error: torch.Tensor, vmax: float = 0.20) -> torch.Tensor:
    return error.squeeze().float().clamp(0, vmax) / vmax


def _normalize_for_display(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    return (value - value.min()) / (value.max() - value.min()).clamp_min(1e-6)
