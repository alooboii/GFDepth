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
    panels = [
        ("rgb", image_np),
        ("gt", _normalize_depth_for_display(gt)),
        ("graphflow", _normalize_depth_for_display(pred)),
        ("abs_error", _normalize_depth_for_display((pred - gt).abs())),
    ]
    if baseline_depth is not None:
        base = baseline_depth.squeeze().detach().cpu().float()
        panels.insert(2, ("baseline", _normalize_depth_for_display(base)))
    if velocity_maps:
        for name, value in velocity_maps.items():
            panels.append((f"vel_{name}", value.detach().pow(2).sum(dim=0).sqrt().cpu()))

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


def _normalize_depth_for_display(depth: torch.Tensor) -> torch.Tensor:
    depth = depth.float()
    return (depth - depth.min()) / (depth.max() - depth.min()).clamp_min(1e-6)
