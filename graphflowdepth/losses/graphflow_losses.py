from typing import Dict

import torch
import torch.nn.functional as F

from .depth_losses import masked_l1_depth_loss


def flow_matching_loss(aux_outputs: Dict[str, object]) -> torch.Tensor:
    fm = aux_outputs["fm"]
    pred = fm["predictions"]
    target = fm["targets"]
    if pred.numel() == 0:
        return pred.sum() * 0.0
    return F.mse_loss(pred, target)


def edge_depth_loss(aux_outputs: Dict[str, object]) -> torch.Tensor:
    edge = aux_outputs["edge"]
    pred = edge["predictions"]
    target = edge["targets"]
    valid = edge["valid_mask"]
    if pred.numel() == 0 or not valid.any():
        return pred.sum() * 0.0
    return F.l1_loss(pred[valid.squeeze(-1)], target[valid.squeeze(-1)])


def combined_loss(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    aux_outputs: Dict[str, object],
    depth_weight: float = 1.0,
    fm_weight: float = 0.05,
    edge_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
    depth = masked_l1_depth_loss(pred_depth, target_depth, valid_mask)
    fm = flow_matching_loss(aux_outputs)
    edge = edge_depth_loss(aux_outputs)
    total = depth_weight * depth + fm_weight * fm + edge_weight * edge
    return {
        "depth_loss": depth,
        "fm_loss": fm,
        "edge_loss": edge,
        "total_loss": total,
    }
