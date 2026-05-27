from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn

from .da2_wrapper import FrozenDepthAnythingV2Wrapper
from .graphflow_adapter import GraphFlowAdapter


class GraphFlowDepthModel(nn.Module):
    """Frozen DA2 + trainable Patch Graph Flow adapter."""

    def __init__(
        self,
        backbone: str = "depth_anything_v2_small",
        graph_dim: int = 64,
        directions: Iterable[str] = ("R", "D", "DR", "DL"),
        direction_embed_dim: int = 16,
        alpha_embed_dim: int = 16,
        flow_hidden_dim: int = 128,
        gamma_init: float = 0.0,
        output_calibration: bool = True,
    ) -> None:
        super().__init__()
        self.da2 = FrozenDepthAnythingV2Wrapper(backbone=backbone)
        self.adapter = GraphFlowAdapter(
            input_dim=self.da2.feature_dim,
            graph_dim=graph_dim,
            directions=directions,
            direction_embed_dim=direction_embed_dim,
            alpha_embed_dim=alpha_embed_dim,
            flow_hidden_dim=flow_hidden_dim,
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.output_calibration = output_calibration
        self.output_log_scale = nn.Parameter(torch.zeros(()), requires_grad=output_calibration)
        self.output_shift = nn.Parameter(torch.zeros(()), requires_grad=output_calibration)
        self.assert_da2_frozen()

    def forward(
        self,
        x: torch.Tensor,
        depth_gt: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        return_baseline: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        self.assert_da2_frozen()
        with torch.no_grad():
            z_maps, raw_features, patch_h, patch_w = self.da2.extract_intermediate_features(x)
            z_maps = [z.detach() for z in z_maps]

        z4 = z_maps[-1]  # [B, 384, Hp, Wp] for DA2-Small
        residual, aux = self.adapter(z4, depth_gt=depth_gt, valid_mask=valid_mask)
        z4_tilde = z4 + self.gamma * residual
        replaced = self.da2.replace_feature(raw_features, -1, z4_tilde)
        pred_depth = self.da2.depth_from_raw_features(replaced, patch_h, patch_w, output_size=x.shape[-2:])
        if self.output_calibration:
            pred_depth = pred_depth * self.output_log_scale.exp() + self.output_shift

        aux["gamma"] = self.gamma
        aux["output_scale"] = self.output_log_scale.exp()
        aux["output_shift"] = self.output_shift
        if return_baseline:
            with torch.no_grad():
                aux["baseline_depth"] = self.da2.depth_from_raw_features(
                    raw_features,
                    patch_h,
                    patch_w,
                    output_size=x.shape[-2:],
                )
        return pred_depth, aux

    def trainable_parameters(self):
        return (param for param in self.parameters() if param.requires_grad)

    def assert_da2_frozen(self) -> None:
        self.da2.assert_frozen()
