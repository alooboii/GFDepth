from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .edge_heads import BoundaryHead, EdgeDepthHead


DIRECTION_OFFSETS = {
    "R": (0, 1),
    "D": (1, 0),
    "DR": (1, 1),
    "DL": (1, -1),
}


class AlphaFourierEmbedding(nn.Module):
    """Small deterministic Fourier embedding for alpha in [0, 1]."""

    def __init__(self, embed_dim: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        freq_count = max(1, embed_dim // 2)
        frequencies = torch.arange(1, freq_count + 1, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        if alpha.ndim == 1:
            alpha = alpha[:, None]
        angles = 2.0 * math.pi * alpha * self.frequencies[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] > self.embed_dim:
            emb = emb[..., : self.embed_dim]
        elif emb.shape[-1] < self.embed_dim:
            emb = F.pad(emb, (0, self.embed_dim - emb.shape[-1]))
        return emb


class FlowMLP(nn.Module):
    def __init__(
        self,
        graph_dim: int = 64,
        alpha_embed_dim: int = 16,
        direction_embed_dim: int = 16,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        input_dim = graph_dim + alpha_embed_dim + direction_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, graph_dim),
        )

    def forward(
        self,
        z_alpha: torch.Tensor,
        alpha_embedding: torch.Tensor,
        direction_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if direction_embedding.ndim == 1:
            direction_embedding = direction_embedding.unsqueeze(0).expand(z_alpha.shape[0], -1)
        x = torch.cat([z_alpha, alpha_embedding, direction_embedding], dim=-1)
        return self.net(x)


class GraphFlowAdapter(nn.Module):
    """Patch Graph Flow adapter operating on the deepest DA2 feature map.

    Input Z4 has shape [B, C, Hp, Wp]. The adapter projects to graph space,
    predicts local transition velocities for directed patch edges, aggregates
    the velocity maps, and projects back to C channels.
    """

    def __init__(
        self,
        input_dim: int = 384,
        graph_dim: int = 64,
        directions: Iterable[str] = ("R", "D", "DR", "DL"),
        direction_embed_dim: int = 16,
        alpha_embed_dim: int = 16,
        flow_hidden_dim: int = 128,
        enable_boundary_head: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.graph_dim = graph_dim
        self.directions = list(directions)
        for direction in self.directions:
            if direction not in DIRECTION_OFFSETS:
                raise ValueError(f"Unsupported direction {direction!r}. Known: {sorted(DIRECTION_OFFSETS)}")

        self.down_proj = nn.Conv2d(input_dim, graph_dim, kernel_size=1)
        self.alpha_embedding = AlphaFourierEmbedding(alpha_embed_dim)
        self.direction_embedding = nn.Embedding(len(self.directions), direction_embed_dim)
        self.flow_mlp = FlowMLP(graph_dim, alpha_embed_dim, direction_embed_dim, flow_hidden_dim)
        self.velocity_aggregation_proj = nn.Conv2d(len(self.directions) * graph_dim, graph_dim, kernel_size=1)
        self.up_proj = nn.Conv2d(graph_dim, input_dim, kernel_size=1)
        self.edge_depth_head = EdgeDepthHead(graph_dim, direction_embed_dim)
        self.boundary_head = BoundaryHead(graph_dim, direction_embed_dim) if enable_boundary_head else None

    def forward(
        self,
        z4: torch.Tensor,
        depth_gt: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        assert z4.ndim == 4, f"Expected Z4 [B,C,Hp,Wp], got {tuple(z4.shape)}"
        b, c, hp, wp = z4.shape
        assert c == self.input_dim, f"Adapter input_dim={self.input_dim}, got C={c}"

        zp = self.down_proj(z4)  # [B, 64, Hp, Wp]
        depth_patch, valid_patch = self._pool_depth_and_mask(depth_gt, valid_mask, (hp, wp))

        velocity_maps: Dict[str, torch.Tensor] = {}
        fm_predictions: List[torch.Tensor] = []
        fm_targets: List[torch.Tensor] = []
        edge_predictions: List[torch.Tensor] = []
        edge_targets: List[torch.Tensor] = []
        edge_valid_masks: List[torch.Tensor] = []

        for direction_idx, direction in enumerate(self.directions):
            dy, dx = DIRECTION_OFFSETS[direction]
            zi_map, zj_map, source_slices = self._edge_endpoint_maps(zp, dy, dx)
            zi = self._flatten_edge_map(zi_map)
            zj = self._flatten_edge_map(zj_map)
            direction_emb = self.direction_embedding.weight[direction_idx]

            # Flow-matching supervision samples an interpolation point per edge.
            alpha = torch.rand(zi.shape[0], 1, device=zi.device, dtype=zi.dtype) if self.training else torch.full(
                (zi.shape[0], 1), 0.5, device=zi.device, dtype=zi.dtype
            )
            z_alpha = (1.0 - alpha) * zi + alpha * zj
            alpha_emb = self.alpha_embedding(alpha)
            fm_pred = self.flow_mlp(z_alpha, alpha_emb, direction_emb)
            fm_target = (zj - zi).detach()
            fm_predictions.append(fm_pred)
            fm_targets.append(fm_target)

            # Midpoint velocity is used for feature fusion and edge-depth head.
            midpoint = 0.5 * (zi + zj)
            midpoint_alpha = torch.full((zi.shape[0], 1), 0.5, device=zi.device, dtype=zi.dtype)
            v_mid = self.flow_mlp(midpoint, self.alpha_embedding(midpoint_alpha), direction_emb)
            velocity_maps[direction] = self._scatter_velocity_map(v_mid, source_slices, (b, self.graph_dim, hp, wp))

            if depth_patch is not None and valid_patch is not None:
                d_i, d_j = self._edge_endpoint_maps(depth_patch, dy, dx)[:2]
                m_i, m_j = self._edge_endpoint_maps(valid_patch, dy, dx)[:2]
                edge_valid = (self._flatten_edge_map(m_i) > 0.5) & (self._flatten_edge_map(m_j) > 0.5)
                scale = math.sqrt(2.0) if direction in {"DR", "DL"} else 1.0
                true_gradient = (self._flatten_edge_map(d_j) - self._flatten_edge_map(d_i)) / scale
                pred_gradient = self.edge_depth_head(zi, zj, v_mid, direction_emb)
                edge_predictions.append(pred_gradient)
                edge_targets.append(true_gradient.detach())
                edge_valid_masks.append(edge_valid.detach())

        v = torch.cat([velocity_maps[d] for d in self.directions], dim=1)  # [B, 4*64, Hp, Wp]
        mp = self.velocity_aggregation_proj(v)
        m = self.up_proj(mp)  # [B, C, Hp, Wp]

        aux = {
            "projected_features": zp,
            "velocities": velocity_maps,
            "fm": {
                "predictions": torch.cat(fm_predictions, dim=0),
                "targets": torch.cat(fm_targets, dim=0),
            },
            "edge": self._pack_edge_aux(edge_predictions, edge_targets, edge_valid_masks, z4.device, z4.dtype),
        }
        return m, aux

    def _pool_depth_and_mask(
        self,
        depth_gt: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
        patch_size: Tuple[int, int],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if depth_gt is None:
            return None, None
        if depth_gt.ndim == 3:
            depth_gt = depth_gt[:, None]
        if valid_mask is None:
            valid_mask = depth_gt > 0
        if valid_mask.ndim == 3:
            valid_mask = valid_mask[:, None]
        valid_float = valid_mask.to(dtype=depth_gt.dtype)
        pooled_valid = F.adaptive_avg_pool2d(valid_float, patch_size)
        pooled_depth_sum = F.adaptive_avg_pool2d(depth_gt * valid_float, patch_size)
        depth_patch = pooled_depth_sum / pooled_valid.clamp_min(1e-6)
        valid_patch = pooled_valid > 0.5
        return depth_patch, valid_patch.to(dtype=depth_gt.dtype)

    @staticmethod
    def _axis_slices(length: int, delta: int) -> Tuple[slice, slice]:
        if delta == 1:
            return slice(0, length - 1), slice(1, length)
        if delta == -1:
            return slice(1, length), slice(0, length - 1)
        return slice(0, length), slice(0, length)

    def _edge_endpoint_maps(
        self,
        feature: torch.Tensor,
        dy: int,
        dx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[slice, slice]]:
        _, _, h, w = feature.shape
        ys, yt = self._axis_slices(h, dy)
        xs, xt = self._axis_slices(w, dx)
        return feature[:, :, ys, xs], feature[:, :, yt, xt], (ys, xs)

    @staticmethod
    def _flatten_edge_map(edge_map: torch.Tensor) -> torch.Tensor:
        return edge_map.permute(0, 2, 3, 1).reshape(-1, edge_map.shape[1])

    def _scatter_velocity_map(
        self,
        values: torch.Tensor,
        source_slices: Tuple[slice, slice],
        output_shape: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        b, c, h, w = output_shape
        velocity = values.new_zeros(output_shape)
        ys, xs = source_slices
        edge_h = len(range(*ys.indices(h)))
        edge_w = len(range(*xs.indices(w)))
        velocity[:, :, ys, xs] = values.reshape(b, edge_h, edge_w, c).permute(0, 3, 1, 2)
        return velocity

    @staticmethod
    def _pack_edge_aux(
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        valid_masks: List[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        if not predictions:
            empty = torch.empty(0, 1, device=device, dtype=dtype)
            return {"predictions": empty, "targets": empty, "valid_mask": empty.bool()}
        return {
            "predictions": torch.cat(predictions, dim=0),
            "targets": torch.cat(targets, dim=0),
            "valid_mask": torch.cat(valid_masks, dim=0),
        }
