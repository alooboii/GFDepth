from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


DA2_SMALL_CONFIG = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}


class FrozenDepthAnythingV2Wrapper(nn.Module):
    """Frozen Depth Anything V2 wrapper for adapter experiments.

    The official DA2 implementation exposes DINOv2 intermediate token features
    and a DPT depth head. This wrapper adapts the deepest token feature as a
    [B, C, Hp, Wp] map, then converts it back to the format expected by the DPT
    head. If your local DA2 fork differs, adjust only this wrapper.
    """

    def __init__(
        self,
        backbone: str = "depth_anything_v2_small",
        checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        if backbone != "depth_anything_v2_small":
            raise ValueError("MVP currently supports only depth_anything_v2_small")
        self.backbone = backbone
        self.patch_size = 14
        self.feature_dim = 384
        self.model = self._build_model(checkpoint_path)
        self.freeze()

    def _build_model(self, checkpoint_path: Optional[str]) -> nn.Module:
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as exc:
            raise ImportError(
                "Could not import Depth Anything V2. Install the official Depth-Anything-V2 "
                "package/repo so `from depth_anything_v2.dpt import DepthAnythingV2` works."
            ) from exc

        model = DepthAnythingV2(**DA2_SMALL_CONFIG)
        ckpt = checkpoint_path or os.environ.get("DEPTH_ANYTHING_V2_SMALL_CKPT")
        if ckpt:
            state = torch.load(ckpt, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=True)
        return model

    def freeze(self) -> None:
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def assert_frozen(self) -> None:
        trainable = [name for name, param in self.model.named_parameters() if param.requires_grad]
        if trainable:
            raise RuntimeError(f"DA2 must stay frozen, but trainable params were found: {trainable[:5]}")

    def extract_intermediate_features(
        self,
        x: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], Sequence[torch.Tensor], int, int]:
        """Return Z1..Z4 maps plus raw DA2 features for the frozen DPT head."""
        self.assert_frozen()
        patch_h = x.shape[-2] // self.patch_size
        patch_w = x.shape[-1] // self.patch_size
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError(f"Input too small for DA2 patch size {self.patch_size}: {tuple(x.shape)}")

        layer_idx = self.model.intermediate_layer_idx[self.model.encoder]
        raw_features = self.model.pretrained.get_intermediate_layers(
            x,
            layer_idx,
            return_class_token=True,
        )
        maps = [self._tokens_to_map(feature, patch_h, patch_w) for feature in raw_features]
        return maps, raw_features, patch_h, patch_w

    def replace_feature(
        self,
        raw_features: Sequence[torch.Tensor],
        feature_index: int,
        feature_map: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Replace one token feature in the raw feature list with a map value."""
        replaced = list(raw_features)
        original = replaced[feature_index]
        tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
        if isinstance(original, tuple):
            replaced[feature_index] = (tokens, original[1])
        else:
            replaced[feature_index] = tokens
        return replaced

    def depth_from_raw_features(
        self,
        raw_features: Sequence[torch.Tensor],
        patch_h: int,
        patch_w: int,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Run the frozen DPT head. Gradients may flow to replaced feature tensors."""
        self.assert_frozen()
        depth = self.model.depth_head(raw_features, patch_h, patch_w)
        depth = F.relu(depth)
        if depth.ndim == 3:
            depth = depth[:, None]
        if output_size is not None and depth.shape[-2:] != output_size:
            depth = F.interpolate(depth, size=output_size, mode="bilinear", align_corners=True)
        return depth

    @torch.no_grad()
    def forward_baseline(self, x: torch.Tensor) -> torch.Tensor:
        maps, raw, patch_h, patch_w = self.extract_intermediate_features(x)
        del maps
        return self.depth_from_raw_features(raw, patch_h, patch_w, output_size=x.shape[-2:])

    def _tokens_to_map(self, feature: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        tokens = feature[0] if isinstance(feature, tuple) else feature
        b, n, c = tokens.shape
        expected = patch_h * patch_w
        if n != expected:
            raise RuntimeError(
                f"DA2 token count {n} does not match patch grid {patch_h}x{patch_w}. "
                "Check image sizing or local DA2 internals."
            )
        return tokens.transpose(1, 2).reshape(b, c, patch_h, patch_w).contiguous()
