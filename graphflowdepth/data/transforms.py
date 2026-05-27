from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_rgb(path: str, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize((size[1], size[0]), Image.BICUBIC)
    array = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def load_depth(
    path: str,
    size: Optional[Tuple[int, int]] = None,
    target_mode: str = "auto",
) -> torch.Tensor:
    depth, _ = load_depth_with_valid_mask(path, size=size, target_mode=target_mode)
    return depth


def load_depth_with_valid_mask(
    path: str,
    size: Optional[Tuple[int, int]] = None,
    target_mode: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if target_mode == "display_inverse":
        return load_depth_display_with_valid_mask(path, size=size)
    if target_mode not in {"auto", "metric"}:
        raise ValueError(f"Unknown depth target_mode: {target_mode}")

    if path.endswith(".npy"):
        depth = np.load(path).astype("float32")
        valid = np.isfinite(depth)
        if depth.size > 0 and np.nanmax(depth) > 1.0:
            valid = valid & (depth > 0)
    else:
        depth_image = Image.open(path)
        if depth_image.mode not in {"I;16", "I", "F"}:
            depth_image = depth_image.convert("L")
        depth_array = np.asarray(depth_image)
        depth = depth_array.astype("float32")
        if depth_array.dtype == np.uint8:
            valid = np.isfinite(depth)
            if target_mode == "metric":
                depth = depth / 255.0
            else:
                # NYU Depth V2 Kaggle stores targets as 8-bit grayscale where
                # brighter means farther. DA2-style visual depth is inverse:
                # closer is brighter, farther is darker.
                depth = 1.0 - depth / 255.0
        elif depth.max() > 255.0:
            valid = np.isfinite(depth) & (depth > 0)
            # Common 16-bit metric depth convention: millimeters -> meters.
            depth = depth / 1000.0
        else:
            valid = np.isfinite(depth)
            if target_mode == "metric":
                valid = valid & (depth > 0)
    tensor = torch.from_numpy(depth)
    valid_tensor = torch.from_numpy(valid)
    if tensor.ndim == 2:
        tensor = tensor[None]
    if valid_tensor.ndim == 2:
        valid_tensor = valid_tensor[None]
    if size is not None and tuple(tensor.shape[-2:]) != size:
        tensor = F.interpolate(tensor[None], size=size, mode="bilinear", align_corners=False)[0]
        valid_tensor = F.interpolate(valid_tensor[None].float(), size=size, mode="nearest")[0] > 0.5
    return tensor.float(), valid_tensor.bool()


def load_depth_display(path: str, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    depth, _ = load_depth_display_with_valid_mask(path, size=size)
    return depth


def load_depth_display_with_valid_mask(
    path: str,
    size: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a depth target as normalized inverse depth in [0, 1].

    Handles both NYU Kaggle 8-bit grayscale depth images and 16-bit/metric
    depth images. The output is near-bright/far-dark, matching DA2-style depth
    visualizations and the GraphFlowDepth MVP training target.
    """
    if path.endswith(".npy"):
        depth = np.load(path).astype("float32")
    else:
        depth_image = Image.open(path)
        if depth_image.mode not in {"I;16", "I", "F"}:
            depth_image = depth_image.convert("L")
        depth_array = np.asarray(depth_image)
        depth = depth_array.astype("float32")
        if depth_array.dtype == np.uint8:
            # NYU Depth V2 Kaggle stores targets as 8-bit grayscale where
            # brighter means farther. DA2-style visual depth is inverse:
            # closer is brighter, farther is darker.
            display = 1.0 - depth / 255.0
            valid = np.isfinite(display)
            return _resize_depth_and_mask(torch.from_numpy(display), torch.from_numpy(valid), size)
        if depth.max() > 255.0:
            depth = depth / 1000.0

    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        display = np.zeros_like(depth, dtype="float32")
    else:
        lo = np.quantile(depth[valid], 0.01)
        hi = np.quantile(depth[valid], 0.99)
        metric_norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        display = 1.0 - metric_norm
        display[~valid] = 0.0
    return _resize_depth_and_mask(torch.from_numpy(display.astype("float32")), torch.from_numpy(valid), size)


def _resize_depth_display(tensor: torch.Tensor, size: Optional[Tuple[int, int]]) -> torch.Tensor:
    tensor, _ = _resize_depth_and_mask(tensor, torch.isfinite(tensor), size)
    return tensor


def _resize_depth_and_mask(
    tensor: torch.Tensor,
    valid_mask: torch.Tensor,
    size: Optional[Tuple[int, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if tensor.ndim == 2:
        tensor = tensor[None]
    if valid_mask.ndim == 2:
        valid_mask = valid_mask[None]
    if size is not None and tuple(tensor.shape[-2:]) != size:
        tensor = F.interpolate(tensor[None], size=size, mode="bilinear", align_corners=False)[0]
        valid_mask = F.interpolate(valid_mask[None].float(), size=size, mode="nearest")[0] > 0.5
    return tensor.float(), valid_mask.bool()
