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


def load_depth(path: str, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    if path.endswith(".npy"):
        depth = np.load(path).astype("float32")
    else:
        depth_image = Image.open(path)
        if depth_image.mode not in {"I;16", "I", "F"}:
            depth_image = depth_image.convert("L")
        depth_array = np.asarray(depth_image)
        depth = depth_array.astype("float32")
        if depth_array.dtype == np.uint8:
            # NYU Depth V2 Kaggle stores depth targets as 8-bit grayscale images.
            depth = depth / 255.0
        elif depth.max() > 255.0:
            # Common 16-bit metric depth convention: millimeters -> meters.
            depth = depth / 1000.0
    tensor = torch.from_numpy(depth)
    if tensor.ndim == 2:
        tensor = tensor[None]
    if size is not None and tuple(tensor.shape[-2:]) != size:
        tensor = F.interpolate(tensor[None], size=size, mode="bilinear", align_corners=False)[0]
    return tensor.float()
