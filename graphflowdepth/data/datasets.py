import os
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .transforms import load_depth, load_rgb


class RGBDepthDataset(Dataset):
    """Simple RGB/depth list dataset.

    List file format per line:
        rgb_path depth_path

    Paths may be absolute or relative to data_root.
    """

    def __init__(
        self,
        data_root: str,
        list_path: str,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.data_root = data_root
        self.image_size = image_size
        with open(list_path, "r", encoding="utf-8") as handle:
            self.samples: List[Tuple[str, str]] = []
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Expected `rgb depth` in {list_path}, got: {line}")
                self.samples.append((self._resolve(parts[0]), self._resolve(parts[1])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rgb_path, depth_path = self.samples[index]
        image = load_rgb(rgb_path, self.image_size)
        depth = load_depth(depth_path, self.image_size)
        valid_mask = torch.isfinite(depth) & (depth > 0)
        return {
            "image": image,
            "depth": depth,
            "valid_mask": valid_mask,
            "rgb_path": rgb_path,
            "depth_path": depth_path,
        }

    def _resolve(self, path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(self.data_root, path)
