import csv
import os
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .transforms import load_depth, load_rgb


class RGBDepthDataset(Dataset):
    """Simple RGB/depth dataset for text or CSV lists.

    Text list format per line:
        rgb_path depth_path

    CSV list format per row:
        rgb_path,depth_path

    The Kaggle NYU Depth V2 dataset uses CSV rows with paths relative to
    `/kaggle/input/nyu-depth-v2/nyu_data`; pass that as `data_root`.
    """

    def __init__(
        self,
        data_root: str,
        list_path: str,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.data_root = data_root
        self.image_size = image_size
        self.list_path = self._resolve_list_path(list_path)
        self.samples = self._read_samples(self.list_path)
        if not self.samples:
            raise ValueError(f"No RGB/depth samples found in {self.list_path}")

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

    def _resolve_list_path(self, list_path: str) -> str:
        if os.path.isabs(list_path):
            return list_path
        if os.path.exists(list_path):
            return list_path
        return os.path.join(self.data_root, list_path)

    def _read_samples(self, list_path: str) -> List[Tuple[str, str]]:
        if list_path.lower().endswith(".csv"):
            return self._read_csv_samples(list_path)
        return self._read_text_samples(list_path)

    def _read_text_samples(self, list_path: str) -> List[Tuple[str, str]]:
        samples: List[Tuple[str, str]] = []
        with open(list_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Expected `rgb depth` in {list_path}, got: {line}")
                samples.append((self._resolve(parts[0]), self._resolve(parts[1])))
        return samples

    def _read_csv_samples(self, list_path: str) -> List[Tuple[str, str]]:
        samples: List[Tuple[str, str]] = []
        with open(list_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row or row[0].strip().startswith("#"):
                    continue
                if len(row) < 2:
                    raise ValueError(f"Expected `rgb_path,depth_path` in {list_path}, got: {row}")
                rgb_path = row[0].strip()
                depth_path = row[1].strip()
                if rgb_path.lower() in {"image", "rgb", "rgb_path"}:
                    continue
                samples.append((self._resolve(rgb_path), self._resolve(depth_path)))
        return samples
