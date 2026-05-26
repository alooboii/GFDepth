import torch
from torch import nn


class EdgeDepthHead(nn.Module):
    """Predict a scalar patch-depth gradient for one directed edge."""

    def __init__(self, graph_dim: int = 64, direction_embed_dim: int = 16) -> None:
        super().__init__()
        input_dim = graph_dim * 3 + direction_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        zi: torch.Tensor,
        zj: torch.Tensor,
        vij: torch.Tensor,
        direction_embedding: torch.Tensor,
    ) -> torch.Tensor:
        assert zi.shape == zj.shape == vij.shape
        if direction_embedding.ndim == 1:
            direction_embedding = direction_embedding.unsqueeze(0).expand(zi.shape[0], -1)
        x = torch.cat([zi, zj, vij, direction_embedding], dim=-1)
        return self.net(x)


class BoundaryHead(nn.Module):
    """Optional boundary head. Present for later experiments, disabled by default."""

    def __init__(self, graph_dim: int = 64, direction_embed_dim: int = 16) -> None:
        super().__init__()
        input_dim = graph_dim * 3 + direction_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        zi: torch.Tensor,
        zj: torch.Tensor,
        vij: torch.Tensor,
        direction_embedding: torch.Tensor,
    ) -> torch.Tensor:
        assert zi.shape == zj.shape == vij.shape
        if direction_embedding.ndim == 1:
            direction_embedding = direction_embedding.unsqueeze(0).expand(zi.shape[0], -1)
        x = torch.cat([zi, zj, vij, direction_embedding], dim=-1)
        return self.net(x)
