from typing import Dict

import torch
from torch import nn


def trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    state = model.state_dict()
    return {name: tensor.detach().cpu() for name, tensor in state.items() if name in trainable_names}


def save_trainable_checkpoint(model: nn.Module, path: str, extra: Dict | None = None) -> None:
    payload = {"model": trainable_state_dict(model)}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def save_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    path: str,
    extra: Dict | None = None,
) -> None:
    payload = {
        "model": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_trainable_checkpoint(model: nn.Module, path: str, map_location: str | torch.device = "cpu") -> Dict:
    payload = torch.load(path, map_location=map_location)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {"missing": missing, "unexpected": unexpected, "payload": payload}


def load_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    path: str,
    map_location: str | torch.device = "cpu",
) -> Dict:
    info = load_trainable_checkpoint(model, path, map_location=map_location)
    payload = info["payload"]
    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    return info
