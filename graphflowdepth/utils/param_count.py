from typing import Dict

from torch import nn


def count_parameters(model: nn.Module) -> Dict[str, int | float]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = total - trainable
    trainable_pct = 100.0 * trainable / total if total else 0.0
    return {
        "total": total,
        "frozen": frozen,
        "trainable": trainable,
        "trainable_pct": trainable_pct,
    }


def format_parameter_report(model: nn.Module) -> str:
    counts = count_parameters(model)
    return (
        f"total parameters: {counts['total']:,}\n"
        f"frozen parameters: {counts['frozen']:,}\n"
        f"trainable parameters: {counts['trainable']:,}\n"
        f"trainable percentage: {counts['trainable_pct']:.4f}%"
    )
