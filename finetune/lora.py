"""LoRA (Low-Rank Adaptation) -- same three pieces as appendix-E.ipynb in the
book (LoRALayer / LinearWithLoRA / replace_linear_with_lora), reimplemented as
a plain module so it can wrap our own MalayaLM's Linear layers instead of the
book's GPT-2 clone. Lets instruction-finetuning update a small fraction of
parameters instead of the whole (few-hundred-million-parameter) model.
"""
import math

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * (x @ self.A @ self.B)


class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(module: nn.Module, rank: int, alpha: float, skip_substrings=("gate",)):
    """Recursively swap every nn.Linear for a LinearWithLoRA, freezing the
    original weight. `skip_substrings` lets you leave e.g. the MoE router
    ("gate") as a small fully-trainable layer instead of LoRA-adapting it."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not any(s in name for s in skip_substrings):
            setattr(module, name, LinearWithLoRA(child, rank, alpha))
        else:
            replace_linear_with_lora(child, rank, alpha, skip_substrings)


def freeze_base_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def count_trainable(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(total=total, trainable=trainable, pct=100 * trainable / total)
