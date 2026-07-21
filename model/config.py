"""Model configuration: a plain dataclass plus a YAML loader, so the same
model code runs at "smoke" scale (CPU, fits in a few hundred MB, for local
sanity-checking) and "shannon" scale (RTX 5090, 32GB VRAM) just by pointing
at a different config file.
"""
from dataclasses import dataclass, asdict

import yaml


@dataclass
class MalayaLMConfig:
    vocab_size: int
    context_length: int = 512
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_kv_groups: int = 2
    hidden_dim: int = 683          # dense-layer FFN intermediate size (non-MoE layers, if any)
    moe_hidden_dim: int = 512      # per-expert intermediate size
    n_experts: int = 4
    n_experts_per_tok: int = 2
    moe_every_n: int = 1           # 1 = every layer is MoE; 2 = alternate dense/MoE, etc.
    aux_loss_alpha: float = 0.01
    rope_theta: float = 10000.0
    dropout: float = 0.0
    qkv_bias: bool = False
    tie_weights: bool = True

    def to_dict(self):
        return asdict(self)


def load_config(path: str, vocab_size: int = None) -> MalayaLMConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    if vocab_size is not None:
        raw["vocab_size"] = vocab_size
    return MalayaLMConfig(**raw)


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    # "active" params = the ones actually used for a given token's forward pass:
    # all attention/norm/embedding params, but only top-k experts' worth of MoE params.
    moe_total = 0
    moe_active = 0
    for block in model.blocks:
        if getattr(block, "is_moe", False):
            per_expert = sum(p.numel() for p in block.ffn.experts[0].parameters())
            moe_total += per_expert * block.ffn.n_experts
            moe_active += per_expert * block.ffn.n_experts_per_tok
    dense_total = total - moe_total
    active = dense_total + moe_active
    return dict(total=total, active=active, moe_total=moe_total, moe_active=moe_active)
