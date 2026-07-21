"""Core building blocks: RMSNorm, RoPE, Grouped-Query Attention, SwiGLU.

Same role as ch03/ch04 in the book (attention + GPT block internals), but
using the architecture Llama/Mistral/Qwen-class models actually use instead
of the book's plain learned-position + standard-MHA + GELU-MLP GPT-2 clone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).to(dtype)


def precompute_rope(head_dim: int, max_seq_len: int, theta_base: float = 10000.0):
    """Precompute cos/sin tables for rotary position embeddings."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)          # (max_seq_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)            # (max_seq_len, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (batch, n_heads, seq_len, head_dim); cos/sin: (max_seq_len, head_dim)."""
    seq_len = x.size(-2)
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype, device=x.device)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype, device=x.device)
    return x * cos + _rotate_half(x) * sin


class GroupedQueryAttention(nn.Module):
    """Multi-head attention with fewer key/value heads than query heads (GQA),
    the same trick Llama 3 / Mistral / Qwen2 use to shrink the KV cache."""

    def __init__(self, d_model: int, n_heads: int, n_kv_groups: int,
                 context_length: int, dropout: float = 0.0, qkv_bias: bool = False):
        super().__init__()
        assert n_heads % n_kv_groups == 0, "n_heads must be divisible by n_kv_groups"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.group_size = n_heads // n_kv_groups
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.context_length = context_length

        self.W_query = nn.Linear(d_model, n_heads * self.head_dim, bias=qkv_bias)
        self.W_key = nn.Linear(d_model, n_kv_groups * self.head_dim, bias=qkv_bias)
        self.W_value = nn.Linear(d_model, n_kv_groups * self.head_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.W_query(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(b, t, self.n_kv_groups, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(b, t, self.n_kv_groups, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        return self.out_proj(attn_out)


class SwiGLU(nn.Module):
    """Gated feed-forward used by every Llama-family model; also doubles as
    a single "expert" inside the MoE layer in model/moe.py."""

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
