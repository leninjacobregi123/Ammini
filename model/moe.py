"""Sparse Mixture-of-Experts feed-forward layer (Mixtral/DeepSeek-style top-k
routing over SwiGLU experts), with a Switch-Transformer-style load-balancing
auxiliary loss so the router doesn't collapse onto a handful of experts.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import SwiGLU


class MoEFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, n_experts: int,
                 n_experts_per_tok: int, aux_loss_alpha: float = 0.01):
        super().__init__()
        assert 0 < n_experts_per_tok <= n_experts
        self.n_experts = n_experts
        self.n_experts_per_tok = n_experts_per_tok
        self.aux_loss_alpha = aux_loss_alpha

        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d_model, hidden_dim) for _ in range(n_experts)])

        # populated on every forward() call; read by the model to add to the loss
        self.last_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        x_flat = x.reshape(-1, d)

        router_logits = self.gate(x_flat)                                   # (n_tok, n_experts)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(x.dtype)
        topk_probs, topk_idx = router_probs.topk(self.n_experts_per_tok, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)       # renormalize over the top-k

        out = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            token_rows, k_slot = (topk_idx == expert_id).nonzero(as_tuple=True)
            if token_rows.numel() == 0:
                continue
            expert_out = expert(x_flat[token_rows])
            weight = topk_probs[token_rows, k_slot].unsqueeze(-1)
            out.index_add_(0, token_rows, expert_out * weight)

        out = out.view(b, t, d)

        # load-balancing aux loss: encourage uniform routing probability mass
        # (density) and uniform selection frequency (importance) across experts.
        density = router_probs.mean(dim=0)
        selected_one_hot = F.one_hot(topk_idx, num_classes=self.n_experts).sum(dim=1).to(router_probs.dtype)
        importance = selected_one_hot.mean(dim=0)
        aux_loss = self.n_experts * (density * importance).sum()
        self.last_aux_loss = self.aux_loss_alpha * aux_loss

        return out
