"""MalayaLM: a from-scratch Llama-style decoder with sparse MoE feed-forward
layers -- the architectural equivalent of ch04's GPTModel, upgraded with the
RoPE + RMSNorm + SwiGLU + GQA + MoE stack the current generation of small
open models (Llama/Mistral/Qwen/Mixtral-class) actually use.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import MalayaLMConfig
from model.layers import RMSNorm, GroupedQueryAttention, SwiGLU, precompute_rope
from model.moe import MoEFeedForward


class TransformerBlock(nn.Module):
    def __init__(self, cfg: MalayaLMConfig, use_moe: bool):
        super().__init__()
        self.is_moe = use_moe
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = GroupedQueryAttention(
            cfg.d_model, cfg.n_heads, cfg.n_kv_groups, cfg.context_length,
            cfg.dropout, cfg.qkv_bias,
        )
        self.ffn_norm = RMSNorm(cfg.d_model)
        if use_moe:
            self.ffn = MoEFeedForward(
                cfg.d_model, cfg.moe_hidden_dim, cfg.n_experts,
                cfg.n_experts_per_tok, cfg.aux_loss_alpha,
            )
        else:
            self.ffn = SwiGLU(cfg.d_model, cfg.hidden_dim)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MalayaLM(nn.Module):
    def __init__(self, cfg: MalayaLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, use_moe=(cfg.n_experts > 0 and i % cfg.moe_every_n == 0))
            for i in range(cfg.n_layers)
        ])
        self.final_norm = RMSNorm(cfg.d_model)
        self.out_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.out_head.weight = self.tok_emb.weight

        head_dim = cfg.d_model // cfg.n_heads
        cos, sin = precompute_rope(head_dim, cfg.context_length, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        b, t = idx.shape
        if t > self.cfg.context_length:
            raise ValueError(f"sequence length {t} exceeds context_length {self.cfg.context_length}")

        x = self.tok_emb(idx)
        aux_losses = []
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
            if block.is_moe:
                aux_losses.append(block.ffn.last_aux_loss)
        x = self.final_norm(x)
        logits = self.out_head(x)

        loss = None
        if targets is not None:
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100,
            )
            aux = torch.stack(aux_losses).sum() if aux_losses else logits.new_zeros(())
            loss = ce + aux
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int = 40, eos_id: int = None):
        """Simple recompute-based sampling loop (no KV cache -- fine for a
        demo-scale chat app; a KV cache is the natural next optimization)."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

            if eos_id is not None and next_id.item() == eos_id:
                break
        return idx
