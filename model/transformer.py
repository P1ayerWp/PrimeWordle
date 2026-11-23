"""Pure-PyTorch DeepLogic PrimeGPT: 32K-context, 80-vocab, zero-CoT logic Transformer.

Architecture highlights:
  - Pre-RMSNorm + QK-Norm (stabilizes 16-layer deep attention up to 32K context)
  - Rotary Positional Embeddings (RoPE, theta=500_000 for native 32K context)
  - Grouped-Query Attention (GQA) with explicit K/V repeat before SDPA so Windows
    PyTorch 2.5.1 uses O(N) memory-efficient / cuDNN attention instead of O(N^2) math
  - SwiGLU Feed-Forward Networks
  - Tied 80-token Embedding & LM Head
  - Incremental KV-Cache for millisecond inference & RL rollouts
  - O(T) last-token attention extraction hook for Web UI attention backtracking
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import PrimeGPTConfig

KVCache = list[tuple[torch.Tensor, torch.Tensor]]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Lazy-cached float32 RoPE supporting up to 32K+ context and KV-cache offsets."""

    def __init__(self, head_dim: int, theta: float = 500_000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_len: int = 0
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        self._cos_cached = freqs.cos()[None, None, :, :]  # (1, 1, T, head_dim/2)
        self._sin_cached = freqs.sin()[None, None, :, :]
        self._cached_len = seq_len

    def get_cos_sin(
        self, start_pos: int, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        needed = start_pos + seq_len
        if (
            self._cos_cached is None
            or needed > self._cached_len
            or self._cos_cached.device != device
        ):
            self._build_cache(max(needed, 2048), device)
        assert self._cos_cached is not None and self._sin_cached is not None
        return (
            self._cos_cached[:, :, start_pos:needed, :],
            self._sin_cached[:, :, start_pos:needed, :],
        )

    def get_cos_sin_from_pos(
        self, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lookup RoPE cos/sin for arbitrary per-batch position_ids of shape (B, T)."""
        needed = int(position_ids.max().item()) + 1
        device = position_ids.device
        if (
            self._cos_cached is None
            or needed > self._cached_len
            or self._cos_cached.device != device
        ):
            self._build_cache(max(needed, 2048), device)
        assert self._cos_cached is not None and self._sin_cached is not None
        # _cos_cached is (1, 1, L, D/2) -> index [0, 0, position_ids] gives (B, T, D/2)
        cos = self._cos_cached[0, 0, position_ids, :].unsqueeze(1)
        sin = self._sin_cached[0, 0, position_ids, :].unsqueeze(1)
        return cos, sin


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embedding to x of shape (B, H, T, D)."""
    x_f = x.float()
    x1 = x_f[..., 0::2]
    x2 = x_f[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack((out1, out2), dim=-1).flatten(-2).to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: PrimeGPTConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_rep
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout

        self.w_q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.w_k = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.w_v = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.w_o = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

        self.q_norm = RMSNorm(cfg.head_dim, eps=cfg.norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(cfg.head_dim, eps=cfg.norm_eps) if cfg.qk_norm else None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        return_last_attn: bool = False,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        Optional[tuple[torch.Tensor, torch.Tensor]],
        Optional[torch.Tensor],
    ]:
        bsz, seq_len, _ = x.shape

        q = self.w_q(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None

        # Explicitly expand GQA K/V so Windows PyTorch 2.5.1 triggers O(N) MEM_EFF/cuDNN SDPA
        if self.n_rep > 1:
            k_attn = k.repeat_interleave(self.n_rep, dim=1)
            v_attn = v.repeat_interleave(self.n_rep, dim=1)
        else:
            k_attn = k
            v_attn = v

        drop_p = self.dropout if self.training else 0.0
        if past_kv is None and key_padding_mask is None:
            y = F.scaled_dot_product_attention(
                q, k_attn, v_attn, dropout_p=drop_p, is_causal=True
            )
        elif seq_len == 1:
            attn_mask = (
                key_padding_mask[:, None, None, :]
                if key_padding_mask is not None
                else None
            )
            y = F.scaled_dot_product_attention(
                q, k_attn, v_attn, attn_mask=attn_mask, dropout_p=drop_p, is_causal=False
            )
        else:
            total_len = k_attn.size(2)
            past_len = total_len - seq_len
            q_pos = torch.arange(past_len, total_len, device=x.device)[:, None]
            k_pos = torch.arange(total_len, device=x.device)[None, :]
            attn_mask = (k_pos <= q_pos)[None, None, :, :]
            if key_padding_mask is not None:
                attn_mask = attn_mask & key_padding_mask[:, None, None, :]
            y = F.scaled_dot_product_attention(
                q, k_attn, v_attn, attn_mask=attn_mask, dropout_p=drop_p, is_causal=False
            )

        last_attn: Optional[torch.Tensor] = None
        if return_last_attn:
            # O(T_k) attention vector of ONLY the final query position (for UI visualization)
            q_last = q[:, :, -1:, :].float()  # (B, H, 1, D)
            scores = torch.matmul(q_last, k_attn.float().transpose(-2, -1)) / math.sqrt(
                self.head_dim
            )
            if key_padding_mask is not None:
                scores = scores.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))
            last_attn = F.softmax(scores.squeeze(2), dim=-1)  # (B, H, T_k)

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return self.w_o(y), new_kv, last_attn


class SwiGLU(nn.Module):
    def __init__(self, cfg: PrimeGPTConfig) -> None:
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: PrimeGPTConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        return_last_attn: bool = False,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        Optional[tuple[torch.Tensor, torch.Tensor]],
        Optional[torch.Tensor],
    ]:
        h, new_kv, last_attn = self.attn(
            self.norm1(x),
            cos,
            sin,
            past_kv=past_kv,
            use_cache=use_cache,
            return_last_attn=return_last_attn,
            key_padding_mask=key_padding_mask,
        )
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x, new_kv, last_attn


class PrimeGPT(nn.Module):
    """DeepLogic Mini-GPT for PrimeWordle."""

    def __init__(self, cfg: PrimeGPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing: bool = False

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(cfg.head_dim, theta=cfg.rope_theta)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        resid_std = 0.02 / math.sqrt(2.0 * self.cfg.n_layers)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        for block in self.blocks:
            for lin in (
                block.attn.w_q,
                block.attn.w_k,
                block.attn.w_v,
                block.ffn.w_gate,
                block.ffn.w_up,
            ):
                nn.init.normal_(lin.weight, mean=0.0, std=0.02)
            nn.init.normal_(block.attn.w_o.weight, mean=0.0, std=resid_std)
            nn.init.normal_(block.ffn.w_down.weight, mean=0.0, std=resid_std)
        if not self.cfg.tie_weights:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if (not trainable_only or p.requires_grad)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False,
        return_last_attn: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[KVCache], Optional[list[torch.Tensor]]]:
        """Forward pass.

        Returns:
          logits: (B, T, vocab_size)
          new_kv: list of (K, V) tuples per layer if use_cache=True else None
          last_attns: list of (B, n_heads, T_k) per layer if return_last_attn=True else None
        """
        _, seq_len = input_ids.shape
        start_pos = 0 if past_kv is None else past_kv[0][0].size(2)
        if start_pos + seq_len > self.cfg.max_seq_len:
            raise ValueError(
                f"Sequence length {start_pos + seq_len} exceeds max_seq_len={self.cfg.max_seq_len}."
            )

        x = self.tok_emb(input_ids)
        if position_ids is not None:
            cos, sin = self.rope.get_cos_sin_from_pos(position_ids)
        else:
            cos, sin = self.rope.get_cos_sin(start_pos, seq_len, input_ids.device)

        new_kv: Optional[KVCache] = [] if use_cache else None
        last_attns: Optional[list[torch.Tensor]] = [] if return_last_attn else None

        for idx, block in enumerate(self.blocks):
            layer_past = past_kv[idx] if past_kv is not None else None
            if (
                self.gradient_checkpointing
                and self.training
                and not use_cache
                and not return_last_attn
                and key_padding_mask is None
            ):
                def _ckpt_fn(
                    h: torch.Tensor,
                    c: torch.Tensor,
                    s: torch.Tensor,
                    blk: TransformerBlock = block,
                ) -> torch.Tensor:
                    out, _, _ = blk(h, c, s, past_kv=None, use_cache=False, return_last_attn=False)
                    return out

                x = checkpoint(_ckpt_fn, x, cos, sin, use_reentrant=False)
            else:
                x, layer_kv, layer_attn = block(
                    x,
                    cos,
                    sin,
                    past_kv=layer_past,
                    use_cache=use_cache,
                    return_last_attn=return_last_attn,
                    key_padding_mask=key_padding_mask,
                )
                if new_kv is not None and layer_kv is not None:
                    new_kv.append(layer_kv)
                if last_attns is not None and layer_attn is not None:
                    last_attns.append(layer_attn)

        logits = self.lm_head(self.final_norm(x))
        return logits, new_kv, last_attns


def compute_masked_lm_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Shifted next-token cross-entropy evaluated ONLY where target loss_mask == 1."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = input_ids[:, 1:].long().contiguous()
    shift_mask = loss_mask[:, 1:].float().contiguous()

    per_tok_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        reduction="none",
    ).view_as(shift_mask)

    denom = shift_mask.sum().clamp(min=1.0)
    return (per_tok_loss * shift_mask).sum() / denom
