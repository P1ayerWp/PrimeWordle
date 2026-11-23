"""Model configuration and presets for PrimeGPT (pure-logic Decoder-only Transformer)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PrimeGPTConfig:
    vocab_size: int = 80
    max_seq_len: int = 32_768
    n_layers: int = 16
    d_model: int = 384
    n_heads: int = 6
    n_kv_heads: int = 2
    d_ff: int = 1024
    rope_theta: float = 500_000.0
    norm_eps: float = 1e-6
    qk_norm: bool = True
    dropout: float = 0.0
    tie_weights: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads.")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_rep(self) -> int:
        return self.n_heads // self.n_kv_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrimeGPTConfig":
        return cls(**data)


# Official architecture presets (all aligned to 80-token vocab & 32K RoPE context)
PRESETS: dict[str, PrimeGPTConfig] = {
    "compact_9m": PrimeGPTConfig(
        n_layers=12,
        d_model=256,
        n_heads=4,
        n_kv_heads=2,
        d_ff=704,
    ),
    "sweetspot_19m": PrimeGPTConfig(
        n_layers=12,
        d_model=384,
        n_heads=6,
        n_kv_heads=2,
        d_ff=1024,
    ),
    "deeplogic_25m": PrimeGPTConfig(
        n_layers=16,
        d_model=384,
        n_heads=6,
        n_kv_heads=2,
        d_ff=1024,
    ),
}
