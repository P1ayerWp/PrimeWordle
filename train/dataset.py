"""Length-bucketed PyTorch Dataset & Token-Budget BatchSampler for PrimeWordle shards.

Why token-budget bucketing?
  PrimeWordle trajectories span from ~60 tokens (short K=1 games) to ~15,000+
  tokens (extreme P=32, K=8, L=32 games). Grouping similar-length games under a
  fixed `max_tokens_per_batch` cap achieves:
    - <10% right-padding waste
    - Exact causal semantics with zero custom attention mask (pure O(N) SDPA!)
    - Automatic OOM protection on 8GB VRAM (short games batch large; 16K-32K run B=1)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch.utils.data import Dataset, Sampler


class ShardEpisodeDataset(Dataset):
    """Loads one or more uint8 `.pt` shards produced by `solver.generate_dataset`."""

    def __init__(self, shard_paths: Sequence[str | Path]) -> None:
        self.shards: list[dict[str, torch.Tensor]] = []
        self.index: list[tuple[int, int, int, int]] = []  # (shard_idx, start, end, length)
        self.vocab_size: int = 80

        for s_idx, path in enumerate(shard_paths):
            data = torch.load(path, map_location="cpu", weights_only=True)
            self.vocab_size = int(data["vocab_size"])
            ids: torch.Tensor = data["input_ids"]
            mask: torch.Tensor = data["loss_mask"]
            offsets: torch.Tensor = data["offsets"]
            self.shards.append({"ids": ids, "mask": mask})

            off_list = offsets.tolist()
            for i in range(len(off_list) - 1):
                st, ed = off_list[i], off_list[i + 1]
                length = ed - st
                if length >= 2:
                    self.index.append((s_idx, st, ed, length))

        self.lengths: list[int] = [item[3] for item in self.index]
        self.total_tokens: int = sum(self.lengths)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s_idx, st, ed, _ = self.index[idx]
        shard = self.shards[s_idx]
        return (
            shard["ids"][st:ed].to(dtype=torch.long),
            shard["mask"][st:ed].to(dtype=torch.uint8),
        )


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Dynamic length-bucketed batch sampler enforcing `B * max_len <= max_tokens`."""

    def __init__(
        self,
        lengths: Sequence[int],
        max_tokens: int = 16_384,
        max_batch_size: int = 64,
        shuffle: bool = True,
        noise_scale: float = 0.08,
        seed: int = 42,
    ) -> None:
        self.lengths = list(lengths)
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.noise_scale = noise_scale
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _build_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1009)
        n = len(self.lengths)
        if self.shuffle:
            # Sort by length + slight multiplicative jitter so batches stay tight
            # while varying every epoch
            order = sorted(
                range(n),
                key=lambda i: self.lengths[i]
                * (1.0 + rng.uniform(-self.noise_scale, self.noise_scale)),
            )
        else:
            order = sorted(range(n), key=lambda i: self.lengths[i])

        batches: list[list[int]] = []
        cur: list[int] = []
        cur_max_len = 0

        for idx in order:
            seq_len = self.lengths[idx]
            new_max = max(cur_max_len, seq_len)
            new_tokens = new_max * (len(cur) + 1)
            if cur and (
                new_tokens > self.max_tokens or len(cur) >= self.max_batch_size
            ):
                batches.append(cur)
                cur = [idx]
                cur_max_len = seq_len
            else:
                cur.append(idx)
                cur_max_len = new_max

        if cur:
            batches.append(cur)

        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._build_batches()

    def __len__(self) -> int:
        return len(self._build_batches())


def collate_episodes(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Right-pad variable-length episodes to the batch's max length (aligned to 8)."""
    bsz = len(batch)
    raw_max = max(item[0].numel() for item in batch)
    # Round up to multiple of 8 for Tensor Core friendliness
    max_len = ((raw_max + 7) // 8) * 8

    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    loss_mask = torch.zeros((bsz, max_len), dtype=torch.uint8)
    real_tokens = 0

    for i, (ids, mask) in enumerate(batch):
        length = ids.numel()
        input_ids[i, :length] = ids
        loss_mask[i, :length] = mask
        real_tokens += length

    return input_ids, loss_mask, real_tokens
