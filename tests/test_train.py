"""Unit tests for length-bucketed dataset, batch sampler, and KV-cache gameplay evaluator."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from engine.game import GameConfig
from model import PrimeGPT, PrimeGPTConfig
from train.dataset import ShardEpisodeDataset, TokenBudgetBatchSampler, collate_episodes
from train.eval_play import play_game_with_model, play_games_batched


class TestTrainPipeline(unittest.TestCase):
    def test_token_budget_sampler_and_collate_padding(self) -> None:
        lengths = [50, 52, 55, 60, 400, 410, 420, 4000, 12000]
        sampler = TokenBudgetBatchSampler(
            lengths, max_tokens=1000, max_batch_size=8, shuffle=True, seed=7
        )
        batches = list(sampler)
        seen = [idx for b in batches for idx in b]
        self.assertEqual(sorted(seen), list(range(len(lengths))))

        for b in batches:
            max_l = max(lengths[i] for i in b)
            if len(b) > 1:
                self.assertLessEqual(max_l * len(b), 1000)

        fake_batch = [
            (torch.arange(1, 11, dtype=torch.long), torch.ones(10, dtype=torch.uint8)),
            (torch.arange(1, 15, dtype=torch.long), torch.ones(14, dtype=torch.uint8)),
        ]
        ids, mask, real_toks = collate_episodes(fake_batch, pad_id=0)
        self.assertEqual(real_toks, 24)
        # 14 rounded up to multiple of 8 is 16
        self.assertEqual(tuple(ids.shape), (2, 16))
        self.assertEqual(int(ids[0, 10:].sum().item()), 0)
        self.assertEqual(int(mask[0, 10:].sum().item()), 0)
        self.assertEqual(int(mask[1, 14:].sum().item()), 0)

    def test_shard_loader_and_live_play_eval(self) -> None:
        smoke_pt = Path("data/sft/smoke_500.pt")
        if smoke_pt.exists():
            ds = ShardEpisodeDataset([smoke_pt])
            self.assertEqual(len(ds), 500)
            self.assertEqual(ds.vocab_size, 80)
            ids, mask = ds[0]
            self.assertEqual(ids.dtype, torch.long)
            self.assertEqual(ids.shape, mask.shape)

        # Verify batched KV-cache play matches B=1 play identically even with underfill=True
        tiny_cfg = PrimeGPTConfig(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128)
        tiny_model = PrimeGPT(tiny_cfg).eval()
        cfg_u = GameConfig(num_primes=5, max_multiplier=2, code_length=3, allow_underfill=True)
        seeds = [123, 124, 125]
        single = [
            play_game_with_model(
                tiny_model, cfg_u, seed=s, device=torch.device("cpu"), max_turns=4
            )
            for s in seeds
        ]
        batched = play_games_batched(
            tiny_model, cfg_u, seeds=seeds, device=torch.device("cpu"), max_turns=4
        )
        self.assertEqual(batched, single)


if __name__ == "__main__":
    unittest.main()
