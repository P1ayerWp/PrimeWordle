"""Idempotent overnight launcher for 400M-token pure 1-Epoch DeepLogic PrimeGPT SFT.

Step 1: Ensures all 8 disjoint 50M-token shards (`data/sft/run_400m/shard_00..07.pt`)
        exist, generating any missing shards with 12 CPU workers.
Step 2: Trains `deeplogic_25m` (16L, 25.21M params, 80 vocab, 32K RoPE) for 1 pure
        Epoch over all 400M tokens with Cosine LR calibrated to 400M tokens,
        running tiered eval_play + X-ray mine diagnostics every 40M tokens (~25 min)
        and automatically resuming from `latest.pt` if present.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

from solver.generate_dataset import DEFAULT_TRAIN_SEED_BASE, generate_sharded_dataset
from train.train_sft import train_sft


def _set_windows_keep_awake(enable: bool) -> None:
    """Keep Windows system awake during overnight training (screen may still turn off)."""
    if sys.platform != "win32":
        return
    try:
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        flags = (ES_CONTINUOUS | ES_SYSTEM_REQUIRED) if enable else ES_CONTINUOUS
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 400M-token 1-Epoch DeepLogic SFT.")
    parser.add_argument("--shard-dir", type=Path, default=Path("data/sft/run_400m"))
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--tokens-per-shard", type=int, default=50_000_000)
    parser.add_argument("--gen-workers", type=int, default=12)
    parser.add_argument("--val-shard", type=Path, default=Path("data/sft/val_600.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/deeplogic_400m"))
    parser.add_argument("--eval-every-tokens", type=int, default=40_000_000)
    parser.add_argument("--play-eval-games", type=int, default=8)
    parser.add_argument("--max-minutes", type=float, default=0.0, help="If >0, pause cleanly after M minutes.")
    parser.add_argument("--skip-gen", action="store_true", help="Train on existing shards only.")
    args = parser.parse_args()

    _set_windows_keep_awake(True)
    try:
        # Step 1: Generate any missing shards
        if not args.skip_gen:
            for s_idx in range(args.num_shards):
                pt_path = args.shard_dir / f"shard_{s_idx:02d}.pt"
                if pt_path.exists():
                    print(f"[Skip] {pt_path} already exists ({pt_path.stat().st_size/1024**2:.1f} MB)", flush=True)
                    continue
                generate_sharded_dataset(
                    shard_dir=args.shard_dir,
                    num_shards=1,
                    tokens_per_shard=args.tokens_per_shard,
                    seed_base=DEFAULT_TRAIN_SEED_BASE,
                    start_shard=s_idx,
                    workers=args.gen_workers,
                )

        existing_shards = sorted(args.shard_dir.glob("shard_*.pt"))
        if not existing_shards:
            raise FileNotFoundError(f"No shards found in {args.shard_dir}")

        latest_ckpt = args.out_dir / "latest.pt"
        total_budget = args.num_shards * args.tokens_per_shard

        train_args = argparse.Namespace(
            preset="deeplogic_25m",
            train_shards=[str(p) for p in existing_shards],
            val_shards=[str(args.val_shard)] if args.val_shard.exists() else [],
            out_dir=str(args.out_dir),
            resume=str(latest_ckpt) if latest_ckpt.exists() else None,
            epochs=1,
            total_tokens_budget=total_budget,
            warmup_tokens=8_000_000,
            max_steps=0,
            max_minutes=args.max_minutes,
            micro_tokens=8192,
            max_batch_size=64,
            grad_accum=8,
            ckpt_threshold=1024,
            lr=6e-4,
            min_lr=6e-5,
            weight_decay=0.1,
            grad_clip=1.0,
            log_every=20,
            eval_every_tokens=args.eval_every_tokens,
            eval_every_steps=0,
            play_eval_games=args.play_eval_games,
            seed=42,
        )
        train_sft(train_args)
    finally:
        _set_windows_keep_awake(False)


if __name__ == "__main__":
    main()
