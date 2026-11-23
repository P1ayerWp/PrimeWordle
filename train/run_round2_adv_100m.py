"""Round-2 Adversarial ("使绊子") 100M-Token Fine-Tuning Launcher for DeepLogic 25M.

Step 1: Generate 2 x 50M-token disjoint adversarial shards (`data/sft/adv_100m/adv_00..01.pt`):
          - 40% Dirty mid-game human takeovers (loss_mask=0 on dirty prefix)
          - 35% Random sparse forbidden-fusion graphs (30%..70% bans, heavy UNDERFILL_OFF + p*p bans)
          - 25% Clean full-universe anti-forgetting mix (boosted wide-P K=1 Wordle & K=6..8)
Step 2: Initialize weights from `checkpoints/deeplogic_400m/best.pt` (fresh AdamW +
        fresh 100M Cosine LR at 25% peak LR: 1.5e-4 -> 1.5e-5, warmup 3M tokens)
        and train pure 1 Epoch, saving checkpoints to `checkpoints/deeplogic_adv100m/`.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

from solver.adversarial_dataset import DEFAULT_ADV_SEED_BASE, generate_adversarial_shards
from train.train_sft import train_sft


def _set_windows_keep_awake(enable: bool) -> None:
    if sys.platform != "win32":
        return
    try:
        flags = (0x80000000 | 0x00000001) if enable else 0x80000000
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Round-2 100M-token adversarial fine-tuning.")
    parser.add_argument("--init-from", type=Path, default=Path("checkpoints/deeplogic_400m/best.pt"))
    parser.add_argument("--shard-dir", type=Path, default=Path("data/sft/adv_100m"))
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--tokens-per-shard", type=int, default=50_000_000)
    parser.add_argument("--gen-workers", type=int, default=12)
    parser.add_argument("--val-shard", type=Path, default=Path("data/sft/val_600.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/deeplogic_adv100m"))
    parser.add_argument("--eval-every-tokens", type=int, default=25_000_000)
    parser.add_argument("--play-eval-games", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--min-lr", type=float, default=1.5e-5)
    parser.add_argument("--warmup-tokens", type=int, default=3_000_000)
    parser.add_argument("--skip-gen", action="store_true")
    args = parser.parse_args()

    _set_windows_keep_awake(True)
    try:
        if not args.skip_gen:
            generate_adversarial_shards(
                shard_dir=args.shard_dir,
                num_shards=args.num_shards,
                tokens_per_shard=args.tokens_per_shard,
                seed_base=DEFAULT_ADV_SEED_BASE,
                start_shard=0,
                workers=args.gen_workers,
            )

        existing_shards = sorted(args.shard_dir.glob("adv_*.pt"))
        if not existing_shards:
            raise FileNotFoundError(f"No adv shards found in {args.shard_dir}")

        latest_ckpt = args.out_dir / "latest.pt"
        total_budget = args.num_shards * args.tokens_per_shard

        train_args = argparse.Namespace(
            preset="deeplogic_25m",
            train_shards=[str(p) for p in existing_shards],
            val_shards=[str(args.val_shard)] if args.val_shard.exists() else [],
            out_dir=str(args.out_dir),
            init_from=None if latest_ckpt.exists() else str(args.init_from),
            resume=str(latest_ckpt) if latest_ckpt.exists() else None,
            epochs=1,
            total_tokens_budget=total_budget,
            warmup_tokens=args.warmup_tokens,
            max_steps=0,
            max_minutes=0.0,
            micro_tokens=8192,
            max_batch_size=64,
            grad_accum=8,
            ckpt_threshold=1024,
            lr=args.lr,
            min_lr=args.min_lr,
            weight_decay=0.1,
            grad_clip=1.0,
            log_every=20,
            eval_every_tokens=args.eval_every_tokens,
            eval_every_steps=0,
            play_eval_games=args.play_eval_games,
            seed=2026,
        )
        train_sft(train_args)
    finally:
        _set_windows_keep_awake(False)


if __name__ == "__main__":
    main()
