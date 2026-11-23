"""Full-universe SFT trajectory dataset generator for PrimeWordle (80-token vocab).

Samples configurations across the ENTIRE agreed parameter space from Day 1:
    P in [2, 32], K in [1, 8], L in [1, 32],
    allow_underfill in {True, False}, allow_duplicate_slots in {True, False},
    plus optional <UNK_CFG> blind-header masking
so all 79 active non-<PAD> tokens in the 80-token vocabulary receive balanced
gradient coverage without cold embeddings.

Supports:
  - Single-file JSONL + `.pt` export (`--games`, `--out`, `--pt-out`)
  - High-speed multi-shard `.pt`-only generation by token budget
    (`--shard-dir`, `--num-shards`, `--target-tokens`, `--workers 12`, `--no-jsonl`)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from engine.game import GameConfig, GameSession
from engine.tokenizer import PrimeWordleTokenizer

from .constraints import count_slot_candidates
from .hybrid import HybridSolver, solve_session

# Benchmark seeds live in [0, 999_999]; val seeds ~9_000_000; train seeds >= 10_000_000.
DEFAULT_TRAIN_SEED_BASE = 10_000_000


def _weighted_int(rng: random.Random, lo: int, hi: int, power: float = 0.0) -> int:
    """Sample integer in [lo, hi] with weight proportional to value**power."""
    vals = list(range(lo, hi + 1))
    if power == 0.0 or len(vals) == 1:
        return rng.choice(vals)
    weights = [v ** power for v in vals]
    return rng.choices(vals, weights=weights, k=1)[0]


def sample_full_universe_config(rng: random.Random) -> tuple[GameConfig, str]:
    """Stratified sampler covering P in [2,32], K in [1,8], L in [1,32]."""
    u = rng.random()
    if u < 0.30:
        tier = "T1_core"
        p = _weighted_int(rng, 2, 16, power=0.35)
        k = rng.randint(1, 3)
        l = rng.randint(1, 8)
    elif u < 0.60:
        tier = "T2_wide_P"
        p = _weighted_int(rng, 17, 32, power=0.4)
        k = rng.randint(1, 4)
        l = rng.randint(2, 12)
    elif u < 0.85:
        tier = "T3_high_K"
        p = _weighted_int(rng, 4, 32, power=0.5)
        k = _weighted_int(rng, 4, 8, power=-0.3)
        l = rng.randint(1, 12)
    else:
        tier = "T4_wide_L"
        p = _weighted_int(rng, 4, 32, power=0.45)
        k = _weighted_int(rng, 1, 8, power=-0.55)
        l = _weighted_int(rng, 13, 32, power=0.2)

    allow_underfill = False if k == 1 else (rng.random() < 0.65)

    probe_cfg = GameConfig(
        num_primes=p,
        max_multiplier=k,
        code_length=l,
        allow_underfill=allow_underfill,
        allow_duplicate_slots=True,
    )
    can_disable_dup = l <= count_slot_candidates(probe_cfg)
    allow_duplicate_slots = not (can_disable_dup and rng.random() < 0.30)

    config = GameConfig(
        num_primes=p,
        max_multiplier=k,
        code_length=l,
        allow_underfill=allow_underfill,
        allow_duplicate_slots=allow_duplicate_slots,
        max_turns=None,
    )
    return config, tier


def _worker_generate_chunk(task: dict[str, Any]) -> dict[str, Any]:
    """Generate a deterministic chunk of won trajectories (by game count OR token budget)."""
    target_games: int = task["num_games"]
    target_tokens: int = task["target_tokens"]
    emit_jsonl: bool = task["emit_jsonl"]
    seed_base: int = task["seed_base"]
    diversity: float = task["diversity"]
    blind_k_prob: float = task["blind_k_prob"]
    blind_underfill_prob: float = task["blind_underfill_prob"]
    blind_dup_prob: float = task["blind_dup_prob"]
    max_tokens: int = task["max_tokens"]
    max_steps: int = task["max_steps"]

    tokenizer = PrimeWordleTokenizer(max_symbols=32)
    cfg_rng = random.Random(seed_base)

    # Compact binary buffers for fast zero-overhead IPC
    flat_ids = bytearray()
    flat_mask = bytearray()
    lengths: list[int] = []
    json_lines: list[str] = []

    token_counts = [0] * tokenizer.vocab_size
    tier_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    dup_counts: Counter[str] = Counter()
    turn_counts: list[int] = []

    skipped_unwon = 0
    skipped_too_long = 0
    attempt = 0
    accepted = 0
    total_toks = 0

    def _done() -> bool:
        if target_tokens > 0:
            return total_toks >= target_tokens
        return accepted >= target_games

    while not _done():
        attempt += 1
        game_seed = seed_base + attempt
        config, tier = sample_full_universe_config(cfg_rng)
        session = GameSession(config, seed=game_seed)
        solver = HybridSolver(seed=game_seed * 3 + 7, diversity=diversity)
        res = solve_session(session, solver=solver, max_steps=max_steps)

        if not res.won:
            skipped_unwon += 1
            continue

        reveal_k = cfg_rng.random() >= blind_k_prob
        reveal_underfill = cfg_rng.random() >= blind_underfill_prob
        reveal_dup = cfg_rng.random() >= blind_dup_prob
        ids, mask = tokenizer.encode_session_with_loss_mask(
            session,
            reveal_k=reveal_k,
            reveal_underfill=reveal_underfill,
            reveal_dup=reveal_dup,
        )
        n_tok = len(ids)
        if n_tok > max_tokens:
            skipped_too_long += 1
            continue

        flat_ids.extend(ids)
        flat_mask.extend(mask)
        lengths.append(n_tok)
        for tid in ids:
            token_counts[tid] += 1

        tier_counts[tier] += 1
        backend_counts[res.backend] += 1
        dup_counts["dup_on" if config.allow_duplicate_slots else "dup_off"] += 1
        turn_counts.append(res.turns)

        if emit_jsonl:
            record = session.to_dict(include_target=True)
            record["meta"] = {
                "tier": tier,
                "solver_backend": res.backend,
                "reveal_k": reveal_k,
                "reveal_underfill": reveal_underfill,
                "reveal_dup": reveal_dup,
                "num_tokens": n_tok,
                "num_action_tokens": sum(mask),
            }
            json_lines.append(json.dumps(record, ensure_ascii=False))

        accepted += 1
        total_toks += n_tok

    return {
        "flat_ids": bytes(flat_ids),
        "flat_mask": bytes(flat_mask),
        "lengths": lengths,
        "json_lines": json_lines,
        "token_counts": token_counts,
        "tier_counts": dict(tier_counts),
        "backend_counts": dict(backend_counts),
        "dup_counts": dict(dup_counts),
        "turn_counts": turn_counts,
        "attempts": attempt,
        "skipped_unwon": skipped_unwon,
        "skipped_too_long": skipped_too_long,
    }


def generate_dataset(
    num_games: int = 0,
    out_jsonl: Path | None = None,
    out_pt: Path | None = None,
    target_tokens: int = 0,
    seed_base: int = DEFAULT_TRAIN_SEED_BASE,
    diversity: float = 0.08,
    blind_k_prob: float = 0.10,
    blind_underfill_prob: float = 0.10,
    blind_dup_prob: float = 0.10,
    max_tokens: int = 32_768,
    max_steps: int = 250,
    workers: int = 1,
) -> dict[str, Any]:
    """Generate won trajectories by game count OR token budget and export JSONL / `.pt`."""
    if target_tokens <= 0 and num_games <= 0:
        raise ValueError("Specify either num_games > 0 or target_tokens > 0.")

    tokenizer = PrimeWordleTokenizer(max_symbols=32)
    emit_jsonl = out_jsonl is not None
    if out_jsonl is not None:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if out_pt is not None:
        out_pt.parent.mkdir(parents=True, exist_ok=True)

    max_w = max(1, workers)
    n_workers = max_w if target_tokens > 0 else max(1, min(max_w, num_games))

    t0 = time.perf_counter()

    tasks: list[dict[str, Any]] = []
    for w in range(n_workers):
        if target_tokens > 0:
            chunk_toks = target_tokens // n_workers + (1 if w < (target_tokens % n_workers) else 0)
            chunk_games = 0
            if chunk_toks <= 0:
                continue
        else:
            chunk_games = num_games // n_workers + (1 if w < (num_games % n_workers) else 0)
            chunk_toks = 0
            if chunk_games <= 0:
                continue

        tasks.append(
            {
                "num_games": chunk_games,
                "target_tokens": chunk_toks,
                "emit_jsonl": emit_jsonl,
                "seed_base": seed_base + w * 2_000_000,
                "diversity": diversity,
                "blind_k_prob": blind_k_prob,
                "blind_underfill_prob": blind_underfill_prob,
                "blind_dup_prob": blind_dup_prob,
                "max_tokens": max_tokens,
                "max_steps": max_steps,
            }
        )

    if len(tasks) == 1:
        chunk_results = [_worker_generate_chunk(tasks[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
            chunk_results = list(pool.map(_worker_generate_chunk, tasks))

    token_counts = [0] * tokenizer.vocab_size
    tier_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    dup_counts: Counter[str] = Counter()
    seq_lengths: list[int] = []
    turn_counts: list[int] = []

    total_attempts = 0
    skipped_unwon = 0
    skipped_too_long = 0

    for chunk in chunk_results:
        total_attempts += chunk["attempts"]
        skipped_unwon += chunk["skipped_unwon"]
        skipped_too_long += chunk["skipped_too_long"]
        for i, c in enumerate(chunk["token_counts"]):
            token_counts[i] += c
        tier_counts.update(chunk["tier_counts"])
        backend_counts.update(chunk["backend_counts"])
        dup_counts.update(chunk["dup_counts"])
        seq_lengths.extend(chunk["lengths"])
        turn_counts.extend(chunk["turn_counts"])

    accepted = len(seq_lengths)
    total_toks = sum(seq_lengths)

    if out_jsonl is not None:
        with out_jsonl.open("w", encoding="utf-8") as fout:
            for chunk in chunk_results:
                for line in chunk["json_lines"]:
                    fout.write(line + "\n")

    if out_pt is not None:
        import torch

        ids_buf = bytearray(total_toks)
        mask_buf = bytearray(total_toks)
        offsets: list[int] = [0]
        pos = 0
        for chunk in chunk_results:
            b_ids = chunk["flat_ids"]
            b_mask = chunk["flat_mask"]
            n_b = len(b_ids)
            ids_buf[pos : pos + n_b] = b_ids
            mask_buf[pos : pos + n_b] = b_mask
            for length in chunk["lengths"]:
                offsets.append(offsets[-1] + length)
            pos += n_b

        payload = {
            "vocab_size": tokenizer.vocab_size,
            "itos": tokenizer.itos,
            "num_games": accepted,
            "input_ids": torch.frombuffer(ids_buf, dtype=torch.uint8).clone(),
            "loss_mask": torch.frombuffer(mask_buf, dtype=torch.uint8).clone(),
            "offsets": torch.tensor(offsets, dtype=torch.int64),
        }
        torch.save(payload, out_pt)

    elapsed = time.perf_counter() - t0

    non_pad_ids = [i for i in range(tokenizer.vocab_size) if tokenizer.itos[i] != "<PAD>"]
    missing_tokens = [tokenizer.itos[i] for i in non_pad_ids if token_counts[i] == 0]
    symbol_hits = {f"S_{i}": token_counts[tokenizer.stoi[f"S_{i}"]] for i in range(32)}
    number_hits = {f"N_{i}": token_counts[tokenizer.stoi[f"N_{i}"]] for i in range(1, 33)}

    sorted_lens = sorted(seq_lengths)
    summary = {
        "accepted_games": accepted,
        "attempts": total_attempts,
        "workers": len(tasks),
        "skipped_unwon": skipped_unwon,
        "skipped_too_long": skipped_too_long,
        "elapsed_sec": round(elapsed, 2),
        "games_per_sec": round(accepted / max(elapsed, 1e-6), 1),
        "tokens_per_sec": round(total_toks / max(elapsed, 1e-6), 1),
        "total_tokens": total_toks,
        "avg_tokens": round(statistics.mean(seq_lengths), 1) if seq_lengths else 0.0,
        "p95_tokens": sorted_lens[int(0.95 * (len(sorted_lens) - 1))] if sorted_lens else 0,
        "max_tokens": max(seq_lengths) if seq_lengths else 0,
        "avg_turns": round(statistics.mean(turn_counts), 2) if turn_counts else 0.0,
        "max_turns": max(turn_counts) if turn_counts else 0,
        "tiers": dict(sorted(tier_counts.items())),
        "backends": dict(sorted(backend_counts.items())),
        "dup_modes": dict(sorted(dup_counts.items())),
        "vocab_size": tokenizer.vocab_size,
        "active_tokens_covered": f"{len(non_pad_ids) - len(missing_tokens)}/{len(non_pad_ids)}",
        "missing_non_pad_tokens": missing_tokens,
        "min_symbol_freq": min(symbol_hits.items(), key=lambda kv: kv[1]),
        "max_symbol_freq": max(symbol_hits.items(), key=lambda kv: kv[1]),
        "min_number_freq": min(number_hits.items(), key=lambda kv: kv[1]),
        "max_number_freq": max(number_hits.items(), key=lambda kv: kv[1]),
        "out_jsonl": str(out_jsonl) if out_jsonl is not None else None,
        "out_pt": str(out_pt) if out_pt is not None else None,
    }
    return summary


def generate_sharded_dataset(
    shard_dir: Path,
    num_shards: int,
    tokens_per_shard: int,
    seed_base: int = DEFAULT_TRAIN_SEED_BASE,
    start_shard: int = 0,
    workers: int = 12,
    diversity: float = 0.08,
    blind_k_prob: float = 0.10,
    blind_underfill_prob: float = 0.10,
    blind_dup_prob: float = 0.10,
    max_tokens: int = 32_768,
    max_steps: int = 250,
) -> list[dict[str, Any]]:
    """Generate `num_shards` disjoint uint8 `.pt` shards (`shard_00.pt`, ...)."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    # Stride 50_000_000 seeds per shard (> 12 workers * 2_000_000 = 24M) -> 100% disjoint
    shard_seed_stride = 50_000_000

    for s_idx in range(start_shard, start_shard + num_shards):
        pt_path = shard_dir / f"shard_{s_idx:02d}.pt"
        meta_path = shard_dir / f"shard_{s_idx:02d}.meta.json"
        s_seed = seed_base + s_idx * shard_seed_stride

        summary = generate_dataset(
            num_games=0,
            out_jsonl=None,
            out_pt=pt_path,
            target_tokens=tokens_per_shard,
            seed_base=s_seed,
            diversity=diversity,
            blind_k_prob=blind_k_prob,
            blind_underfill_prob=blind_underfill_prob,
            blind_dup_prob=blind_dup_prob,
            max_tokens=max_tokens,
            max_steps=max_steps,
            workers=workers,
        )
        meta_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        summaries.append(summary)
        size_mb = pt_path.stat().st_size / 1024**2
        print(
            f"[Shard {s_idx:02d}] {summary['accepted_games']:,} games | "
            f"{summary['total_tokens']:,} tok | {summary['elapsed_sec']}s "
            f"({summary['games_per_sec']} g/s, {summary['tokens_per_sec']:,.0f} tok/s) | "
            f"vocab {summary['active_tokens_covered']} | {size_mb:.1f} MB -> {pt_path}",
            flush=True,
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate full-universe SFT trajectories for PrimeWordle (80-token vocab)."
    )
    parser.add_argument("--games", type=int, default=500, help="Number of won games (single-file mode).")
    parser.add_argument("--target-tokens", type=int, default=0, help="If >0, generate by token budget instead of --games.")
    parser.add_argument("--out", type=Path, default=Path("data/sft/smoke_500.jsonl"), help="Output JSONL path.")
    parser.add_argument("--no-jsonl", action="store_true", help="Skip writing JSONL (fast .pt-only mode).")
    parser.add_argument("--pt-out", type=Path, default=None, help="Compact uint8 PyTorch shard output path (.pt).")

    # Multi-shard mode
    parser.add_argument("--shard-dir", type=Path, default=None, help="If set, generate numbered .pt shards in this directory.")
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--start-shard", type=int, default=0)

    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_SEED_BASE)
    parser.add_argument("--diversity", type=float, default=0.08)
    parser.add_argument("--blind-k", type=float, default=0.10)
    parser.add_argument("--blind-underfill", type=float, default=0.10)
    parser.add_argument("--blind-dup", type=float, default=0.10)
    parser.add_argument("--max-tokens", type=int, default=32_768)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=f"Parallel worker processes (CPU count: {os.cpu_count() or 1}).",
    )
    args = parser.parse_args()

    if args.shard_dir is not None:
        if args.target_tokens <= 0:
            raise ValueError("--shard-dir requires --target-tokens (tokens per shard).")
        generate_sharded_dataset(
            shard_dir=args.shard_dir,
            num_shards=args.num_shards,
            tokens_per_shard=args.target_tokens,
            seed_base=args.seed,
            start_shard=args.start_shard,
            workers=args.workers,
            diversity=args.diversity,
            blind_k_prob=args.blind_k,
            blind_underfill_prob=args.blind_underfill,
            blind_dup_prob=args.blind_dup,
            max_tokens=args.max_tokens,
            max_steps=args.max_steps,
        )
        return

    summary = generate_dataset(
        num_games=args.games,
        out_jsonl=None if args.no_jsonl else args.out,
        out_pt=args.pt_out,
        target_tokens=args.target_tokens,
        seed_base=args.seed,
        diversity=args.diversity,
        blind_k_prob=args.blind_k,
        blind_underfill_prob=args.blind_underfill,
        blind_dup_prob=args.blind_dup,
        max_tokens=args.max_tokens,
        max_steps=args.max_steps,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
