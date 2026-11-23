"""Fixed-seed multi-tier benchmark suite for PrimeWordle solvers.

Establishes reproducible baseline metrics (win rate, avg/P95 turns, avg/max
token length, speed) across the full (P in [2,32], K in [1,8], L in [1,32])
envelope so future Mini-Transformer and RL checkpoints have an exact ruler.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

from engine.game import GameConfig, GameSession
from engine.tokenizer import PrimeWordleTokenizer

from .hybrid import HybridSolver, solve_session


@dataclass(frozen=True)
class BenchmarkPreset:
    name: str
    config: GameConfig
    games: int
    base_seed: int


DEFAULT_PRESETS: tuple[BenchmarkPreset, ...] = (
    BenchmarkPreset(
        name="K1_Wordle_P6_L5",
        config=GameConfig(num_primes=6, max_multiplier=1, code_length=5, allow_underfill=False),
        games=40,
        base_seed=10_000,
    ),
    BenchmarkPreset(
        name="K1_Alpha_P26_L5",
        config=GameConfig(num_primes=26, max_multiplier=1, code_length=5, allow_underfill=False),
        games=30,
        base_seed=20_000,
    ),
    BenchmarkPreset(
        name="K2_Plant_P5_L5_U",
        config=GameConfig(num_primes=5, max_multiplier=2, code_length=5, allow_underfill=True),
        games=40,
        base_seed=30_000,
    ),
    BenchmarkPreset(
        name="K2_Wide_P32_L8_U",
        config=GameConfig(num_primes=32, max_multiplier=2, code_length=8, allow_underfill=True),
        games=20,
        base_seed=40_000,
    ),
    BenchmarkPreset(
        name="K3_Med_P8_L5_U",
        config=GameConfig(num_primes=8, max_multiplier=3, code_length=5, allow_underfill=True),
        games=30,
        base_seed=50_000,
    ),
    BenchmarkPreset(
        name="K4_Wide_P25_L8_U",
        config=GameConfig(num_primes=25, max_multiplier=4, code_length=8, allow_underfill=True),
        games=20,
        base_seed=60_000,
    ),
    BenchmarkPreset(
        name="K6_Hard_P32_L8_U",
        config=GameConfig(num_primes=32, max_multiplier=6, code_length=8, allow_underfill=True),
        games=15,
        base_seed=70_000,
    ),
    BenchmarkPreset(
        name="K6_Hard_P32_L8_Full",
        config=GameConfig(num_primes=32, max_multiplier=6, code_length=8, allow_underfill=False),
        games=15,
        base_seed=75_000,
    ),
    BenchmarkPreset(
        name="K8_Max_P32_L5_U",
        config=GameConfig(num_primes=32, max_multiplier=8, code_length=5, allow_underfill=True),
        games=10,
        base_seed=80_000,
    ),
    BenchmarkPreset(
        name="K8_Max_P32_L32_U",
        config=GameConfig(num_primes=32, max_multiplier=8, code_length=32, allow_underfill=True),
        games=5,
        base_seed=90_000,
    ),
)


def _percentile(sorted_vals: list[int], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round((len(sorted_vals) - 1) * pct)))
    return float(sorted_vals[idx])


def run_benchmark(
    presets: tuple[BenchmarkPreset, ...] = DEFAULT_PRESETS,
    scale_games: float = 1.0,
    max_steps: int = 250,
) -> list[dict]:
    tokenizer = PrimeWordleTokenizer(max_symbols=32)
    results: list[dict] = []

    header = (
        f"{'Preset':<22} | {'Back':<6} | {'Games':>5} | {'Win%':>6} | "
        f"{'AvgT':>5} | {'P95T':>5} | {'AvgTok':>7} | {'MaxTok':>7} | {'ms/g':>6}"
    )
    print(header)
    print("-" * len(header))

    for preset in presets:
        n_games = max(1, int(round(preset.games * scale_games)))
        turns_list: list[int] = []
        tok_lens: list[int] = []
        wins = 0
        backend_used = ""

        t0 = time.perf_counter()
        for i in range(n_games):
            seed = preset.base_seed + i
            session = GameSession(preset.config, seed=seed)
            solver = HybridSolver(seed=seed, diversity=0.0)
            res = solve_session(session, solver=solver, max_steps=max_steps)
            backend_used = res.backend
            if res.won:
                wins += 1
            turns_list.append(res.turns)
            tok_lens.append(len(tokenizer.encode_session(session)))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        turns_sorted = sorted(turns_list)
        row = {
            "preset": preset.name,
            "backend": backend_used,
            "games": n_games,
            "win_rate": wins / n_games,
            "avg_turns": statistics.mean(turns_list),
            "p95_turns": _percentile(turns_sorted, 0.95),
            "avg_tokens": statistics.mean(tok_lens),
            "max_tokens": max(tok_lens),
            "ms_per_game": elapsed_ms / n_games,
        }
        results.append(row)
        print(
            f"{row['preset']:<22} | {row['backend']:<6} | {row['games']:>5d} | "
            f"{row['win_rate']*100:5.1f}% | {row['avg_turns']:5.1f} | "
            f"{row['p95_turns']:5.1f} | {row['avg_tokens']:7.0f} | "
            f"{row['max_tokens']:7d} | {row['ms_per_game']:6.1f}"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PrimeWordle baseline solver benchmark.")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for game counts per preset (e.g. 0.5 for quick run).",
    )
    parser.add_argument("--max-steps", type=int, default=250)
    args = parser.parse_args()
    run_benchmark(scale_games=args.scale, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
