"""Round-2 Adversarial ("使绊子") SFT Dataset Generator for PrimeWordle (80-token vocab).

Designed specifically to fine-tune `best.pt` into an indestructible general-purpose
logic engine capable of handling:
  - 40% Dirty Mid-Game Takeovers: 1..10 messy human-like turns (random, unsorted,
        occasional repeats/mine-steps) masked with `loss_mask = 0`, followed by
        clean solver recovery turns (`loss_mask = 1`).
  - 35% Random Sparse Forbidden-Graph games (30%..70% forbidden fusions, heavy
        coverage of `UNDERFILL_OFF` + forbidden self-fusions `p*p`, modeling
        real PvZ 10x10 Plant Decoding — with 40% of these ALSO prefixed by dirty
        legal human turns for double-whammy robustness!).
  - 25% Clean Full-Universe anti-forgetting games (`P in [2,32], K in [1,8], L in [1,32]`,
        with extra weight on wide-P K=1 Wordle and high-K K=6..8).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from engine.game import GameConfig, GameSession
from engine.primes import encode_slot, get_primes
from engine.tokenizer import PrimeWordleTokenizer

from .constraints import (
    BoardKnowledge,
    SlotCandidate,
    enumerate_all_slot_candidates,
    symbols_to_mask,
)
from .generate_dataset import _weighted_int, sample_full_universe_config
from .hybrid import HybridSolver, solve_session

# Disjoint seed band from Phase 1 (10M..450M): Round 2 starts at 600_000_000!
DEFAULT_ADV_SEED_BASE = 600_000_000


class ConstrainedSubsetSolver:
    """Exact-pool entropy solver restricted STRICTLY to a legal subset of slot multisets.

    Models "根本填不进去" (forbidden plant fusions / restricted dictionaries): every
    emitted slot is guaranteed to belong to `legal_candidates`.
    """

    def __init__(
        self,
        legal_candidates: Sequence[SlotCandidate],
        seed: int = 0,
        diversity: float = 0.06,
        max_eval_actions: int = 44,
        max_pool_sample: int = 128,
    ) -> None:
        self.legal = list(legal_candidates)
        self.rng = random.Random(seed)
        self.diversity = max(0.0, float(diversity))
        self.max_eval_actions = max_eval_actions
        self.max_pool_sample = max_pool_sample

    def _opening_guess(self, config: GameConfig) -> list[list[int]]:
        """Pick high-coverage legal slots across all L positions on Turn 1."""
        l = config.code_length
        cands = list(self.legal)
        self.rng.shuffle(cands)
        # Prefer diverse masks covering distinct primes across slots
        chosen: list[list[int]] = []
        covered_mask = 0
        used_prods: set[int] = set()
        for _ in range(l):
            best_c = None
            best_gain = -1
            sample = cands[: min(len(cands), 40)]
            for c in sample:
                if c.product in used_prods and len(cands) > l:
                    continue
                gain = (c.mask & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c is None:
                best_c = self.rng.choice(cands)
            chosen.append(list(best_c.symbols))
            covered_mask |= best_c.mask
            used_prods.add(best_c.product)
            if covered_mask == (1 << config.num_primes) - 1:
                covered_mask = 0
                self.rng.shuffle(cands)
        return chosen

    @staticmethod
    def _score_entropy(
        action_prod: int,
        action_mask: int,
        eval_pool: Sequence[SlotCandidate],
        in_pool: bool,
    ) -> float:
        w_exact = w_wrong = w_partial = total = 0.0
        for c in eval_pool:
            w = c.prior
            total += w
            if c.product == action_prod:
                w_exact += w
            elif (c.mask & action_mask) == 0:
                w_wrong += w
            else:
                w_partial += w
        if total <= 0.0:
            return 0.0
        ent = 0.0
        for w in (w_exact, w_wrong, w_partial):
            if w > 0.0:
                p = w / total
                ent -= p * math.log2(p)
        if in_pool:
            ent += 0.03
        return ent

    def next_guess(self, session: GameSession) -> list[list[int]]:
        config = session.config
        if not session.history:
            return self._opening_guess(config)

        knowledge = BoardKnowledge(session)
        pools: list[list[SlotCandidate]] = []
        for i in range(config.code_length):
            if i in knowledge.solved_slots:
                syms = knowledge.solved_slots[i]
                prod = knowledge.solved_products[i]
                pools.append([SlotCandidate(syms, prod, symbols_to_mask(syms), 1.0)])
            else:
                excl = knowledge.excluded_mask[i]
                req = knowledge.required_mask[i]
                clauses = knowledge.partial_clauses[i]
                forb = knowledge.forbidden_products[i]
                valid: list[SlotCandidate] = []
                for c in self.legal:
                    if c.product in forb:
                        continue
                    if c.mask & excl:
                        continue
                    if (c.mask & req) != req:
                        continue
                    ok = True
                    for cl in clauses:
                        if (c.mask & cl) == 0:
                            ok = False
                            break
                    if ok:
                        valid.append(c)
                pools.append(valid)

        guess: list[list[int] | None] = [None] * config.code_length

        # 1. Fill uniquely determined slots
        for i in range(config.code_length):
            if len(pools[i]) == 1:
                guess[i] = list(pools[i][0].symbols)

        # 2. Place active global MISPLACED hints into eligible unsolved slots
        used_slots: set[int] = {i for i in range(config.code_length) if guess[i] is not None}
        for hint in knowledge.misplaced_hints:
            placed = 0
            elig = [
                u
                for u in hint.eligible_slots
                if u not in used_slots and any(c.product == hint.product for c in pools[u])
            ]
            elig.sort(key=lambda u: len(pools[u]))
            for u in elig:
                if placed >= hint.remaining_needed:
                    break
                guess[u] = list(hint.symbols)
                used_slots.add(u)
                placed += 1

        # 3. Entropy-maximize strictly over legal candidates for remaining slots
        for i in range(config.code_length):
            if guess[i] is not None:
                continue
            pool = pools[i]
            if not pool:
                guess[i] = list(self.legal[0].symbols)
                continue
            if len(pool) <= 2:
                guess[i] = list(pool[0].symbols)
                continue

            eval_pool = (
                pool[:: max(1, len(pool) // self.max_pool_sample)][: self.max_pool_sample]
                if len(pool) > self.max_pool_sample
                else pool
            )
            pool_prods = {c.product for c in pool}
            tried = knowledge.tried_at_slot[i]

            # Build action set: surviving pool candidates + untried legal probes
            actions: list[SlotCandidate] = []
            seen: set[int] = set()
            for c in pool[: self.max_eval_actions // 2]:
                if c.product not in seen:
                    seen.add(c.product)
                    actions.append(c)
            step = max(1, len(self.legal) // (self.max_eval_actions // 2))
            for c in self.legal[::step]:
                if len(actions) >= self.max_eval_actions:
                    break
                if c.product not in seen and c.product not in tried:
                    seen.add(c.product)
                    actions.append(c)

            scored = [
                (
                    self._score_entropy(
                        c.product, c.mask, eval_pool, in_pool=(c.product in pool_prods)
                    ),
                    c.symbols,
                )
                for c in actions
            ]
            scored.sort(key=lambda x: -x[0])
            best_sc = scored[0][0]
            if self.diversity > 0.0:
                top = [s for sc, s in scored if sc >= best_sc - self.diversity]
                chosen = self.rng.choice(top[:4])
            else:
                chosen = scored[0][1]
            guess[i] = list(chosen)

        return [g if g is not None else list(self.legal[0].symbols) for g in guess]


def _encode_session_with_prefix_mask(
    tokenizer: PrimeWordleTokenizer,
    session: GameSession,
    masked_prefix_turns: int,
    reveal_k: bool,
    reveal_underfill: bool,
    reveal_dup: bool,
) -> tuple[list[int], list[int]]:
    """Encode session with `loss_mask = 0` for the header AND all `masked_prefix_turns` dirty turns."""
    header = tokenizer.encode_config(
        session.config,
        reveal_k=reveal_k,
        reveal_underfill=reveal_underfill,
        reveal_dup=reveal_dup,
    )
    ids: list[int] = list(header)
    mask: list[int] = [0] * len(header)

    for turn_idx, turn in enumerate(session.history):
        is_recovery = turn_idx >= masked_prefix_turns
        guess_tokens = ["<TURN>"]
        for slot in turn.guess_slots:
            guess_tokens.append("<SLOT>")
            for sym_idx in slot:
                guess_tokens.append(f"S_{sym_idx}")
        guess_tokens.append("<FB>")
        guess_ids = [tokenizer.stoi[t] for t in guess_tokens]
        ids.extend(guess_ids)
        mask.extend([1 if is_recovery else 0] * len(guess_ids))

        fb_ids = [tokenizer.stoi[fb.value] for fb in turn.feedback]
        ids.extend(fb_ids)
        mask.extend([0] * len(fb_ids))

    if session.status == "won":
        ids.append(tokenizer.eos_id)
        mask.append(1)

    return ids, mask


def _play_messy_human_prefix(
    session: GameSession,
    rng: random.Random,
    n_turns: int,
    legal_slots: list[list[int]] | None = None,
) -> int:
    """Simulate 1..n_turns of noisy, imperfect human guesses (canonical sorted symbols).

    Returns actual number of turns played before either hitting `n_turns` or stopping
    early if the board gets close to winning.
    """
    cfg = session.config
    k_lo = 1 if cfg.allow_underfill else cfg.max_multiplier

    def _rand_slot() -> list[int]:
        if legal_slots is not None:
            return sorted(rng.choice(legal_slots))
        k = rng.randint(k_lo, cfg.max_multiplier)
        return sorted(rng.randrange(cfg.num_primes) for _ in range(k))

    played = 0
    for _ in range(n_turns):
        if session.status != "playing":
            break
        knowledge = BoardKnowledge(session) if session.history else None
        guess: list[list[int]] = []
        style = rng.random()

        for i in range(cfg.code_length):
            # 60% of time humans keep already-EXACT slots; 40% forget/change them
            if (
                knowledge is not None
                and i in knowledge.confirmed_exact
                and rng.random() < 0.65
            ):
                guess.append(list(knowledge.solved_slots[i]))
            elif style < 0.25 and session.history and rng.random() < 0.35:
                # Realistic human mistake: occasionally repeat an earlier guess at slot i
                prev = rng.choice(session.history)
                guess.append(sorted(prev.guess_slots[i]))
            else:
                guess.append(_rand_slot())

        # Avoid accidental instant-win during messy prefix so solver still has work to do
        prods = [encode_slot(s, session.primes) for s in guess]
        if prods == session.target_products:
            # Perturb one slot if possible
            alt = _rand_slot()
            if encode_slot(alt, session.primes) != session.target_products[0]:
                guess[0] = alt
            else:
                break

        session.submit_guess(guess)
        played += 1

    return played


def _sample_forbidden_graph_episode(
    rng: random.Random,
    game_seed: int,
) -> tuple[GameSession, ConstrainedSubsetSolver, list[list[int]], str]:
    """Sample a random sparse forbidden-fusion game (heavy PvZ 10x10 UNDERFILL_OFF + p*p bans)."""
    if rng.random() < 0.80:
        # K=2 PvZ Plant Decoding focus (especially P in 6..12, L in 5..12, UNDERFILL_OFF!)
        k = 2
        if rng.random() < 0.45:
            p = rng.randint(8, 12)
            l = rng.randint(6, 12)
        else:
            p = rng.randint(5, 16)
            l = rng.randint(4, 16)
        allow_underfill = rng.random() < 0.35  # 65% UNDERFILL_OFF (true PvZ 10x10!)
    else:
        # K=3 sparse forbidden combinations
        k = 3
        p = rng.randint(4, 10)
        l = rng.randint(3, 10)
        allow_underfill = rng.random() < 0.50

    base_cfg = GameConfig(
        num_primes=p,
        max_multiplier=k,
        code_length=l,
        allow_underfill=allow_underfill,
        allow_duplicate_slots=True,
    )
    all_cands = enumerate_all_slot_candidates(base_cfg)

    forbid_ratio = rng.uniform(0.30, 0.70)
    forbid_self_ratio = rng.uniform(0.30, 1.00)  # Up to 100% self-fusion p*p forbidden!

    if k == 2:
        singles = [c for c in all_cands if len(c.symbols) == 1]
        self_pairs = [c for c in all_cands if len(c.symbols) == 2 and c.symbols[0] == c.symbols[1]]
        cross_pairs = [c for c in all_cands if len(c.symbols) == 2 and c.symbols[0] != c.symbols[1]]
        rng.shuffle(self_pairs)
        rng.shuffle(cross_pairs)
        keep_cross = max(p, int(round(len(cross_pairs) * (1.0 - forbid_ratio))))
        keep_self = int(round(len(self_pairs) * (1.0 - forbid_self_ratio)))
        legal_cands = singles + cross_pairs[:keep_cross] + self_pairs[:keep_self]
        if len(legal_cands) < max(8, l):
            legal_cands = all_cands[: max(8, l)]
    else:
        shuffled = list(all_cands)
        rng.shuffle(shuffled)
        keep_n = max(max(12, l + 4), int(round(len(shuffled) * (1.0 - forbid_ratio))))
        legal_cands = shuffled[:keep_n]

    # Uniform prior over the surviving legal subset for clean subset inference
    legal_cands = [
        SlotCandidate(c.symbols, c.product, c.mask, 1.0 / len(legal_cands))
        for c in legal_cands
    ]
    can_no_dup = l <= len(legal_cands)
    allow_dup = not (can_no_dup and rng.random() < 0.25)

    cfg = GameConfig(
        num_primes=p,
        max_multiplier=k,
        code_length=l,
        allow_underfill=allow_underfill,
        allow_duplicate_slots=allow_dup,
    )
    if allow_dup:
        target_slots = [list(rng.choice(legal_cands).symbols) for _ in range(l)]
    else:
        picked = rng.sample(legal_cands, l)
        target_slots = [list(c.symbols) for c in picked]

    session = GameSession(cfg, seed=game_seed, target_slots=target_slots)
    solver = ConstrainedSubsetSolver(legal_cands, seed=game_seed * 5 + 11, diversity=0.06)
    legal_slot_lists = [list(c.symbols) for c in legal_cands]
    sub_tag = "forbid_uoff" if not allow_underfill else "forbid_uon"
    return session, solver, legal_slot_lists, sub_tag


def _sample_anti_forgetting_config(rng: random.Random) -> tuple[GameConfig, str]:
    """Full-universe config with boosted K=1 wide-P (Wordle) and K=6..8 coverage."""
    u = rng.random()
    if u < 0.20:
        # Boost wide-P K=1 (Wordle / alphabet preservation)
        p = _weighted_int(rng, 16, 32, power=0.3)
        k = 1
        l = rng.randint(4, 12)
        allow_u = False
        allow_dup = rng.random() < 0.70
        return GameConfig(p, k, l, allow_underfill=allow_u, allow_duplicate_slots=allow_dup), "clean_k1_wide"
    if u < 0.45:
        # Boost K=6..8 high-order preservation & uplift
        p = _weighted_int(rng, 12, 32, power=0.4)
        k = rng.randint(6, 8)
        l = rng.randint(3, 12)
        allow_u = rng.random() < 0.70
        allow_dup = rng.random() < 0.75
        return GameConfig(p, k, l, allow_underfill=allow_u, allow_duplicate_slots=allow_dup), "clean_high_k"
    cfg, tier = sample_full_universe_config(rng)
    return cfg, f"clean_{tier}"


def _worker_generate_adv_chunk(task: dict[str, Any]) -> dict[str, Any]:
    target_tokens: int = task["target_tokens"]
    seed_base: int = task["seed_base"]
    diversity: float = task["diversity"]
    blind_prob: float = task["blind_prob"]
    max_tokens: int = task["max_tokens"]
    max_steps: int = task["max_steps"]

    tokenizer = PrimeWordleTokenizer(max_symbols=32)
    rng = random.Random(seed_base)

    flat_ids = bytearray()
    flat_mask = bytearray()
    lengths: list[int] = []
    token_counts = [0] * tokenizer.vocab_size
    mode_counts: Counter[str] = Counter()
    turn_counts: list[int] = []

    skipped_unwon = 0
    skipped_too_long = 0
    attempt = 0
    total_toks = 0

    while total_toks < target_tokens:
        attempt += 1
        game_seed = seed_base + attempt
        u = rng.random()

        if u < 0.40:
            # 40% Dirty Mid-Game Takeover (full universe, boosted tactical human range)
            if rng.random() < 0.45:
                p = _weighted_int(rng, 4, 16, power=0.3)
                k = rng.randint(1, 4)
                l = rng.randint(3, 12)
                allow_u = False if k == 1 else (rng.random() < 0.55)
                cfg = GameConfig(p, k, l, allow_underfill=allow_u, allow_duplicate_slots=True)
            else:
                cfg, _ = sample_full_universe_config(rng)

            session = GameSession(cfg, seed=game_seed)
            max_messy = 6 if (cfg.max_multiplier <= 2 and cfg.num_primes <= 8) else 10
            n_messy_target = rng.randint(1, max_messy)
            n_messy = _play_messy_human_prefix(session, rng, n_messy_target, legal_slots=None)
            if session.status != "playing":
                continue

            solver = HybridSolver(seed=game_seed * 3 + 7, diversity=diversity)
            res = solve_session(session, solver=solver, max_steps=max_steps)
            if not res.won:
                skipped_unwon += 1
                continue
            mode_tag = "dirty_takeover"
            masked_prefix = n_messy

        elif u < 0.75:
            # 35% Random Sparse Forbidden-Graph (40% of these ALSO get dirty legal human prefix!)
            session, sub_solver, legal_lists, sub_tag = _sample_forbidden_graph_episode(rng, game_seed)
            masked_prefix = 0
            if rng.random() < 0.40:
                n_messy = _play_messy_human_prefix(
                    session, rng, rng.randint(1, 6), legal_slots=legal_lists
                )
                if session.status != "playing":
                    continue
                masked_prefix = n_messy
                mode_tag = f"{sub_tag}_dirty"
            else:
                mode_tag = f"{sub_tag}_fresh"

            res = solve_session(session, solver=sub_solver, max_steps=max_steps)
            if not res.won:
                skipped_unwon += 1
                continue

        else:
            # 25% Clean Full-Universe Anti-Forgetting
            cfg, clean_tag = _sample_anti_forgetting_config(rng)
            session = GameSession(cfg, seed=game_seed)
            solver = HybridSolver(seed=game_seed * 3 + 7, diversity=diversity)
            res = solve_session(session, solver=solver, max_steps=max_steps)
            if not res.won:
                skipped_unwon += 1
                continue
            mode_tag = clean_tag
            masked_prefix = 0

        reveal_k = rng.random() >= blind_prob
        reveal_underfill = rng.random() >= blind_prob
        reveal_dup = rng.random() >= blind_prob
        ids, mask = _encode_session_with_prefix_mask(
            tokenizer,
            session,
            masked_prefix_turns=masked_prefix,
            reveal_k=reveal_k,
            reveal_underfill=reveal_underfill,
            reveal_dup=reveal_dup,
        )
        n_tok = len(ids)
        if n_tok > max_tokens or sum(mask) <= 1:
            skipped_too_long += 1
            continue

        flat_ids.extend(ids)
        flat_mask.extend(mask)
        lengths.append(n_tok)
        for tid in ids:
            token_counts[tid] += 1
        mode_counts[mode_tag] += 1
        turn_counts.append(len(session.history))
        total_toks += n_tok

    return {
        "flat_ids": bytes(flat_ids),
        "flat_mask": bytes(flat_mask),
        "lengths": lengths,
        "token_counts": token_counts,
        "mode_counts": dict(mode_counts),
        "turn_counts": turn_counts,
        "attempts": attempt,
        "skipped_unwon": skipped_unwon,
        "skipped_too_long": skipped_too_long,
    }


def generate_adversarial_shards(
    shard_dir: Path,
    num_shards: int = 2,
    tokens_per_shard: int = 50_000_000,
    seed_base: int = DEFAULT_ADV_SEED_BASE,
    start_shard: int = 0,
    workers: int = 12,
    diversity: float = 0.08,
    blind_prob: float = 0.08,
    max_tokens: int = 32_768,
    max_steps: int = 220,
) -> list[dict[str, Any]]:
    """Generate disjoint Round-2 adversarial `.pt` shards (`adv_00.pt`, ...)."""
    import torch

    tokenizer = PrimeWordleTokenizer(max_symbols=32)
    shard_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    shard_seed_stride = 50_000_000

    for s_idx in range(start_shard, start_shard + num_shards):
        pt_path = shard_dir / f"adv_{s_idx:02d}.pt"
        meta_path = shard_dir / f"adv_{s_idx:02d}.meta.json"
        if pt_path.exists() and meta_path.exists():
            print(f"[Skip] {pt_path} already exists.", flush=True)
            summaries.append(json.loads(meta_path.read_text(encoding="utf-8")))
            continue

        t0 = time.perf_counter()
        s_seed = seed_base + s_idx * shard_seed_stride
        n_w = max(1, workers)
        tasks = [
            {
                "target_tokens": tokens_per_shard // n_w + (1 if w < tokens_per_shard % n_w else 0),
                "seed_base": s_seed + w * 2_000_000,
                "diversity": diversity,
                "blind_prob": blind_prob,
                "max_tokens": max_tokens,
                "max_steps": max_steps,
            }
            for w in range(n_w)
        ]

        if n_w == 1:
            chunks = [_worker_generate_adv_chunk(tasks[0])]
        else:
            with ProcessPoolExecutor(max_workers=n_w) as pool:
                chunks = list(pool.map(_worker_generate_adv_chunk, tasks))

        token_counts = [0] * tokenizer.vocab_size
        mode_counts: Counter[str] = Counter()
        seq_lengths: list[int] = []
        turn_counts: list[int] = []
        total_attempts = skipped_unwon = skipped_too_long = 0

        for ch in chunks:
            total_attempts += ch["attempts"]
            skipped_unwon += ch["skipped_unwon"]
            skipped_too_long += ch["skipped_too_long"]
            for i, c in enumerate(ch["token_counts"]):
                token_counts[i] += c
            mode_counts.update(ch["mode_counts"])
            seq_lengths.extend(ch["lengths"])
            turn_counts.extend(ch["turn_counts"])

        accepted = len(seq_lengths)
        total_toks = sum(seq_lengths)
        ids_buf = bytearray(total_toks)
        mask_buf = bytearray(total_toks)
        offsets = [0]
        pos = 0
        for ch in chunks:
            b_ids = ch["flat_ids"]
            b_mask = ch["flat_mask"]
            n_b = len(b_ids)
            ids_buf[pos : pos + n_b] = b_ids
            mask_buf[pos : pos + n_b] = b_mask
            for length in ch["lengths"]:
                offsets.append(offsets[-1] + length)
            pos += n_b

        torch.save(
            {
                "vocab_size": tokenizer.vocab_size,
                "itos": tokenizer.itos,
                "num_games": accepted,
                "input_ids": torch.frombuffer(ids_buf, dtype=torch.uint8).clone(),
                "loss_mask": torch.frombuffer(mask_buf, dtype=torch.uint8).clone(),
                "offsets": torch.tensor(offsets, dtype=torch.int64),
            },
            pt_path,
        )
        elapsed = time.perf_counter() - t0
        non_pad = [i for i in range(tokenizer.vocab_size) if tokenizer.itos[i] != "<PAD>"]
        missing = [tokenizer.itos[i] for i in non_pad if token_counts[i] == 0]
        sorted_lens = sorted(seq_lengths)
        summary = {
            "accepted_games": accepted,
            "attempts": total_attempts,
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
            "modes": dict(sorted(mode_counts.items())),
            "active_tokens_covered": f"{len(non_pad) - len(missing)}/{len(non_pad)}",
            "missing_non_pad_tokens": missing,
            "out_pt": str(pt_path),
        }
        meta_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        summaries.append(summary)
        size_mb = pt_path.stat().st_size / 1024**2
        print(
            f"[AdvShard {s_idx:02d}] {accepted:,} games | {total_toks:,} tok | "
            f"{elapsed:.1f}s ({summary['games_per_sec']} g/s, {summary['tokens_per_sec']:,.0f} tok/s) | "
            f"vocab {summary['active_tokens_covered']} | {size_mb:.1f} MB -> {pt_path}",
            flush=True,
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Round-2 adversarial SFT shards.")
    parser.add_argument("--shard-dir", type=Path, default=Path("data/sft/adv_100m"))
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--tokens-per-shard", type=int, default=50_000_000)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=DEFAULT_ADV_SEED_BASE)
    args = parser.parse_args()

    generate_adversarial_shards(
        shard_dir=args.shard_dir,
        num_shards=args.num_shards,
        tokens_per_shard=args.tokens_per_shard,
        seed_base=args.seed,
        start_shard=args.start_shard,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
