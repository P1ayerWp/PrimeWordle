"""Exact candidate-pool information-gain solver for small/medium slot spaces.

Suitable whenever single-slot combination count M <= ~2500 (covers ALL K=1,
ALL K=2 up to P=32, K=3 up to P=22, K=4 up to P=13, etc.).
"""

from __future__ import annotations

import math
import random
from functools import lru_cache
from typing import Sequence

from engine.game import GameConfig, GameSession
from engine.primes import encode_slot

from .constraints import (
    BoardKnowledge,
    SlotCandidate,
    enumerate_all_slot_candidates,
    mask_to_symbols,
    symbols_to_mask,
)


@lru_cache(maxsize=256)
def _cached_candidates(
    num_primes: int, max_multiplier: int, allow_underfill: bool
) -> tuple[SlotCandidate, ...]:
    cfg = GameConfig(
        num_primes=num_primes,
        max_multiplier=max_multiplier,
        code_length=1,
        allow_underfill=allow_underfill,
    )
    cands = enumerate_all_slot_candidates(cfg)
    cands.sort(key=lambda c: (-c.prior, len(c.symbols), c.symbols))
    return tuple(cands)


def _pad_symbols(
    syms: Sequence[int], config: GameConfig
) -> tuple[int, ...]:
    """Ensure canonical sorted tuple obeying allow_underfill & max_multiplier."""
    base = list(syms[: config.max_multiplier])
    if not config.allow_underfill and len(base) < config.max_multiplier:
        base.extend([base[0]] * (config.max_multiplier - len(base)))
    base.sort()
    return tuple(base)


class ExactPoolSolver:
    """Per-slot exact candidate pool + global MISPLACED + entropy probing solver."""

    def __init__(
        self,
        seed: int = 0,
        diversity: float = 0.0,
        max_eval_actions: int = 40,
        max_pool_sample: int = 128,
    ) -> None:
        self.rng = random.Random(seed)
        self.diversity = max(0.0, float(diversity))
        self.max_eval_actions = max_eval_actions
        self.max_pool_sample = max_pool_sample

    def _opening_guess(self, config: GameConfig) -> list[list[int]]:
        """Construct a high-coverage Turn-1 guess across all slots."""
        p = config.num_primes
        k = config.max_multiplier
        l = config.code_length

        order = list(range(p))
        self.rng.shuffle(order)

        if k == 1:
            return [[order[i % p]] for i in range(l)]

        # Target distinct primes per slot ~40-50% of P (capped at K) so WRONG/PARTIAL splits cleanly
        target_distinct = max(1, min(k, max(1, int(round(p * 0.45)))))
        if not config.allow_underfill and target_distinct == p and p > 1:
            target_distinct = p - 1

        guess: list[list[int]] = []
        cursor = 0
        for _ in range(l):
            picked: list[int] = []
            for _j in range(target_distinct):
                picked.append(order[cursor % p])
                cursor += 1
                if cursor % p == 0:
                    self.rng.shuffle(order)
            guess.append(list(_pad_symbols(picked, config)))
        return guess

    def _build_probe_actions(
        self,
        pool: Sequence[SlotCandidate],
        knowledge: BoardKnowledge,
        slot_idx: int,
        config: GameConfig,
    ) -> list[tuple[tuple[int, ...], int, int]]:
        """Generate candidate actions (symbols, product, mask) mixing pool & pure probes."""
        primes = knowledge.primes
        seen_prods: set[int] = set()
        actions: list[tuple[tuple[int, ...], int, int]] = []

        # 1. Top-prior surviving candidates from the pool
        step = max(1, len(pool) // self.max_eval_actions)
        for c in pool[: self.max_eval_actions // 2]:
            if c.product not in seen_prods:
                seen_prods.add(c.product)
                actions.append((c.symbols, c.product, c.mask))
        for c in pool[self.max_eval_actions // 2 :: step]:
            if len(actions) >= self.max_eval_actions:
                break
            if c.product not in seen_prods:
                seen_prods.add(c.product)
                actions.append((c.symbols, c.product, c.mask))

        # 2. Pure-probe subsets targeting primes that vary across the surviving pool
        union_mask = 0
        inter_mask = knowledge.all_primes_mask
        for c in pool:
            union_mask |= c.mask
            inter_mask &= c.mask
        varying_mask = union_mask & ~inter_mask
        varying_syms = mask_to_symbols(varying_mask)

        if varying_syms and config.max_multiplier >= 1:
            tried = knowledge.tried_at_slot[slot_idx]
            # Always include single-prime probes for every unresolved varying prime
            # (essential when K is large and P is small, e.g. P=6, K=8, so testing p^K
            # immediately eliminates absent primes instead of enumerating exponents).
            for s in varying_syms:
                sub = _pad_symbols([s], config)
                prod = encode_slot(sub, primes)
                if prod not in seen_prods and prod not in tried:
                    seen_prods.add(prod)
                    actions.append((sub, prod, symbols_to_mask(sub)))

            # Rank varying primes by how close their presence frequency is to 50%
            n_pool = len(pool)
            freq = []
            for s in varying_syms:
                bit = 1 << s
                cnt = sum(1 for c in pool if c.mask & bit)
                freq.append((abs(cnt / n_pool - 0.5), s))
            freq.sort(key=lambda x: x[0])
            ranked = [s for _, s in freq]

            max_probe_sz = min(config.max_multiplier, len(ranked))
            probe_sizes = sorted(
                {max(1, max_probe_sz // 2), max_probe_sz}
            )
            for sz in probe_sizes:
                if sz <= 1:
                    continue
                for start in (0, min(len(ranked) - sz, sz)):
                    sub = _pad_symbols(ranked[start : start + sz], config)
                    prod = encode_slot(sub, primes)
                    if prod not in seen_prods and prod not in tried:
                        seen_prods.add(prod)
                        actions.append((sub, prod, symbols_to_mask(sub)))

        return actions

    def _score_action_entropy(
        self,
        action_prod: int,
        action_mask: int,
        eval_pool: Sequence[SlotCandidate],
        in_pool: bool,
    ) -> float:
        """Expected Shannon information gain (in bits) of testing this slot action."""
        w_exact = 0.0
        w_wrong = 0.0
        w_partial = 0.0
        total_w = 0.0

        for c in eval_pool:
            w = c.prior
            total_w += w
            if c.product == action_prod:
                w_exact += w
            elif (c.mask & action_mask) == 0:
                w_wrong += w
            else:
                w_partial += w

        if total_w <= 0.0:
            return 0.0

        entropy = 0.0
        for w in (w_exact, w_wrong, w_partial):
            if w > 0.0:
                p = w / total_w
                entropy -= p * math.log2(p)

        # Slight tie-break bonus for valid candidates (can solve slot immediately)
        if in_pool:
            entropy += 0.02
        return entropy

    def next_guess(self, session: GameSession) -> list[list[int]]:
        config = session.config
        if not session.history:
            return self._opening_guess(config)

        knowledge = BoardKnowledge(session)
        all_cands = _cached_candidates(
            config.num_primes, config.max_multiplier, config.allow_underfill
        )

        # Build filtered pools per slot
        pools: list[list[SlotCandidate]] = []
        for i in range(config.code_length):
            if i in knowledge.solved_slots:
                syms = knowledge.solved_slots[i]
                prod = knowledge.solved_products[i]
                pools.append(
                    [
                        SlotCandidate(
                            symbols=syms,
                            product=prod,
                            mask=symbols_to_mask(syms),
                            prior=1.0,
                        )
                    ]
                )
            else:
                excl = knowledge.excluded_mask[i]
                req = knowledge.required_mask[i]
                clauses = knowledge.partial_clauses[i]
                forb = knowledge.forbidden_products[i]
                valid: list[SlotCandidate] = []
                for c in all_cands:
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
        used_slots: set[int] = {
            i for i in range(config.code_length) if guess[i] is not None
        }
        for hint in knowledge.misplaced_hints:
            placed = 0
            # Prefer eligible slots with smallest remaining pool
            elig = [
                u
                for u in hint.eligible_slots
                if u not in used_slots
                and any(c.product == hint.product for c in pools[u])
            ]
            elig.sort(key=lambda u: len(pools[u]))
            for u in elig:
                if placed >= hint.remaining_needed:
                    break
                guess[u] = list(hint.symbols)
                used_slots.add(u)
                placed += 1

        # 3. Pick best information-gain move for remaining unsolved slots
        for i in range(config.code_length):
            if guess[i] is not None:
                continue
            pool = pools[i]
            if not pool:
                # Defensive fallback (should never trigger for valid histories)
                guess[i] = list(_pad_symbols([0], config))
                continue
            if len(pool) <= 2:
                guess[i] = list(pool[0].symbols)
                continue

            if len(pool) > self.max_pool_sample:
                step = max(1, len(pool) // self.max_pool_sample)
                eval_pool = pool[::step][: self.max_pool_sample]
            else:
                eval_pool = pool

            pool_prods = {c.product for c in pool}
            actions = self._build_probe_actions(pool, knowledge, i, config)

            scored: list[tuple[float, tuple[int, ...]]] = []
            for syms, prod, mask in actions:
                sc = self._score_action_entropy(
                    prod, mask, eval_pool, in_pool=(prod in pool_prods)
                )
                scored.append((sc, syms))

            scored.sort(key=lambda x: -x[0])
            best_score = scored[0][0]
            if self.diversity > 0.0:
                top_choices = [
                    syms for sc, syms in scored if sc >= best_score - self.diversity
                ]
                chosen = self.rng.choice(top_choices[:5])
            else:
                chosen = scored[0][1]
            guess[i] = list(chosen)

        return [g if g is not None else list(_pad_symbols([0], config)) for g in guess]
