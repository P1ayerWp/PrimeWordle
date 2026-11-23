"""Scalable factor-level group-testing solver for large (P, K, L) spaces.

Handles huge combinatorial slot spaces (e.g. P=32, K=8, L=32 where a single
slot has >20 million combinations) without ever enumerating the full pool:
  Phase 1 (Factor Discovery): Adaptive Dorfman / binary-splitting group tests
      on prime bitmasks to find each slot's exact distinct prime support S_i.
  Phase 2 (Exponent Resolution): Enumerates ONLY multisets matching the exact
      discovered support S_i (at most C(8,4)=70 for K<=8) ordered by exact
      Bayesian multinomial prior probability.
Also integrates global MISPLACED constraint locking and placement hints.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from engine.game import GameConfig, GameSession
from engine.primes import encode_slot

from .constraints import (
    BoardKnowledge,
    enumerate_support_candidates,
    mask_to_symbols,
)


def _materialize_probe(
    probe_syms: Sequence[int],
    config: GameConfig,
    primes: Sequence[int],
    forbidden: set[int],
) -> list[int]:
    """Turn a set of prime indices into a legal canonical slot, dodging forbidden products.

    Crucial edge case: if a previous probe with the exact same product received
    MISPLACED (not WRONG/PARTIAL), it landed in `forbidden` without giving factor
    info. Tweaking multiplicity (repeating a prime) preserves the EXACT same
    prime-factor bitmask while producing a distinct Gödel product!
    """
    k_max = config.max_multiplier
    base = sorted(set(probe_syms))[:k_max]
    if not base:
        base = [0]

    def pad(syms: list[int]) -> list[int]:
        out = list(syms)
        if not config.allow_underfill and len(out) < k_max:
            out.extend([out[0]] * (k_max - len(out)))
        out.sort()
        return out

    # Primary attempt
    cand = pad(base)
    if encode_slot(cand, primes) not in forbidden:
        return cand

    # Fallback 1: vary multiplicity of an element in `base` (preserves prime mask!)
    if config.allow_underfill and len(base) < k_max:
        for rep in base:
            alt = sorted([*base, rep])
            if encode_slot(alt, primes) not in forbidden:
                return alt
        # Try adding 2 copies if space permits
        if len(base) + 2 <= k_max:
            for rep in base:
                alt = sorted([*base, rep, rep])
                if encode_slot(alt, primes) not in forbidden:
                    return alt
    elif not config.allow_underfill and len(base) < k_max:
        # Pad using a different element of `base`
        for rep in base[1:]:
            alt = sorted([*base, *([rep] * (k_max - len(base)))])
            if encode_slot(alt, primes) not in forbidden:
                return alt

    # Fallback 2: shrink subset by half to break collision (strictly smaller probe)
    if len(base) >= 2:
        half = pad(base[: len(base) // 2])
        if encode_slot(half, primes) not in forbidden:
            return half
        other = pad(base[len(base) // 2 :])
        if encode_slot(other, primes) not in forbidden:
            return other

    return cand


class FactorProbeSolver:
    """Adaptive group-testing + Bayesian exponent solver for arbitrary (P, K, L)."""

    def __init__(self, seed: int = 0, diversity: float = 0.0) -> None:
        self.rng = random.Random(seed)
        self.diversity = max(0.0, float(diversity))

    def _optimal_batch_size(
        self,
        num_unknown: int,
        num_required: int,
        config: GameConfig,
    ) -> int:
        """Compute group-testing batch size b so P(WRONG) = (1 - q)^b ~= 0.5."""
        if num_unknown <= 1:
            return num_unknown
        k = config.max_multiplier
        exp_k = (k + 1) / 2.0 if config.allow_underfill else float(k)
        # Expected remaining distinct primes in unknown set
        rem_expected = max(0.5, exp_k - num_required)
        q = min(0.85, max(0.03, rem_expected / num_unknown))
        b = int(round(math.log(0.5) / math.log(1.0 - q)))
        b = max(1, min(k, num_unknown, b))

        # Critical: if num_required == 0, the slot MUST contain at least 1 prime
        # from unknown; testing ALL unknown primes at once guarantees PARTIAL (0 info)!
        if num_required == 0 and b >= num_unknown:
            b = max(1, num_unknown // 2)
        return b

    def _pick_slot_move(
        self,
        slot_idx: int,
        knowledge: BoardKnowledge,
        global_prime_order: list[int],
    ) -> list[int]:
        config = knowledge.config
        primes = knowledge.primes
        req_mask = knowledge.required_mask[slot_idx]
        unk_mask = knowledge.unknown_mask(slot_idx)
        clauses = knowledge.partial_clauses[slot_idx]
        forb = knowledge.forbidden_products[slot_idx]
        tried = knowledge.tried_at_slot[slot_idx]

        # Phase 2: All primes resolved (unknown == 0) -> Bayesian exponent resolution
        if unk_mask == 0:
            sup = mask_to_symbols(req_mask)
            cands = [
                c
                for c in enumerate_support_candidates(sup, config, primes)
                if c.product not in forb
            ]
            if not cands:
                # Defensive fallback
                return _materialize_probe(sup or [0], config, primes, set())
            if self.diversity > 0.0 and len(cands) > 1:
                best_p = cands[0].prior
                tied = [c for c in cands if c.prior >= best_p * (1.0 - 0.15 * self.diversity)]
                return list(self.rng.choice(tied[:4]).symbols)
            return list(cands[0].symbols)

        # Phase 1a: Bisect the smallest active unresolved PARTIAL clause
        if clauses:
            smallest = min(clauses, key=lambda m: m.bit_count())
            syms = mask_to_symbols(smallest)
            if self.diversity > 0.0:
                self.rng.shuffle(syms)
            half_sz = max(1, min(config.max_multiplier, len(syms) // 2))
            return _materialize_probe(syms[:half_sz], config, primes, tried)

        # Phase 1b: Group-test unknown primes with adaptive optimal batch size
        u_syms = mask_to_symbols(unk_mask)
        b = self._optimal_batch_size(len(u_syms), req_mask.bit_count(), config)

        # Rotate start index by slot_idx so parallel slots cover different primes
        u_set = set(u_syms)
        ordered_u = [p for p in global_prime_order if p in u_set]
        if ordered_u:
            rot = (slot_idx * b) % len(ordered_u)
            ordered_u = ordered_u[rot:] + ordered_u[:rot]
        else:
            ordered_u = u_syms

        probe = ordered_u[:b]
        return _materialize_probe(probe, config, primes, tried)

    def next_guess(self, session: GameSession) -> list[list[int]]:
        config = session.config
        knowledge = BoardKnowledge(session)
        guess: list[list[int] | None] = [None] * config.code_length

        # 1. Fill all already-solved or deduced slots
        for i, syms in knowledge.solved_slots.items():
            guess[i] = list(syms)

        # 2. Prioritize placing active global MISPLACED hints into eligible slots
        used_slots: set[int] = set(knowledge.solved_slots.keys())
        for hint in knowledge.misplaced_hints:
            placed = 0
            elig = [u for u in hint.eligible_slots if u not in used_slots]
            # Prefer slots closest to being resolved
            elig.sort(key=lambda u: knowledge.unknown_mask(u).bit_count())
            for u in elig:
                if placed >= hint.remaining_needed:
                    break
                guess[u] = list(hint.symbols)
                used_slots.add(u)
                placed += 1

        # 3. Run per-slot factor probing / exponent resolution for remaining slots
        prime_order = list(range(config.num_primes))
        self.rng.shuffle(prime_order)
        for i in range(config.code_length):
            if guess[i] is None:
                guess[i] = self._pick_slot_move(i, knowledge, prime_order)

        return [g if g is not None else [0] for g in guess]
