"""Sound mathematical constraint propagation for Gödel-Wordle (Prime Decoding).

Combines per-slot prime-factor deduction with exact global MISPLACED multiset
counting across all turns, iterating to a fixed point without ever eliminating
a valid target.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Iterable, Sequence

from engine.game import Feedback, GameConfig, GameSession, TurnRecord, evaluate_guess
from engine.primes import encode_slot, get_primes


def symbols_to_mask(symbols: Iterable[int]) -> int:
    """Convert an iterable of 0-based symbol indices into a prime-presence bitmask."""
    mask = 0
    for s in symbols:
        mask |= 1 << s
    return mask


def mask_to_symbols(mask: int) -> list[int]:
    """Decode a bitmask into sorted 0-based symbol indices."""
    out: list[int] = []
    idx = 0
    m = mask
    while m:
        if m & 1:
            out.append(idx)
        m >>= 1
        idx += 1
    return out


def slot_multiset_prior_weight(
    symbols: Sequence[int], num_primes: int
) -> float:
    """Relative prior probability of a sorted multiset under `generate_random_slot`.

    `generate_random_slot` picks k uniformly in 1..K (or fixed K), then draws k
    symbols i.i.d. uniformly from 0..P-1 and sorts them. Therefore, up to a
    constant factor (1/K or 1), P(multiset) = (k! / prod(e_i!)) / P^k.
    """
    k = len(symbols)
    counts = Counter(symbols)
    denom = 1
    for c in counts.values():
        denom *= math.factorial(c)
    multinomial = math.factorial(k) // denom
    return multinomial / (num_primes ** k)


def count_slot_candidates(config: GameConfig) -> int:
    """Return total number of distinct legal multisets for a single slot."""
    p = config.num_primes
    k_max = config.max_multiplier
    if not config.allow_underfill:
        return math.comb(p + k_max - 1, k_max)
    return sum(math.comb(p + k - 1, k) for k in range(1, k_max + 1))


@dataclass(frozen=True)
class SlotCandidate:
    symbols: tuple[int, ...]
    product: int
    mask: int
    prior: float


def enumerate_all_slot_candidates(config: GameConfig) -> list[SlotCandidate]:
    """Enumerate all legal single-slot multisets (for small/medium spaces)."""
    primes = get_primes(config.num_primes)
    k_min = 1 if config.allow_underfill else config.max_multiplier
    candidates: list[SlotCandidate] = []
    for k in range(k_min, config.max_multiplier + 1):
        for combo in combinations_with_replacement(range(config.num_primes), k):
            prod = encode_slot(combo, primes)
            mask = symbols_to_mask(combo)
            prior = slot_multiset_prior_weight(combo, config.num_primes)
            candidates.append(
                SlotCandidate(symbols=combo, product=prod, mask=mask, prior=prior)
            )
    return candidates


def enumerate_support_candidates(
    support_symbols: Sequence[int],
    config: GameConfig,
    primes: Sequence[int],
) -> list[SlotCandidate]:
    """Enumerate all legal multisets whose EXACT set of distinct primes is `support_symbols`.

    For K <= 8, this has at most C(8, 4) = 70 entries even when P = 32!
    Results are sorted in descending Bayesian prior probability order.
    """
    sup = sorted(set(support_symbols))
    d = len(sup)
    k_max = config.max_multiplier
    if d == 0 or d > k_max:
        return []
    k_min = d if config.allow_underfill else k_max
    if k_min < d:
        return []

    candidates: list[SlotCandidate] = []
    base_mask = symbols_to_mask(sup)

    for total_k in range(k_min, k_max + 1):
        extra = total_k - d
        if extra == 0:
            combo = tuple(sup)
            candidates.append(
                SlotCandidate(
                    symbols=combo,
                    product=encode_slot(combo, primes),
                    mask=base_mask,
                    prior=slot_multiset_prior_weight(combo, config.num_primes),
                )
            )
        else:
            for extra_syms in combinations_with_replacement(sup, extra):
                combo = tuple(sorted((*sup, *extra_syms)))
                candidates.append(
                    SlotCandidate(
                        symbols=combo,
                        product=encode_slot(combo, primes),
                        mask=base_mask,
                        prior=slot_multiset_prior_weight(combo, config.num_primes),
                    )
                )

    candidates.sort(key=lambda c: (-c.prior, len(c.symbols), c.symbols))
    return candidates


@dataclass(frozen=True)
class MisplacedHint:
    product: int
    symbols: tuple[int, ...]
    mask: int
    remaining_needed: int
    eligible_slots: tuple[int, ...]


class BoardKnowledge:
    """Sound constraint state derived from a GameSession's full history."""

    def __init__(self, session: GameSession) -> None:
        self.config: GameConfig = session.config
        self.primes: list[int] = session.primes
        self.num_primes: int = self.config.num_primes
        self.max_multiplier: int = self.config.max_multiplier
        self.code_length: int = self.config.code_length
        self.all_primes_mask: int = (1 << self.num_primes) - 1

        # Directly solved (EXACT) or mathematically deduced unique slot multisets
        self.solved_slots: dict[int, tuple[int, ...]] = {}
        self.solved_products: dict[int, int] = {}
        # Tracks slots that already received EXACT in history (need no verification)
        self.confirmed_exact: set[int] = set()

        # Per-slot sound factor constraints (bitmasks over 0..P-1)
        self.excluded_mask: list[int] = [0] * self.code_length
        self.required_mask: list[int] = [0] * self.code_length
        self.partial_clauses: list[list[int]] = [[] for _ in range(self.code_length)]
        self.forbidden_products: list[set[int]] = [
            set() for _ in range(self.code_length)
        ]
        # Products actually guessed at slot i in prior turns (subset of forbidden_products)
        self.tried_at_slot: list[set[int]] = [
            set() for _ in range(self.code_length)
        ]

        # Track product -> canonical sorted symbols for any product seen in history
        self.product_to_symbols: dict[int, tuple[int, ...]] = {}
        self.product_to_mask: dict[int, int] = {}

        # Active global MISPLACED placement hints after fixed-point propagation
        self.misplaced_hints: list[MisplacedHint] = []

        self._ingest_history(session.history)

    def _lock_slot(self, slot_idx: int, symbols: tuple[int, ...], prod: int) -> bool:
        """Mark slot_idx as uniquely determined. Returns True if newly locked."""
        if slot_idx in self.solved_slots:
            return False
        self.solved_slots[slot_idx] = symbols
        self.solved_products[slot_idx] = prod
        m = symbols_to_mask(symbols)
        self.required_mask[slot_idx] = m
        self.excluded_mask[slot_idx] = self.all_primes_mask & ~m
        self.partial_clauses[slot_idx] = []
        if not self.config.allow_duplicate_slots:
            for other in range(self.code_length):
                if other != slot_idx:
                    self.forbidden_products[other].add(prod)
        return True

    def is_product_compatible_with_slot(
        self, slot_idx: int, prod: int, mask: int, length: int
    ) -> bool:
        """Check whether a candidate product satisfies all known constraints at slot_idx."""
        if slot_idx in self.solved_products:
            return self.solved_products[slot_idx] == prod
        if prod in self.forbidden_products[slot_idx]:
            return False
        if length < 1 or length > self.max_multiplier:
            return False
        if not self.config.allow_underfill and length != self.max_multiplier:
            return False
        if mask & self.excluded_mask[slot_idx]:
            return False
        if (mask & self.required_mask[slot_idx]) != self.required_mask[slot_idx]:
            return False
        for clause in self.partial_clauses[slot_idx]:
            if (mask & clause) == 0:
                return False
        return True

    def _simplify_slot_clauses(self, slot_idx: int) -> bool:
        """Reduce PARTIAL clauses and promote forced requirements/exclusions."""
        if slot_idx in self.solved_slots:
            return False
        changed = False

        # If already have K distinct required primes, NO other prime can appear!
        if self.required_mask[slot_idx].bit_count() == self.max_multiplier:
            new_excl = self.all_primes_mask & ~self.required_mask[slot_idx]
            if new_excl != self.excluded_mask[slot_idx]:
                self.excluded_mask[slot_idx] = new_excl
                changed = True

        # Reduce partial clauses by stripping excluded primes
        new_clauses: list[int] = []
        seen: set[int] = set()
        for clause in self.partial_clauses[slot_idx]:
            reduced = clause & ~self.excluded_mask[slot_idx]
            if reduced == 0:
                # Should never happen for a valid history
                continue
            if reduced & self.required_mask[slot_idx]:
                # Already satisfied by a proven required prime
                continue
            if reduced.bit_count() == 1:
                if not (self.required_mask[slot_idx] & reduced):
                    self.required_mask[slot_idx] |= reduced
                    changed = True
                continue
            if reduced not in seen:
                seen.add(reduced)
                new_clauses.append(reduced)

        if len(new_clauses) != len(self.partial_clauses[slot_idx]):
            changed = True
        self.partial_clauses[slot_idx] = new_clauses

        # Re-check K-prime saturation after promoting new required primes
        if self.required_mask[slot_idx].bit_count() == self.max_multiplier:
            new_excl = self.all_primes_mask & ~self.required_mask[slot_idx]
            if new_excl != self.excluded_mask[slot_idx]:
                self.excluded_mask[slot_idx] = new_excl
                changed = True
            # When K distinct primes are required, multiset is UNIQUE (each exp == 1)!
            sup = tuple(mask_to_symbols(self.required_mask[slot_idx]))
            prod = encode_slot(sup, self.primes)
            if self._lock_slot(slot_idx, sup, prod):
                changed = True

        # If all primes are either required or excluded (unknown == 0), check if only 1
        # valid exponent assignment remains!
        if slot_idx not in self.solved_slots:
            unknown = (
                self.all_primes_mask
                & ~self.excluded_mask[slot_idx]
                & ~self.required_mask[slot_idx]
            )
            if unknown == 0 and self.required_mask[slot_idx] != 0:
                sup = mask_to_symbols(self.required_mask[slot_idx])
                cands = [
                    c
                    for c in enumerate_support_candidates(
                        sup, self.config, self.primes
                    )
                    if c.product not in self.forbidden_products[slot_idx]
                ]
                if len(cands) == 1:
                    if self._lock_slot(slot_idx, cands[0].symbols, cands[0].product):
                        changed = True

        return changed

    def _ingest_history(self, history: Sequence[TurnRecord]) -> None:
        # Step 1: Collect direct per-slot observations and per-turn global count constraints
        # Each global constraint: (unmatched_slots_tuple, prod, min_count, exact_count_flag)
        global_constraints: list[tuple[tuple[int, ...], int, int, bool]] = []

        for turn in history:
            unmatched: list[int] = []
            mis_counts: Counter[int] = Counter()
            other_counts: Counter[int] = Counter()

            for i in range(self.code_length):
                syms = tuple(sorted(turn.guess_slots[i]))
                prod = turn.guess_products[i]
                mask = symbols_to_mask(syms)
                self.product_to_symbols[prod] = syms
                self.product_to_mask[prod] = mask
                fb = turn.feedback[i]

                if fb == Feedback.EXACT:
                    self.confirmed_exact.add(i)
                    self._lock_slot(i, syms, prod)
                else:
                    unmatched.append(i)
                    self.forbidden_products[i].add(prod)
                    self.tried_at_slot[i].add(prod)
                    if fb == Feedback.WRONG:
                        self.excluded_mask[i] |= mask
                        other_counts[prod] += 1
                    elif fb == Feedback.PARTIAL:
                        self.partial_clauses[i].append(mask)
                        other_counts[prod] += 1
                    elif fb == Feedback.MISPLACED:
                        mis_counts[prod] += 1
                        if self.max_multiplier == 1:
                            # When K=1, each slot holds a single prime; != implies coprime.
                            self.excluded_mask[i] |= mask

            unmatched_tuple = tuple(unmatched)
            all_unmatched_prods = set(mis_counts.keys()) | set(other_counts.keys())
            for prod in all_unmatched_prods:
                m_mis = mis_counts[prod]
                m_other = other_counts[prod]
                is_exact = m_other > 0
                if is_exact and m_mis == 0:
                    # Rule A: prod cannot appear in ANY unmatched slot of this turn!
                    for u in unmatched_tuple:
                        self.forbidden_products[u].add(prod)
                elif m_mis > 0:
                    global_constraints.append(
                        (unmatched_tuple, prod, m_mis, is_exact)
                    )

        # Step 2: Iterate per-slot and global MISPLACED propagation to fixed point
        changed = True
        while changed:
            changed = False
            for i in range(self.code_length):
                if self._simplify_slot_clauses(i):
                    changed = True

            for unmatched_tuple, prod, m_mis, is_exact in global_constraints:
                syms = self.product_to_symbols[prod]
                mask = self.product_to_mask[prod]
                length = len(syms)

                known_matches = sum(
                    1
                    for u in unmatched_tuple
                    if self.solved_products.get(u) == prod
                )
                elig = [
                    u
                    for u in unmatched_tuple
                    if u not in self.solved_slots
                    and self.is_product_compatible_with_slot(u, prod, mask, length)
                ]
                rem_needed = m_mis - known_matches

                if is_exact and rem_needed <= 0:
                    for u in elig:
                        if prod not in self.forbidden_products[u]:
                            self.forbidden_products[u].add(prod)
                            changed = True
                elif rem_needed > 0 and len(elig) == rem_needed:
                    for u in elig:
                        if self._lock_slot(u, syms, prod):
                            changed = True

        # Step 3: Build clean deduplicated list of active MISPLACED hints for solvers
        best_hints: dict[int, MisplacedHint] = {}
        for unmatched_tuple, prod, m_mis, _is_exact in global_constraints:
            syms = self.product_to_symbols[prod]
            mask = self.product_to_mask[prod]
            length = len(syms)
            known_matches = sum(
                1 for u in unmatched_tuple if self.solved_products.get(u) == prod
            )
            rem_needed = m_mis - known_matches
            if rem_needed <= 0:
                continue
            elig = tuple(
                u
                for u in unmatched_tuple
                if u not in self.solved_slots
                and self.is_product_compatible_with_slot(u, prod, mask, length)
            )
            if not elig:
                continue
            hint = MisplacedHint(
                product=prod,
                symbols=syms,
                mask=mask,
                remaining_needed=rem_needed,
                eligible_slots=elig,
            )
            prev = best_hints.get(prod)
            # Prefer tighter hint (fewer eligible slots or higher remaining_needed)
            if prev is None or (
                len(hint.eligible_slots) < len(prev.eligible_slots)
                or (
                    len(hint.eligible_slots) == len(prev.eligible_slots)
                    and hint.remaining_needed > prev.remaining_needed
                )
            ):
                best_hints[prod] = hint

        self.misplaced_hints = sorted(
            best_hints.values(),
            key=lambda h: (len(h.eligible_slots) / max(1, h.remaining_needed), len(h.eligible_slots)),
        )

    def unknown_mask(self, slot_idx: int) -> int:
        return (
            self.all_primes_mask
            & ~self.excluded_mask[slot_idx]
            & ~self.required_mask[slot_idx]
        )


def is_board_consistent(
    candidate_products: Sequence[int], history: Sequence[TurnRecord]
) -> bool:
    """Verify whether a full board of slot products reproduces all historical feedbacks."""
    for turn in history:
        if evaluate_guess(turn.guess_products, candidate_products) != turn.feedback:
            return False
    return True
