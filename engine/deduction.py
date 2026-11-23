"""Sound per-slot deduction engine ("记牌器" / card counter).

Given a GameSession's history, computes mathematically guaranteed per-slot
constraints without leaking the secret target:
  - WRONG at slot i       => every prime factor in guess[i] is 100% EXCLUDED at slot i.
  - EXACT at slot i       => slot i is SOLVED (exact multiset known; all other symbols excluded).
  - PARTIAL at slot i     => at least one prime factor in guess[i] is present at slot i.
                             After removing excluded symbols from that clue:
                             * if 1 symbol remains -> 100% REQUIRED at slot i;
                             * if >=2 remain and none is already required -> UNCERTAIN ('?').
  - MISPLACED at slot i   => intentionally NOT treated as factor-exclusion because
                             priority is MISPLACED > PARTIAL (it may still share factors).
"""

from __future__ import annotations

from typing import Any

from .game import Feedback, GameSession


def analyze_slot(session: GameSession, slot_idx: int) -> dict[str, Any]:
    """Compute sound deduction state for a single slot index."""
    num_primes = session.config.num_primes
    excluded: set[int] = set()
    partial_clues: list[set[int]] = []
    tried_non_exact_products: set[str] = set()
    exact_slot: list[int] | None = None

    for turn in session.history:
        fb = turn.feedback[slot_idx]
        syms = set(turn.guess_slots[slot_idx])
        prod_str = str(turn.guess_products[slot_idx])

        if fb == Feedback.EXACT:
            exact_slot = sorted(turn.guess_slots[slot_idx])
        elif fb == Feedback.WRONG:
            excluded.update(syms)
            tried_non_exact_products.add(prod_str)
        elif fb == Feedback.PARTIAL:
            partial_clues.append(syms)
            tried_non_exact_products.add(prod_str)
        elif fb == Feedback.MISPLACED:
            tried_non_exact_products.add(prod_str)

    if exact_slot is not None:
        exact_unique = set(exact_slot)
        all_syms = set(range(num_primes))
        excluded = all_syms - exact_unique
        per_symbol = {
            str(s): ("exact" if s in exact_unique else "excluded")
            for s in range(num_primes)
        }
        return {
            "slot": slot_idx,
            "solved": True,
            "exact_slot": exact_slot,
            "excluded": sorted(excluded),
            "required": sorted(exact_unique),
            "uncertain": [],
            "tried_non_exact_products": sorted(tried_non_exact_products),
            "per_symbol": per_symbol,
        }

    # Deduce required symbols from PARTIAL clues once excluded symbols are removed
    required: set[int] = set()
    reduced_clues: list[set[int]] = []
    for clue in partial_clues:
        rem = clue - excluded
        if len(rem) == 1:
            required.update(rem)
        elif len(rem) > 1:
            reduced_clues.append(rem)

    # Symbols appearing in unresolved PARTIAL clues (not already satisfied by a required symbol)
    uncertain: set[int] = set()
    for rem in reduced_clues:
        if rem.isdisjoint(required):
            uncertain.update(rem)

    per_symbol = {}
    for s in range(num_primes):
        if s in excluded:
            per_symbol[str(s)] = "excluded"
        elif s in required:
            per_symbol[str(s)] = "required"
        elif s in uncertain:
            per_symbol[str(s)] = "uncertain"
        else:
            per_symbol[str(s)] = "unknown"

    return {
        "slot": slot_idx,
        "solved": False,
        "exact_slot": None,
        "excluded": sorted(excluded),
        "required": sorted(required),
        "uncertain": sorted(uncertain),
        "tried_non_exact_products": sorted(tried_non_exact_products),
        "per_symbol": per_symbol,
    }


def analyze_session(session: GameSession) -> dict[str, Any]:
    """Compute per-slot sound deduction for the entire board."""
    slots = [analyze_slot(session, i) for i in range(session.config.code_length)]
    return {
        "slots": slots,
        "solved_count": sum(1 for s in slots if s["solved"]),
    }
