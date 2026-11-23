"""Prime number utilities and Gödel-product slot encoding.

Each base symbol index i (0-based) maps bijectively to the i-th prime p_i
(2, 3, 5, 7, 11, ...). A slot containing any multiset of symbols maps uniquely
to the product of their corresponding primes (Fundamental Theorem of Arithmetic),
making verification order-invariant and O(1) via equality and math.gcd.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence


def generate_primes(count: int) -> list[int]:
    """Generate the first `count` prime numbers."""
    if count <= 0:
        return []
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        is_prime = True
        limit = math.isqrt(candidate)
        for p in primes:
            if p > limit:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1 if candidate == 2 else 2
    return primes


DEFAULT_MAX_PRIMES = 64
PRIMES: tuple[int, ...] = tuple(generate_primes(DEFAULT_MAX_PRIMES))


def get_primes(count: int) -> list[int]:
    """Return the first `count` primes."""
    if count <= len(PRIMES):
        return list(PRIMES[:count])
    return generate_primes(count)


def encode_slot(symbol_indices: Iterable[int], primes: Sequence[int] = PRIMES) -> int:
    """Encode a multiset (any order) of 0-based symbol indices into its Gödel product."""
    product = 1
    has_any = False
    for idx in symbol_indices:
        if idx < 0 or idx >= len(primes):
            raise ValueError(f"Symbol index {idx} out of range [0, {len(primes) - 1}]")
        product *= primes[idx]
        has_any = True
    if not has_any:
        raise ValueError("A slot must contain at least one symbol.")
    return product


def factorize_slot(product: int, num_primes: int) -> list[int]:
    """Factorize a Gödel product back into sorted 0-based symbol indices."""
    if product < 2:
        raise ValueError(f"Invalid slot product {product}; expected integer >= 2.")
    primes = get_primes(num_primes)
    factors: list[int] = []
    rem = product
    for idx, p in enumerate(primes):
        while rem % p == 0:
            factors.append(idx)
            rem //= p
        if rem == 1:
            break
    if rem != 1:
        raise ValueError(
            f"Product {product} contains prime factors outside the first {num_primes} primes."
        )
    return factors


def slot_to_exponents(symbol_indices: Iterable[int]) -> list[tuple[int, int]]:
    """Convert symbol indices into sorted (symbol_index, exponent) pairs."""
    counts = Counter(symbol_indices)
    return sorted(counts.items(), key=lambda kv: kv[0])


def format_slot_powers(
    symbol_indices: Iterable[int], primes: Sequence[int] = PRIMES
) -> str:
    """Format a slot as a human-readable prime factorization, e.g. '2^3 * 3 * 5'."""
    parts = []
    for idx, exp in slot_to_exponents(symbol_indices):
        p = primes[idx]
        parts.append(f"{p}^{exp}" if exp > 1 else f"{p}")
    return " * ".join(parts)
