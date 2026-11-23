"""Core Gödel-Wordle (Prime Decoding) game engine.

Design invariants:
- Each slot holds 1..K symbols (or exactly K when allow_underfill=False),
  allowing repeated symbols (prime powers) and arbitrary symbol order.
- Verification uses Gödel products (unique prime factorization) with strict
  4-state priority:
      EXACT > MISPLACED > PARTIAL > WRONG
- All products are stored/exported as strings in JSON so arbitrary precision
  (e.g. K=10 large primes) never loses precision in JavaScript.
"""

from __future__ import annotations

import json
import math
import random
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from .primes import encode_slot, get_primes


class Feedback(str, Enum):
    EXACT = "EXACT"          # 完全正确: exact same multiset in this slot
    MISPLACED = "MISPLACED"  # 栏位错误: exact multiset exists in another unmatched slot
    PARTIAL = "PARTIAL"      # 部分正确: shares >= 1 prime factor with target in this slot
    WRONG = "WRONG"          # 完全错误: neither misplaced nor shares any factor here


@dataclass(frozen=True)
class GameConfig:
    num_primes: int = 5             # P: number of available base symbols (primes)
    max_multiplier: int = 2         # K: max symbols multiplied per slot
    code_length: int = 5            # L: number of slots
    allow_underfill: bool = True    # If True, slots may have 1..K symbols; else exact K
    allow_duplicate_slots: bool = True  # Whether two slots in target may have same product
    max_turns: int | None = None    # None = unlimited (supports 50+ turns freely)

    def __post_init__(self) -> None:
        if self.num_primes < 2 or self.num_primes > 50:
            raise ValueError("num_primes (P) must be between 2 and 50.")
        if self.max_multiplier < 1 or self.max_multiplier > 12:
            raise ValueError("max_multiplier (K) must be between 1 and 12.")
        if self.code_length < 1 or self.code_length > 32:
            raise ValueError("code_length (L) must be between 1 and 32.")
        if self.max_turns is not None and self.max_turns < 1:
            raise ValueError("max_turns must be >= 1 or None.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_primes": self.num_primes,
            "max_multiplier": self.max_multiplier,
            "code_length": self.code_length,
            "allow_underfill": self.allow_underfill,
            "allow_duplicate_slots": self.allow_duplicate_slots,
            "max_turns": self.max_turns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameConfig":
        return cls(
            num_primes=int(data.get("num_primes", 5)),
            max_multiplier=int(data.get("max_multiplier", 2)),
            code_length=int(data.get("code_length", 5)),
            allow_underfill=bool(data.get("allow_underfill", True)),
            allow_duplicate_slots=bool(data.get("allow_duplicate_slots", True)),
            max_turns=(
                int(data["max_turns"]) if data.get("max_turns") is not None else None
            ),
        )


def evaluate_guess(
    guess_products: Sequence[int], target_products: Sequence[int]
) -> list[Feedback]:
    """Evaluate a guess against the target using Gödel products.

    Priority: EXACT > MISPLACED > PARTIAL > WRONG.
    Uses standard 2-pass multiset counting for MISPLACED so duplicate guesses
    only consume remaining unmatched target slots.
    """
    n = len(target_products)
    if len(guess_products) != n:
        raise ValueError(
            f"Guess length {len(guess_products)} does not match target length {n}."
        )

    result: list[Feedback | None] = [None] * n
    remaining: Counter[int] = Counter()

    # Pass 1: exact matches
    for i in range(n):
        if guess_products[i] == target_products[i]:
            result[i] = Feedback.EXACT
        else:
            remaining[target_products[i]] += 1

    # Pass 2: misplaced > partial > wrong
    for i in range(n):
        if result[i] is not None:
            continue
        g = guess_products[i]
        if remaining[g] > 0:
            result[i] = Feedback.MISPLACED
            remaining[g] -= 1
        elif math.gcd(g, target_products[i]) > 1:
            result[i] = Feedback.PARTIAL
        else:
            result[i] = Feedback.WRONG

    return [fb for fb in result if fb is not None]


@dataclass
class TurnRecord:
    turn: int
    guess_slots: list[list[int]]    # Raw symbol index lists as submitted (preserves order!)
    guess_products: list[int]       # Gödel products
    feedback: list[Feedback]        # Per-slot feedback
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "guess_slots": self.guess_slots,
            "guess_products": [str(p) for p in self.guess_products],
            "feedback": [fb.value for fb in self.feedback],
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TurnRecord":
        return cls(
            turn=int(data["turn"]),
            guess_slots=[[int(x) for x in slot] for slot in data["guess_slots"]],
            guess_products=[int(p) for p in data["guess_products"]],
            feedback=[Feedback(fb) for fb in data["feedback"]],
            timestamp=str(data.get("timestamp", "")),
        )


def _count_unique_possible_slots(num_primes: int, max_multiplier: int, allow_underfill: bool) -> int:
    """Count total distinct multisets (with repetition) available for a single slot."""
    if not allow_underfill:
        return math.comb(num_primes + max_multiplier - 1, max_multiplier)
    return sum(
        math.comb(num_primes + k - 1, k) for k in range(1, max_multiplier + 1)
    )


def generate_random_slot(
    config: GameConfig, rng: random.Random
) -> list[int]:
    """Generate one random slot (sorted 0-based symbol indices)."""
    k = (
        rng.randint(1, config.max_multiplier)
        if config.allow_underfill
        else config.max_multiplier
    )
    symbols = [rng.randrange(config.num_primes) for _ in range(k)]
    symbols.sort()
    return symbols


def generate_target(
    config: GameConfig, rng: random.Random
) -> tuple[list[list[int]], list[int]]:
    """Generate a secret target (symbol lists + Gödel products)."""
    primes = get_primes(config.num_primes)
    max_unique = _count_unique_possible_slots(
        config.num_primes, config.max_multiplier, config.allow_underfill
    )
    if not config.allow_duplicate_slots and config.code_length > max_unique:
        raise ValueError(
            f"Cannot generate {config.code_length} unique slots with only "
            f"{max_unique} distinct combinations available."
        )

    slots: list[list[int]] = []
    products: list[int] = []
    used_products: set[int] = set()

    for _ in range(config.code_length):
        for _attempt in range(10_000):
            slot = generate_random_slot(config, rng)
            prod = encode_slot(slot, primes)
            if config.allow_duplicate_slots or prod not in used_products:
                slots.append(slot)
                products.append(prod)
                used_products.add(prod)
                break
        else:
            raise RuntimeError("Failed to sample unique target slots.")

    return slots, products


class GameSession:
    """Stateful game session shared by human Web UI, solvers, and RL environments."""

    def __init__(
        self,
        config: GameConfig,
        seed: int | None = None,
        target_slots: list[list[int]] | None = None,
        game_id: str | None = None,
    ) -> None:
        self.game_id: str = game_id or uuid.uuid4().hex[:12]
        self.config: GameConfig = config
        self.seed: int = seed if seed is not None else random.SystemRandom().randrange(2**31)
        self.primes: list[int] = get_primes(config.num_primes)
        self.created_at: str = datetime.now(timezone.utc).isoformat()

        rng = random.Random(self.seed)
        if target_slots is not None:
            self._validate_slots(target_slots, enforce_exact_if_disabled=True)
            self.target_slots: list[list[int]] = [sorted(s) for s in target_slots]
            self.target_products: list[int] = [
                encode_slot(s, self.primes) for s in self.target_slots
            ]
        else:
            self.target_slots, self.target_products = generate_target(config, rng)

        self.history: list[TurnRecord] = []
        self.status: str = "playing"  # "playing" | "won" | "lost" | "abandoned"
        self.peeked: bool = False     # Tracks whether cheat/peek was used
        self.notes: dict[str, Any] = {
            "symbol_marks": {},       # symbol_index (str) -> "none"|"confirmed"|"excluded"|"suspect"
            "scratchpad": "",
        }

    def _validate_slots(
        self, slots: Sequence[Sequence[int]], enforce_exact_if_disabled: bool = True
    ) -> None:
        if len(slots) != self.config.code_length:
            raise ValueError(
                f"Expected {self.config.code_length} slots, got {len(slots)}."
            )
        for i, slot in enumerate(slots):
            if len(slot) < 1:
                raise ValueError(f"Slot {i + 1} cannot be empty (must have >= 1 symbol).")
            if len(slot) > self.config.max_multiplier:
                raise ValueError(
                    f"Slot {i + 1} has {len(slot)} symbols, exceeding max_multiplier={self.config.max_multiplier}."
                )
            if (
                enforce_exact_if_disabled
                and not self.config.allow_underfill
                and len(slot) != self.config.max_multiplier
            ):
                raise ValueError(
                    f"Underfill is disabled: slot {i + 1} must have exactly {self.config.max_multiplier} symbols."
                )
            for idx in slot:
                if not isinstance(idx, int) or idx < 0 or idx >= self.config.num_primes:
                    raise ValueError(
                        f"Invalid symbol index {idx} in slot {i + 1} (valid: 0..{self.config.num_primes - 1})."
                    )

    def submit_guess(self, guess_slots: Sequence[Sequence[int]]) -> TurnRecord:
        """Validate and evaluate one turn; appends permanently to self.history."""
        if self.status != "playing":
            raise ValueError(f"Game is already over (status={self.status}).")

        clean_slots = [[int(x) for x in slot] for slot in guess_slots]
        self._validate_slots(clean_slots, enforce_exact_if_disabled=True)

        guess_products = [encode_slot(s, self.primes) for s in clean_slots]
        feedback = evaluate_guess(guess_products, self.target_products)

        record = TurnRecord(
            turn=len(self.history) + 1,
            guess_slots=clean_slots,
            guess_products=guess_products,
            feedback=feedback,
        )
        self.history.append(record)

        if all(fb == Feedback.EXACT for fb in feedback):
            self.status = "won"
        elif self.config.max_turns is not None and len(self.history) >= self.config.max_turns:
            self.status = "lost"

        return record

    def to_dict(self, include_target: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "game_id": self.game_id,
            "created_at": self.created_at,
            "seed": self.seed,
            "config": self.config.to_dict(),
            "primes": self.primes,
            "status": self.status,
            "turns": len(self.history),
            "peeked": self.peeked,
            "history": [rec.to_dict() for rec in self.history],
            "notes": self.notes,
        }
        if include_target or self.status != "playing" or self.peeked:
            payload["target"] = {
                "slots": self.target_slots,
                "products": [str(p) for p in self.target_products],
            }
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameSession":
        config = GameConfig.from_dict(data["config"])
        target_slots = data["target"]["slots"] if "target" in data else None
        session = cls(
            config=config,
            seed=int(data["seed"]),
            target_slots=target_slots,
            game_id=str(data["game_id"]),
        )
        session.created_at = str(data.get("created_at", session.created_at))
        session.status = str(data.get("status", "playing"))
        session.peeked = bool(data.get("peeked", False))
        session.history = [TurnRecord.from_dict(r) for r in data.get("history", [])]
        if "notes" in data and isinstance(data["notes"], dict):
            session.notes = data["notes"]
        return session


def save_game_jsonl(session: GameSession, path: str | Path) -> Path:
    """Append or update a game record in a JSONL file (always stores target for replay/training)."""
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    record = session.to_dict(include_target=True)
    existing: list[dict[str, Any]] = []
    if filepath.exists():
        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("game_id") != session.game_id:
                        existing.append(obj)
                except json.JSONDecodeError:
                    continue

    existing.append(record)
    with filepath.open("w", encoding="utf-8") as f:
        for obj in existing:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return filepath


def load_saved_games(path: str | Path) -> list[dict[str, Any]]:
    """Load all saved game summaries/records from a JSONL file (newest first)."""
    filepath = Path(path)
    if not filepath.exists():
        return []
    games: list[dict[str, Any]] = []
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                games.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    games.reverse()
    return games
