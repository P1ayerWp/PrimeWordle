"""Comprehensive unit tests for the Gödel-Wordle (Prime Decoding) core engine."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from engine.deduction import analyze_session
from engine.game import (
    Feedback,
    GameConfig,
    GameSession,
    evaluate_guess,
    load_saved_games,
    save_game_jsonl,
)
from engine.primes import encode_slot, factorize_slot, format_slot_powers, get_primes
from engine.tokenizer import PrimeWordleTokenizer


class TestPrimes(unittest.TestCase):
    def test_prime_generation(self) -> None:
        self.assertEqual(get_primes(10), [2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
        self.assertEqual(get_primes(30)[-1], 113)

    def test_order_invariance_and_powers(self) -> None:
        # 2 * 3 * 5 == 5 * 2 * 3 == 30
        self.assertEqual(encode_slot([0, 1, 2]), 30)
        self.assertEqual(encode_slot([2, 0, 1]), 30)
        # 2^3 == 8
        self.assertEqual(encode_slot([0, 0, 0]), 8)
        self.assertEqual(factorize_slot(8, 5), [0, 0, 0])
        self.assertEqual(format_slot_powers([2, 0, 0, 1, 0]), "2^3 * 3 * 5")

    def test_large_k10_big_integer(self) -> None:
        # 10 copies of the 30th prime (113): 113^10 > 2^53 (JS MAX_SAFE_INTEGER)
        primes = get_primes(30)
        prod = encode_slot([29] * 10, primes)
        self.assertEqual(prod, 113**10)
        self.assertEqual(factorize_slot(prod, 30), [29] * 10)


class TestEvaluateGuess(unittest.TestCase):
    def test_priority_exact_over_all(self) -> None:
        # target: [6, 10, 15] ({2,3}, {2,5}, {3,5})
        fb = evaluate_guess([6, 10, 15], [6, 10, 15])
        self.assertEqual(fb, [Feedback.EXACT, Feedback.EXACT, Feedback.EXACT])

    def test_priority_misplaced_over_partial(self) -> None:
        # Slot 0: target=6 ({2,3}), guess=10 ({2,5})
        # 10 shares prime 2 with 6 (PARTIAL), AND 10 exists at target slot 1 (MISPLACED).
        # Priority MISPLACED > PARTIAL -> slot 0 MUST be MISPLACED.
        # Slot 1: target=10 ({2,5}), guess=15 ({3,5}) -> not in remaining, shares 5 -> PARTIAL.
        fb = evaluate_guess([10, 15], [6, 10])
        self.assertEqual(fb, [Feedback.MISPLACED, Feedback.PARTIAL])

    def test_duplicate_guess_consumes_pool_once_then_falls_back(self) -> None:
        # target: [6 ({2,3}), 15 ({3,5})]
        # guess:  [15, 15]
        # Slot 1 is EXACT (15==15). Remaining pool only has {6: 1}.
        # Slot 0 guess=15 is NOT in pool anymore, but gcd(15, 6)=3 > 1 -> PARTIAL!
        fb = evaluate_guess([15, 15], [6, 15])
        self.assertEqual(fb, [Feedback.PARTIAL, Feedback.EXACT])

        # target: [6 ({2,3}), 35 ({5,7})]
        # guess:  [35, 35]
        # Slot 1 is EXACT. Slot 0 guess=35 is coprime with 6 -> WRONG!
        fb2 = evaluate_guess([35, 35], [6, 35])
        self.assertEqual(fb2, [Feedback.WRONG, Feedback.EXACT])

    def test_powers_partial_matching(self) -> None:
        # target: [8 (2^3), 2 (2^1)]
        # guess:  [2 (2^1), 4 (2^2)]
        # Slot 0: 2 exists in target slot 1 -> MISPLACED!
        # Slot 1: 4 does not exist in target, gcd(4, 2) = 2 > 1 -> PARTIAL!
        fb = evaluate_guess([2, 4], [8, 2])
        self.assertEqual(fb, [Feedback.MISPLACED, Feedback.PARTIAL])

        # Single slot power mismatch: target=8, guess=2 -> PARTIAL
        self.assertEqual(evaluate_guess([2], [8]), [Feedback.PARTIAL])
        self.assertEqual(evaluate_guess([8], [2]), [Feedback.PARTIAL])


class TestGameSessionAndPersistence(unittest.TestCase):
    def test_full_session_unordered_win_and_jsonl_roundtrip(self) -> None:
        cfg = GameConfig(
            num_primes=10,
            max_multiplier=5,
            code_length=3,
            allow_underfill=True,
        )
        # Slot 0: 2^3 (8), Slot 1: 3*5 (15), Slot 2: 29^5 (huge)
        target = [[0, 0, 0], [1, 2], [9, 9, 9, 9, 9]]
        session = GameSession(cfg, seed=42, target_slots=target)

        # Turn 1: wrong + partial + misplaced
        t1 = session.submit_guess([[3], [2, 4], [1, 2]])
        self.assertEqual(t1.feedback, [Feedback.WRONG, Feedback.PARTIAL, Feedback.MISPLACED])
        self.assertEqual(session.status, "playing")

        # Turn 2: unordered exact match! ([2, 1] == [1, 2] == 15)
        t2 = session.submit_guess([[0, 0, 0], [2, 1], [9, 9, 9, 9, 9]])
        self.assertEqual(t2.feedback, [Feedback.EXACT, Feedback.EXACT, Feedback.EXACT])
        self.assertEqual(session.status, "won")
        self.assertEqual(len(session.history), 2)

        # Save and reload via JSONL
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "games.jsonl"
            save_game_jsonl(session, path)
            loaded = load_saved_games(path)
            self.assertEqual(len(loaded), 1)
            restored = GameSession.from_dict(loaded[0])
            self.assertEqual(restored.target_products, session.target_products)
            self.assertEqual(len(restored.history), 2)
            self.assertEqual(restored.status, "won")

    def test_underfill_disabled_enforcement(self) -> None:
        cfg = GameConfig(
            num_primes=5, max_multiplier=3, code_length=2, allow_underfill=False
        )
        session = GameSession(cfg, seed=7)
        with self.assertRaises(ValueError):
            session.submit_guess([[0, 1], [0, 1, 2]])  # Slot 0 has only 2 != 3

    def test_tokenizer_serialization(self) -> None:
        cfg = GameConfig(num_primes=5, max_multiplier=3, code_length=2, allow_underfill=True)
        session = GameSession(cfg, seed=1, target_slots=[[0, 1], [2]])
        session.submit_guess([[0, 1], [3]])
        tok = PrimeWordleTokenizer(max_symbols=30)
        ids = tok.encode_session(
            session, reveal_k=False, reveal_underfill=False, reveal_dup=True
        )
        decoded = tok.decode(ids)
        self.assertEqual(
            decoded,
            [
                "<BOS>", "<CFG>", "N_5", "N_2", "<UNK_CFG>", "<UNK_CFG>", "<DUP_ON>",
                "<TURN>", "<SLOT>", "S_0", "S_1", "<SLOT>", "S_3",
                "<FB>", "EXACT", "WRONG",
            ],
        )

    def test_per_slot_deduction_card_counter(self) -> None:
        # Mirrors a real 5/2/5 game: target slot 0 is [2] (prime 5)
        cfg = GameConfig(num_primes=5, max_multiplier=2, code_length=2, allow_underfill=True)
        session = GameSession(cfg, seed=9, target_slots=[[2], [1, 4]])

        # Turn 1: slot 0 guesses [0, 2] (2*5=10 -> PARTIAL), slot 1 guesses [0, 3] (2*7=14 -> WRONG)
        session.submit_guess([[0, 2], [0, 3]])
        d1 = analyze_session(session)["slots"][0]
        self.assertEqual(d1["excluded"], [])
        self.assertEqual(d1["required"], [])
        self.assertEqual(d1["uncertain"], [0, 2])

        # Turn 2: slot 0 guesses [1, 3] (3*7=21 -> WRONG -> excludes 1, 3)
        session.submit_guess([[1, 3], [0, 3]])
        d2 = analyze_session(session)["slots"][0]
        self.assertEqual(d2["excluded"], [1, 3])
        self.assertEqual(d2["uncertain"], [0, 2])

        # Turn 3: slot 0 guesses [2, 3] (5*7=35 -> PARTIAL; since 3 is excluded, 2 MUST be required!)
        session.submit_guess([[2, 3], [1, 4]])
        d3 = analyze_session(session)
        self.assertEqual(d3["slots"][0]["required"], [2])
        self.assertEqual(d3["slots"][0]["uncertain"], [])
        self.assertTrue(d3["slots"][1]["solved"])
        self.assertEqual(d3["solved_count"], 1)


if __name__ == "__main__":
    unittest.main()
