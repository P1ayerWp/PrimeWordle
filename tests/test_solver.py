"""Unit tests for the Phase 2 constraint engine, solvers, 77-vocab tokenizer, and dataset generator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from engine.game import Feedback, GameConfig, GameSession
from engine.tokenizer import PrimeWordleTokenizer
from solver.constraints import BoardKnowledge, is_board_consistent
from solver.exact_solver import ExactPoolSolver
from solver.factor_solver import FactorProbeSolver
from solver.generate_dataset import generate_dataset
from solver.hybrid import HybridSolver, solve_session


class TestTokenizer80AndMask(unittest.TestCase):
    def test_default_vocab_is_80_and_loss_mask_alignment(self) -> None:
        tok = PrimeWordleTokenizer()
        self.assertEqual(tok.vocab_size, 80)
        self.assertIn("S_31", tok.stoi)
        self.assertNotIn("S_32", tok.stoi)
        self.assertIn("N_32", tok.stoi)
        self.assertNotIn("N_33", tok.stoi)
        for extra in ("<DUP_ON>", "<DUP_OFF>", "<UNK_CFG>"):
            self.assertIn(extra, tok.stoi)

        cfg = GameConfig(
            num_primes=32,
            max_multiplier=8,
            code_length=32,
            allow_underfill=True,
            allow_duplicate_slots=False,
        )
        target = [[i % 32] for i in range(32)]
        session = GameSession(cfg, seed=1, target_slots=target)
        session.submit_guess(target)  # Instant win in 1 turn

        ids, mask = tok.encode_session_with_loss_mask(session, reveal_k=False)
        self.assertEqual(len(ids), len(mask))
        decoded = tok.decode(ids)

        # Fixed 7-token header must be masked out (0)
        self.assertEqual(
            decoded[:7],
            ["<BOS>", "<CFG>", "N_32", "N_32", "<UNK_CFG>", "<UNDERFILL_ON>", "<DUP_OFF>"],
        )
        self.assertEqual(mask[:7], [0, 0, 0, 0, 0, 0, 0])

        # Model guess (<TURN> ... <FB>) must be masked in (1)
        fb_idx = decoded.index("<FB>")
        self.assertEqual(decoded[7], "<TURN>")
        self.assertTrue(all(m == 1 for m in mask[7 : fb_idx + 1]))

        # Environment feedbacks (32 x EXACT) must be masked out (0)
        self.assertEqual(decoded[fb_idx + 1 : fb_idx + 33], ["EXACT"] * 32)
        self.assertTrue(all(m == 0 for m in mask[fb_idx + 1 : fb_idx + 33]))

        # Final <EOS> must be masked in (1)
        self.assertEqual(decoded[-1], "<EOS>")
        self.assertEqual(mask[-1], 1)


class TestConstraintsAndSolvers(unittest.TestCase):
    def test_global_misplaced_exact_lock_and_zero_exclusion(self) -> None:
        # Target: slot 0 = [0], slot 1 = [1], slot 2 = [2]
        cfg = GameConfig(num_primes=5, max_multiplier=1, code_length=3, allow_underfill=False)
        session = GameSession(cfg, seed=1, target_slots=[[0], [1], [2]])
        # Guess: [1, 3, 2] -> slot 2 EXACT, slot 0 MISPLACED (1 must be at slot 1!), slot 1 WRONG (3 absent everywhere)
        session.submit_guess([[1], [3], [2]])

        k = BoardKnowledge(session)
        self.assertEqual(k.solved_slots[2], (2,))
        # Because slot 2 is EXACT and slot 0 guessed 1 (MISPLACED), slot 1 is the ONLY unmatched slot -> locked to (1,)!
        self.assertEqual(k.solved_slots[1], (1,))
        # Product 7 (prime index 3) got WRONG with 0 MISPLACED -> forbidden in ALL unmatched slots (0 and 1)
        self.assertIn(session.primes[3], k.forbidden_products[0])
        self.assertIn(session.primes[3], k.forbidden_products[1])
        self.assertTrue(is_board_consistent(session.target_products, session.history))

    def test_exact_solver_wins_across_configs(self) -> None:
        configs = [
            GameConfig(num_primes=6, max_multiplier=1, code_length=5, allow_underfill=False),
            GameConfig(num_primes=5, max_multiplier=2, code_length=5, allow_underfill=True),
            GameConfig(num_primes=5, max_multiplier=2, code_length=5, allow_underfill=False),
            GameConfig(num_primes=8, max_multiplier=3, code_length=4, allow_underfill=True),
        ]
        for idx, cfg in enumerate(configs):
            for s in range(5):
                session = GameSession(cfg, seed=100 * idx + s)
                res = solve_session(session, solver=ExactPoolSolver(seed=s, diversity=0.05), max_steps=60)
                self.assertTrue(res.won, f"ExactPoolSolver failed on {cfg} seed={100*idx+s}")
                # Verify all guesses are canonical sorted lists
                for turn in session.history:
                    for slot in turn.guess_slots:
                        self.assertEqual(slot, sorted(slot))

    def test_factor_solver_wins_extreme_p32_k8_l32_and_no_underfill(self) -> None:
        configs = [
            GameConfig(num_primes=32, max_multiplier=6, code_length=6, allow_underfill=False),
            GameConfig(num_primes=32, max_multiplier=8, code_length=8, allow_underfill=True),
            GameConfig(num_primes=32, max_multiplier=8, code_length=32, allow_underfill=True),
        ]
        tok = PrimeWordleTokenizer()
        for idx, cfg in enumerate(configs):
            session = GameSession(cfg, seed=500 + idx)
            res = solve_session(session, solver=FactorProbeSolver(seed=idx), max_steps=200)
            self.assertTrue(res.won, f"FactorProbeSolver failed on {cfg}")
            ids = tok.encode_session(session)
            self.assertLessEqual(len(ids), 32_768)

    def test_hybrid_dataset_generator_smoke_and_pt_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_jsonl = Path(tmp) / "smoke.jsonl"
            out_pt = Path(tmp) / "smoke.pt"
            summary = generate_dataset(
                num_games=40,
                out_jsonl=out_jsonl,
                out_pt=out_pt,
                seed_base=10_000_000,
                diversity=0.08,
            )
            self.assertEqual(summary["accepted_games"], 40)
            self.assertEqual(summary["skipped_unwon"], 0)
            self.assertTrue(out_jsonl.exists())
            self.assertTrue(out_pt.exists())

            # Verify JSONL roundtrip into GameSession
            lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 40)
            first = GameSession.from_dict(json.loads(lines[0]))
            self.assertEqual(first.status, "won")

            # Verify uint8 .pt shard integrity
            import torch

            shard = torch.load(out_pt, weights_only=True)
            self.assertEqual(shard["vocab_size"], 80)
            self.assertEqual(shard["num_games"], 40)
            self.assertEqual(shard["input_ids"].dtype, torch.uint8)
            self.assertEqual(shard["loss_mask"].dtype, torch.uint8)
            self.assertEqual(len(shard["offsets"]), 41)
            self.assertEqual(int(shard["offsets"][-1]), len(shard["input_ids"]))


if __name__ == "__main__":
    unittest.main()
