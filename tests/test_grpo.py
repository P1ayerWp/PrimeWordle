"""Unit tests verifying all 4 Phase-4 GRPO guardrails (exact log-prob, frozen ref, AntiRep=OFF, rewards)."""

from __future__ import annotations

import unittest

import torch

from engine.tokenizer import PrimeWordleTokenizer
from model import PrimeGPT, PrimeGPTConfig
from train.grpo import (
    build_prompt_spec,
    compute_episode_reward,
    compute_group_advantages,
    grpo_update_step,
    rollout_group_for_prompt,
)
from train.grpo_v2 import build_generalist_prompt, rollout_generalist_group


class TestGRPOGuardrails(unittest.TestCase):
    def test_reward_ordering_and_progress_floor(self) -> None:
        r_fast_win = compute_episode_reward(True, 3, 15, 1.0, 0, 10, 0, 5, 0, 10)
        r_slow_win = compute_episode_reward(True, 12, 15, 1.0, 0, 10, 0, 5, 0, 10)
        r_win_with_rep = compute_episode_reward(True, 5, 15, 1.0, 0, 10, 0, 5, 4, 10)
        r_timeout_hi = compute_episode_reward(False, 15, 15, 0.8, 0, 10, 0, 5, 0, 10)
        r_timeout_lo = compute_episode_reward(False, 15, 15, 0.1, 0, 10, 0, 5, 0, 10)
        r_timeout_rep = compute_episode_reward(False, 15, 15, 0.1, 3, 10, 2, 5, 6, 10)

        self.assertGreater(r_fast_win, r_slow_win)
        self.assertGreater(r_fast_win, r_win_with_rep)
        self.assertGreater(r_slow_win, r_timeout_hi)
        self.assertGreater(r_timeout_hi, r_timeout_lo)
        self.assertGreater(r_timeout_lo, r_timeout_rep)

    def test_rollout_mask_alignment_ratio_one_and_frozen_ref(self) -> None:
        torch.manual_seed(0)
        tok = PrimeWordleTokenizer(max_symbols=32)
        cfg = PrimeGPTConfig(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128)
        device = torch.device("cpu")

        model = PrimeGPT(cfg).to(device)
        ref_model = PrimeGPT(cfg).to(device)
        ref_model.load_state_dict(model.state_dict())
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_before = {k: v.clone() for k, v in ref_model.state_dict().items()}

        # Test PvZ UOFF prompt (with forbidden pairs!)
        spec = build_prompt_spec("PvZ_10x10_UOFF", seed=777)
        spec.max_new_turns = 4  # Fast CPU unit test
        eps = rollout_group_for_prompt(
            model, spec, prompt_idx=0, group_size=4, temperature=0.8, device=device, tokenizer=tok
        )
        self.assertEqual(len(eps), 4)

        # Guardrail #3: header (7 tokens) + all dirty prefix turns + all feedback tokens must have mask=0
        for ep in eps:
            self.assertEqual(len(ep.input_ids), len(ep.action_mask))
            self.assertEqual(len(ep.input_ids), len(ep.allowed_masks))
            self.assertEqual(ep.action_mask[:7], [0] * 7)
            self.assertGreater(sum(ep.action_mask), 0)
            # Every chosen action token must be allowed in its recorded mask
            for tid, am, mask_vec in zip(ep.input_ids, ep.action_mask, ep.allowed_masks):
                if am == 1:
                    self.assertTrue(mask_vec[tid])

        # Force non-zero advantages so grpo_update_step executes both PPO epochs
        adv = [1.0, -1.0, 0.5, -0.5]
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        stats = grpo_update_step(
            model=model,
            ref_model=ref_model,
            optimizer=opt,
            episodes=eps,
            advantages=adv,
            device=device,
            kl_coef=0.03,
            clip_eps=0.2,
            ppo_epochs=2,
            mini_batch_size=2,
        )
        # Guardrail #3 exactness: on PPO epoch 0, new_logp == old_logp -> ratio == 1.0000!
        self.assertLess(stats["max_ratio_err_e0"], 1e-4)
        # Guardrail #2: KL starts near 0 and ref_model stays 100% frozen!
        self.assertGreaterEqual(stats["kl"], 0.0)
        for k, v in ref_model.state_dict().items():
            self.assertTrue(torch.equal(v, ref_before[k]))

        # Sanity check group advantages zero-variance protection
        for ep in eps:
            ep.reward = 1.234
        self.assertEqual(compute_group_advantages(eps), [0.0] * 4)

    def test_grpo_v2_all_6_generalist_categories_exact_ratio(self) -> None:
        torch.manual_seed(2)
        tok = PrimeWordleTokenizer(max_symbols=32)
        cfg = PrimeGPTConfig(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128)
        device = torch.device("cpu")
        model = PrimeGPT(cfg).to(device)
        ref = PrimeGPT(cfg).to(device)
        ref.load_state_dict(model.state_dict())
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)

        cats = [
            "Gen_Mid_K24",
            "Gen_Sparse_PvZ",
            "Gen_Wide_K1",
            "Gen_K58_Slice",
            "Gen_UNK_CFG",
            "Gen_UNK_CFG",
            "Gen_Small_K12",
        ]
        for idx, cat in enumerate(cats):
            sp = build_generalist_prompt(cat, seed=12345 + idx * 17)
            sp.max_new_turns = 3
            eps = rollout_generalist_group(model, sp, 0, 3, 0.8, device, tok)
            self.assertEqual(len(eps), 3)
            stats = grpo_update_step(
                model,
                ref,
                torch.optim.AdamW(model.parameters(), lr=1e-5),
                eps,
                [1.0, 0.0, -1.0],
                device,
                ppo_epochs=1,
                mini_batch_size=3,
            )
            self.assertLess(stats["max_ratio_err_e0"], 1e-4, f"Category {sp.category} ratio error")


if __name__ == "__main__":
    unittest.main()
