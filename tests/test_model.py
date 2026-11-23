"""Unit & RTX 4060 GPU smoke tests for DeepLogic PrimeGPT."""

from __future__ import annotations

import unittest

import torch

from model import PRESETS, PrimeGPT, PrimeGPTConfig, compute_masked_lm_loss


class TestPrimeGPT(unittest.TestCase):
    def test_deeplogic_preset_param_count_and_config_roundtrip(self) -> None:
        cfg = PRESETS["deeplogic_25m"]
        self.assertEqual(cfg.vocab_size, 80)
        self.assertEqual(cfg.max_seq_len, 32_768)
        self.assertEqual(cfg.n_layers, 16)
        self.assertEqual(PrimeGPTConfig.from_dict(cfg.to_dict()), cfg)

        model = PrimeGPT(cfg)
        n_params = model.num_parameters()
        # 16 layers * ~1.57M/layer + tiny 80*384 embedding ~= 25.2M
        self.assertGreater(n_params, 24_500_000)
        self.assertLess(n_params, 26_000_000)

    def test_kv_cache_matches_full_forward_and_attention_hook(self) -> None:
        torch.manual_seed(0)
        cfg = PrimeGPTConfig(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128)
        model = PrimeGPT(cfg).eval()

        ids = torch.randint(0, cfg.vocab_size, (1, 12))
        with torch.no_grad():
            full_logits, _, full_attns = model(ids, return_last_attn=True)
            # Prefill first 8 tokens, then step 1 token at a time for remaining 4
            _, kv, _ = model(ids[:, :8], use_cache=True)
            step_logits = []
            last_attns = None
            for t in range(8, 12):
                out, kv, last_attns = model(
                    ids[:, t : t + 1],
                    past_kv=kv,
                    use_cache=True,
                    return_last_attn=(t == 11),
                )
                step_logits.append(out)
            cat_logits = torch.cat(step_logits, dim=1)

        max_diff = (full_logits[:, 8:, :] - cat_logits).abs().max().item()
        self.assertLess(max_diff, 1e-4)

        assert full_attns is not None and last_attns is not None
        self.assertEqual(len(full_attns), 2)
        self.assertEqual(tuple(full_attns[-1].shape), (1, 4, 12))
        self.assertAlmostEqual(full_attns[-1][0, 0].sum().item(), 1.0, places=5)
        self.assertLess((full_attns[-1] - last_attns[-1]).abs().max().item(), 1e-5)

    def test_masked_lm_loss_ignores_zero_mask(self) -> None:
        torch.manual_seed(1)
        cfg = PrimeGPTConfig(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128)
        model = PrimeGPT(cfg)
        ids = torch.randint(1, cfg.vocab_size, (2, 16))
        mask = torch.zeros_like(ids)
        mask[:, 7:] = 1
        logits, _, _ = model(ids)
        loss = compute_masked_lm_loss(logits, ids, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.tok_emb.weight.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
    def test_cuda_bf16_deeplogic_checkpointing_and_kv_inference(self) -> None:
        cfg = PRESETS["deeplogic_25m"]
        model = PrimeGPT(cfg).to(device="cuda", dtype=torch.bfloat16)
        model.gradient_checkpointing = True

        # Training forward + backward at 4096 tokens in bf16
        ids = torch.randint(1, cfg.vocab_size, (1, 4096), device="cuda")
        mask = torch.ones_like(ids)
        mask[:, :7] = 0

        logits, _, _ = model(ids)
        loss = compute_masked_lm_loss(logits, ids, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

        # Eval KV-cache single-token decode
        model.eval()
        with torch.no_grad():
            _, kv, _ = model(ids[:, :128], use_cache=True)
            next_logits, kv, attns = model(
                ids[:, 128:129], past_kv=kv, use_cache=True, return_last_attn=True
            )
            self.assertEqual(tuple(next_logits.shape), (1, 1, 80))
            assert attns is not None
            self.assertEqual(tuple(attns[-1].shape), (1, 6, 129))


if __name__ == "__main__":
    unittest.main()
