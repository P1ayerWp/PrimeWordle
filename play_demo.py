"""Simple CLI inference demo for PrimeWordle using the exported bf16 weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from engine.game import GameConfig, GameSession
from engine.primes import format_slot_powers
from engine.tokenizer import PrimeWordleTokenizer
from model import PrimeGPT, PrimeGPTConfig
from train.stress_test import play_with_custom_constraints


def load_inference_model(weights_path: Path, device: torch.device) -> PrimeGPT:
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    cfg = PrimeGPTConfig.from_dict(ckpt["model_config"])
    model = PrimeGPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch PrimeGPT solve a PrimeWordle game.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/primegpt_25m_grpo_v2_bf16.pt"),
        help="Path to model weights (.pt).",
    )
    parser.add_argument("--p", type=int, default=5, help="Number of base primes P (2..32).")
    parser.add_argument("--k", type=int, default=2, help="Max symbols multiplied per slot K (1..8).")
    parser.add_argument("--l", type=int, default=5, help="Number of slots L (1..32).")
    parser.add_argument("--no-underfill", action="store_true", help="Require exactly K symbols per slot.")
    parser.add_argument("--no-dup", action="store_true", help="Disallow duplicate target slots.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the secret code.")
    parser.add_argument("--max-turns", type=int, default=25)
    parser.add_argument("--anti-rep", action="store_true", help="Enable outer anti-repeat guard.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    tok = PrimeWordleTokenizer(max_symbols=32)
    model = load_inference_model(args.weights, device)

    cfg = GameConfig(
        num_primes=args.p,
        max_multiplier=args.k,
        code_length=args.l,
        allow_underfill=not args.no_underfill,
        allow_duplicate_slots=not args.no_dup,
    )
    session = GameSession(cfg, seed=args.seed)

    print(f"Device: {device} | Weights: {args.weights} ({model.num_parameters()/1e6:.2f}M params)")
    print(f"Config: P={cfg.num_primes}, K={cfg.max_multiplier}, L={cfg.code_length}, "
          f"underfill={cfg.allow_underfill}, dup={cfg.allow_duplicate_slots}, seed={args.seed}")
    target_str = " | ".join(format_slot_powers(s, session.primes) for s in session.target_slots)
    print(f"Secret: [{target_str}]\n")

    res = play_with_custom_constraints(
        model=model,
        session=session,
        device=device,
        tokenizer=tok,
        max_new_turns=args.max_turns,
        anti_repeat_guard=args.anti_rep,
    )

    for rec in session.history:
        slots_fmt = " | ".join(format_slot_powers(s, session.primes) for s in rec.guess_slots)
        fb_fmt = " ".join(f"{fb.value:<9}" for fb in rec.feedback)
        print(f"Turn {rec.turn:2d}: [{slots_fmt}]")
        print(f"      -> {fb_fmt}")

    print(
        f"\nResult: {'WON' if res['won'] else 'TIMEOUT'} in {res['total_turns']} turns "
        f"(wrong_mine={res['wrong_mine']}, exact_drop={res['exact_drop']}, repeat_mine={res['repeat_mine']})"
    )


if __name__ == "__main__":
    main()
