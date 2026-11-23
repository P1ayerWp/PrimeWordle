"""Comprehensive zero-shot stress-test & ceiling probe for a trained PrimeGPT checkpoint.

Tests 4 real-world scenarios on RTX 4060:
  1. Fair-turn ceiling across K1..K8 (including K6_P25_L8 and K8_P32_L5) + truly raw
     unconstrained decoding (no syntax mask at all!)
  2. Zero-shot PvZ Plant Decoding (P=10, K=2, L=10, 50% FORBIDDEN pairs including p*p):
       a) UNDERFILL_ON (1..2 plants per slot)
       b) UNDERFILL_OFF (strictly 2 plants per slot — true PvZ 10x10 rule!)
     Tested BOTH raw-logits and with outer anti-repeat guard!
  3. Zero-shot Mid-Game Takeover ("接着我的残局玩"):
       a) User's actual Phase-1 human game from `data/human_games.jsonl`
       b) Messy synthetic human prefixes (5 noisy turns, disjoint RNG seed!) on
          K2_P5_L5_U, PvZ 10x10 UNDERFILL_ON, and PvZ 10x10 UNDERFILL_OFF
     Tested BOTH raw-logits and with outer anti-repeat guard!
  4. Zero-shot 26-letter English Wordle (P=26, K=1, L=5) via outer Dictionary Trie Mask
     (pure neural logits + Trie mask, zero external candidate elimination!)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from engine.game import GameConfig, GameSession
from engine.primes import encode_slot
from engine.tokenizer import PrimeWordleTokenizer
from model import PrimeGPT, PrimeGPTConfig
from solver.constraints import BoardKnowledge, symbols_to_mask
from train.eval_play import (
    _parse_guess_tokens,
    play_games_batched,
)

# 344 real common 5-letter English words for zero-shot Wordle Trie stress testing
ENGLISH_WORDLE_WORDS: tuple[str, ...] = (
    "ABOUT", "ABOVE", "ADULT", "AFTER", "AGAIN", "AGENT", "AGREE", "AHEAD", "ALARM", "ALBUM",
    "ALERT", "ALIKE", "ALIVE", "ALLOW", "ALONE", "ALONG", "ALTER", "AMONG", "ANGER", "ANGLE",
    "ANGRY", "APPLE", "APPLY", "ARENA", "ARGUE", "ARISE", "ARRAY", "ASIDE", "ASSET", "AUDIO",
    "AVOID", "AWARD", "AWARE", "BADLY", "BAKER", "BASIC", "BEACH", "BEGAN", "BEGIN", "BEING",
    "BELOW", "BENCH", "BIRTH", "BLACK", "BLAME", "BLIND", "BLOCK", "BLOOD", "BOARD", "BOOST",
    "BRAIN", "BRAND", "BREAD", "BREAK", "BRICK", "BRIEF", "BRING", "BROAD", "BROWN", "BUILD",
    "BUYER", "CABLE", "CARRY", "CATCH", "CAUSE", "CHAIN", "CHAIR", "CHART", "CHASE", "CHEAP",
    "CHECK", "CHEST", "CHIEF", "CHILD", "CHINA", "CLAIM", "CLASS", "CLEAN", "CLEAR", "CLICK",
    "CLOCK", "CLOSE", "CLOUD", "COACH", "COAST", "COULD", "COUNT", "COURT", "COVER", "CRAFT",
    "CRANE", "CRASH", "CREAM", "CRIME", "CROSS", "CROWD", "CROWN", "CURVE", "CYCLE", "DAILY",
    "DANCE", "DATED", "DEALT", "DEATH", "DEBUT", "DELAY", "DEPTH", "DOING", "DOUBT", "DOZEN",
    "DRAFT", "DRAMA", "DRAWN", "DREAM", "DRESS", "DRINK", "DRIVE", "EARTH", "LIGHT", "MAGIC",
    "MATCH", "METAL", "MODEL", "MONEY", "MONTH", "MOTOR", "MOUNT", "MOUSE", "MOUTH", "MOVIE",
    "MUSIC", "NIGHT", "NOISE", "NORTH", "NOVEL", "NURSE", "OCEAN", "OFFER", "ORDER", "OTHER",
    "PAINT", "PANEL", "PAPER", "PARTY", "PEACE", "PHASE", "PHONE", "PHOTO", "PIECE", "PILOT",
    "PITCH", "PLACE", "PLAIN", "PLANE", "PLANT", "PLATE", "POINT", "POUND", "POWER", "PRESS",
    "PRICE", "PRIDE", "PRIME", "PRINT", "PRIOR", "PRIZE", "PROOF", "PROUD", "PROVE", "QUEEN",
    "QUICK", "QUIET", "QUITE", "RADIO", "RAISE", "RANGE", "RAPID", "RATIO", "REACH", "REACT",
    "READY", "REFER", "RIGHT", "RIVAL", "RIVER", "ROBOT", "ROUND", "ROUTE", "ROYAL", "RURAL",
    "SCALE", "SCENE", "SCOPE", "SCORE", "SENSE", "SERVE", "SEVEN", "SHADE", "SHAKE", "SHALL",
    "SHAPE", "SHARE", "SHARP", "SHEET", "SHELF", "SHELL", "SHIFT", "SHINE", "SHIRT", "SHOCK",
    "SHOOT", "SHORT", "SHOWN", "SIGHT", "SINCE", "SKILL", "SLATE", "SLEEP", "SLIDE", "SMALL",
    "SMART", "SMILE", "SMITH", "SMOKE", "SOLID", "SOLVE", "SORRY", "SOUND", "SOUTH", "SPACE",
    "SPARE", "SPEAK", "SPEED", "SPEND", "SPENT", "SPLIT", "SPOKE", "SPORT", "STAFF", "STAGE",
    "STAKE", "STAND", "START", "STATE", "STEAM", "STEEL", "STICK", "STILL", "STOCK", "STONE",
    "STOOD", "STORE", "STORM", "STORY", "STRIP", "STUCK", "STUDY", "STUFF", "STYLE", "SUGAR",
    "SUITE", "SUPER", "SWEET", "TABLE", "TAKEN", "TASTE", "TEACH", "TEETH", "THANK", "THEIR",
    "THEME", "THERE", "THESE", "THICK", "THING", "THINK", "THIRD", "THOSE", "THREE", "THREW",
    "THROW", "TIGHT", "TIMES", "TIRED", "TITLE", "TODAY", "TOPIC", "TOTAL", "TOUCH", "TOUGH",
    "TOWER", "TRACK", "TRADE", "TRAIN", "TREAT", "TREND", "TRIAL", "TRIED", "TRIES", "TRUCK",
    "TRULY", "TRUST", "TRUTH", "TWICE", "UNDER", "UNION", "UNITY", "UNTIL", "UPPER", "UPSET",
    "URBAN", "USAGE", "USUAL", "VALID", "VALUE", "VIDEO", "VIRUS", "VISIT", "VITAL", "VOICE",
    "WASTE", "WATCH", "WATER", "WHEEL", "WHERE", "WHICH", "WHILE", "WHITE", "WHOLE", "WHOSE",
    "WOMAN", "WORLD", "WORRY", "WORSE", "WORST", "WORTH", "WOULD", "WOUND", "WRITE", "WRONG",
    "WROTE", "YIELD", "YOUNG", "YOUTH",
)


def load_model(ckpt_path: Path, device: torch.device) -> PrimeGPT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = PrimeGPTConfig.from_dict(ckpt["model_config"])
    model = PrimeGPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _rate(num: int, den: int) -> float | None:
    return None if den <= 0 else round(num / den, 4)


def make_forbidden_fusion_table(
    num_primes: int,
    forbid_ratio: float,
    allow_underfill: bool,
    seed: int,
    forbid_self_ratio: float = 0.5,
) -> tuple[dict[int, set[int]], list[list[int]]]:
    """Generate a symmetric K=2 pair-fusion graph with forbidden cross & self (p*p) fusions."""
    rng = random.Random(seed)
    allowed_pairs: dict[int, set[int]] = {i: set() for i in range(num_primes)}
    legal_slots: list[list[int]] = [[i] for i in range(num_primes)] if allow_underfill else []

    cross_pairs = [(a, b) for a in range(num_primes) for b in range(a + 1, num_primes)]
    self_pairs = [(a, a) for a in range(num_primes)]
    rng.shuffle(cross_pairs)
    rng.shuffle(self_pairs)

    keep_cross = max(num_primes, int(round(len(cross_pairs) * (1.0 - forbid_ratio))))
    keep_self = int(round(len(self_pairs) * (1.0 - forbid_self_ratio)))

    chosen_pairs = cross_pairs[:keep_cross] + self_pairs[:keep_self]
    for a, b in chosen_pairs:
        allowed_pairs[a].add(b)
        legal_slots.append([a, b])

    # Ensure every prime has at least one legal pair when underfill is OFF so no dead start symbol
    if not allow_underfill:
        for a in range(num_primes):
            if not allowed_pairs[a]:
                b = min(num_primes - 1, a + 1) if a < num_primes - 1 else a
                allowed_pairs[a].add(b)
                if [a, b] not in legal_slots:
                    legal_slots.append([a, b])

    return allowed_pairs, legal_slots


def _slot_legal_extensions(
    cur_slot_syms: list[int],
    config: GameConfig,
    primes: list[int],
    allowed_pairs: dict[int, set[int]] | None,
    forbidden_slot_prods: set[int] | None,
) -> tuple[list[int], bool]:
    """Compute legal next symbol indices AND whether closing the slot now is legal.

    Supports BOTH pair-fusion masks (`allowed_pairs`) AND outer anti-repeat guard
    (`forbidden_slot_prods`: prevents re-submitting an already-tried non-EXACT product
    at this exact slot!).
    """
    p = config.num_primes
    k_max = config.max_multiplier
    cur_len = len(cur_slot_syms)
    forb = forbidden_slot_prods or set()

    def _can_complete_from(prefix: list[int]) -> bool:
        """Return True if `prefix` can be completed into at least one non-forbidden legal slot."""
        plen = len(prefix)
        if plen > k_max:
            return False
        if (config.allow_underfill and plen >= 1) or (not config.allow_underfill and plen == k_max):
            if encode_slot(prefix, primes) not in forb:
                return True
        if plen == k_max:
            return False
        if k_max == 2 and plen == 1:
            a = prefix[0]
            bs = (
                sorted(b for b in allowed_pairs.get(a, set()) if b >= a)
                if allowed_pairs is not None
                else range(a, p)
            )
            for b in bs:
                if encode_slot((a, b), primes) not in forb:
                    return True
            return False
        # For K >= 3, forbidding tried products rarely blocks all extensions
        return True

    next_syms: list[int] = []
    if cur_len < k_max:
        if cur_len == 0:
            cands = list(range(p))
            if allowed_pairs is not None and not config.allow_underfill:
                cands = [a for a in cands if allowed_pairs.get(a)]
            for a in cands:
                if _can_complete_from([a]):
                    next_syms.append(a)
        else:
            last_s = cur_slot_syms[-1]
            if allowed_pairs is not None and k_max == 2:
                cands = sorted(b for b in allowed_pairs.get(last_s, set()) if b >= last_s)
            else:
                cands = list(range(last_s, p))
            for b in cands:
                if _can_complete_from([*cur_slot_syms, b]):
                    next_syms.append(b)

    can_close = False
    if (config.allow_underfill and cur_len >= 1) or (not config.allow_underfill and cur_len == k_max):
        if encode_slot(cur_slot_syms, primes) not in forb:
            can_close = True

    return next_syms, can_close


@torch.no_grad()
def play_with_custom_constraints(
    model: PrimeGPT,
    session: GameSession,
    device: torch.device,
    tokenizer: PrimeWordleTokenizer,
    max_new_turns: int = 25,
    allowed_pairs: dict[int, set[int]] | None = None,
    canonicalize_history: bool = True,
    anti_repeat_guard: bool = False,
) -> dict[str, Any]:
    """Continue any (fresh or mid-game!) GameSession with optional pair-fusion & anti-repeat masks."""
    config = session.config
    turn_id = tokenizer.stoi["<TURN>"]
    slot_id = tokenizer.stoi["<SLOT>"]
    fb_id = tokenizer.stoi["<FB>"]
    sym_ids = [tokenizer.stoi[f"S_{i}"] for i in range(config.num_primes)]
    sym_id_set = set(sym_ids)

    prompt_ids = tokenizer.encode_config(config)
    for past_turn in session.history:
        prompt_ids.append(turn_id)
        for slot in past_turn.guess_slots:
            prompt_ids.append(slot_id)
            ordered = sorted(slot) if canonicalize_history else list(slot)
            for s in ordered:
                prompt_ids.append(sym_ids[s])
        prompt_ids.append(fb_id)
        for fb in past_turn.feedback:
            prompt_ids.append(tokenizer.stoi[fb.value])

    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.amp.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
    ):
        logits, kv, _ = model(prompt, use_cache=True)

        wrong_h = wrong_o = exact_h = exact_o = rep_h = rep_o = 0
        start_turns = len(session.history)
        max_gen = 2 + config.code_length * (config.max_multiplier + 1)

        for _ in range(max_new_turns):
            if session.status != "playing":
                break

            knowledge = BoardKnowledge(session) if session.history else None

            gen_ids: list[int] = []
            slots_started = 0
            cur_slot_syms: list[int] = []

            for step in range(max_gen):
                row_logits = logits[0, -1, :].float()
                allowed: list[int] = []
                if step == 0:
                    allowed = [turn_id]
                elif slots_started == 0:
                    allowed = [slot_id]
                else:
                    slot_idx = slots_started - 1
                    forb_prods: set[int] | None = None
                    if anti_repeat_guard and knowledge is not None:
                        if slot_idx not in knowledge.confirmed_exact:
                            forb_prods = knowledge.tried_at_slot[slot_idx]

                    next_syms, can_close = _slot_legal_extensions(
                        cur_slot_syms, config, session.primes, allowed_pairs, forb_prods
                    )
                    # Fallback if all legal combinations at this slot were somehow exhausted
                    if not next_syms and not can_close:
                        next_syms, can_close = _slot_legal_extensions(
                            cur_slot_syms, config, session.primes, allowed_pairs, None
                        )

                    allowed.extend(sym_ids[s] for s in next_syms)
                    if can_close:
                        allowed.append(slot_id if slots_started < config.code_length else fb_id)

                best_id = max(allowed, key=lambda idx: float(row_logits[idx].item()))
                gen_ids.append(best_id)
                if best_id == slot_id:
                    slots_started += 1
                    cur_slot_syms = []
                elif best_id in sym_id_set:
                    cur_slot_syms.append(best_id - sym_ids[0])

                inp = torch.tensor([[best_id]], dtype=torch.long, device=device)
                logits, kv, _ = model(inp, past_kv=kv, use_cache=True)
                if best_id == fb_id:
                    break

            parsed = _parse_guess_tokens(gen_ids, tokenizer, config)
            if parsed is None:
                break

            if knowledge is not None:
                for i, slot_syms in enumerate(parsed):
                    prod = encode_slot(slot_syms, session.primes)
                    mask = symbols_to_mask(slot_syms)
                    if i in knowledge.confirmed_exact:
                        exact_o += 1
                        if prod != knowledge.solved_products[i]:
                            exact_h += 1
                    else:
                        if knowledge.excluded_mask[i] != 0:
                            wrong_o += 1
                            if (mask & knowledge.excluded_mask[i]) != 0:
                                wrong_h += 1
                        if knowledge.tried_at_slot[i]:
                            rep_o += 1
                            if prod in knowledge.tried_at_slot[i]:
                                rep_h += 1

            rec = session.submit_guess(parsed)
            fb_ids = [tokenizer.stoi[fb.value] for fb in rec.feedback]
            fb_t = torch.tensor([fb_ids], dtype=torch.long, device=device)
            logits, kv, _ = model(fb_t, past_kv=kv, use_cache=True)

    return {
        "won": session.status == "won",
        "start_turns": start_turns,
        "added_turns": len(session.history) - start_turns,
        "total_turns": len(session.history),
        "wrong_mine": _rate(wrong_h, wrong_o),
        "exact_drop": _rate(exact_h, exact_o),
        "repeat_mine": _rate(rep_h, rep_o),
    }


def _build_word_trie(words: tuple[str, ...]) -> dict[Any, Any]:
    root: dict[Any, Any] = {}
    for w in words:
        node = root
        for ch in w:
            idx = ord(ch) - ord("A")
            node = node.setdefault(idx, {})
    return root


@torch.no_grad()
def play_english_wordle_trie(
    model: PrimeGPT,
    secret_word: str,
    trie: dict[Any, Any],
    device: torch.device,
    tokenizer: PrimeWordleTokenizer,
    max_turns: int = 10,
    beam_width: int = 8,
    filter_tried_words: bool = True,
) -> dict[str, Any]:
    """Zero-shot 26-letter English Wordle using ONLY neural logits + outer Dictionary Trie."""
    cfg = GameConfig(num_primes=26, max_multiplier=1, code_length=5, allow_underfill=False)
    target_slots = [[ord(ch) - ord("A")] for ch in secret_word]
    session = GameSession(cfg, seed=1, target_slots=target_slots)

    turn_id = tokenizer.stoi["<TURN>"]
    slot_id = tokenizer.stoi["<SLOT>"]
    fb_id = tokenizer.stoi["<FB>"]
    sym_ids = [tokenizer.stoi[f"S_{i}"] for i in range(26)]

    prompt = torch.tensor([tokenizer.encode_config(cfg)], dtype=torch.long, device=device)
    tried_words: set[str] = set()
    guesses: list[str] = []

    with torch.amp.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
    ):
        logits, kv, _ = model(prompt, use_cache=True)

        for _ in range(max_turns):
            if session.status != "playing":
                break

            t_in = torch.tensor([[turn_id]], dtype=torch.long, device=device)
            logits, kv, _ = model(t_in, past_kv=kv, use_cache=True)

            beams: list[tuple[float, list[int], dict[Any, Any], Any, torch.Tensor]] = [
                (0.0, [], trie, kv, logits)
            ]

            for pos in range(5):
                next_beams: list[tuple[float, list[int], dict[Any, Any], Any, torch.Tensor]] = []
                for score, letters, node, b_kv, _b_logits in beams:
                    s_in = torch.tensor([[slot_id]], dtype=torch.long, device=device)
                    s_logits, s_kv, _ = model(s_in, past_kv=b_kv, use_cache=True)
                    logp = torch.log_softmax(s_logits[0, -1, :].float(), dim=-1)

                    legal_letters = list(node.keys())
                    scored_next = sorted(
                        [(score + float(logp[sym_ids[c]].item()), c) for c in legal_letters],
                        key=lambda x: -x[0],
                    )[:beam_width]

                    for new_score, c in scored_next:
                        if pos == 4 and filter_tried_words:
                            cand_word = "".join(chr(ord("A") + x) for x in (*letters, c))
                            if cand_word in tried_words:
                                continue
                        c_in = torch.tensor([[sym_ids[c]]], dtype=torch.long, device=device)
                        c_logits, c_kv, _ = model(c_in, past_kv=s_kv, use_cache=True)
                        next_beams.append((new_score, [*letters, c], node[c], c_kv, c_logits))

                next_beams.sort(key=lambda x: -x[0])
                beams = next_beams[:beam_width]

            _, best_letters, _, kv, logits = beams[0]
            word = "".join(chr(ord("A") + x) for x in best_letters)
            tried_words.add(word)
            guesses.append(word)

            rec = session.submit_guess([[x] for x in best_letters])
            tail = [fb_id] + [tokenizer.stoi[fb.value] for fb in rec.feedback]
            tail_t = torch.tensor([tail], dtype=torch.long, device=device)
            logits, kv, _ = model(tail_t, past_kv=kv, use_cache=True)

    return {
        "secret": secret_word,
        "won": session.status == "won",
        "turns": len(session.history),
        "guesses": guesses,
    }


def _agg_rates(results: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(int(r["won"]) for r in results)
    n = len(results)
    avg_add = sum(r["added_turns"] for r in results) / max(1, n)
    avg_tot = sum(r["total_turns"] for r in results) / max(1, n)
    wms = [r["wrong_mine"] for r in results if r["wrong_mine"] is not None]
    eds = [r["exact_drop"] for r in results if r["exact_drop"] is not None]
    rms = [r["repeat_mine"] for r in results if r["repeat_mine"] is not None]
    return {
        "wins": wins,
        "games": n,
        "win_rate": round(wins / max(1, n), 3),
        "avg_added_turns": round(avg_add, 2),
        "avg_total_turns": round(avg_tot, 2),
        "wrong_mine": round(sum(wms) / len(wms), 4) if wms else None,
        "exact_drop": round(sum(eds) / len(eds), 4) if eds else None,
        "repeat_mine": round(sum(rms) / len(rms), 4) if rms else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot stress test for DeepLogic PrimeGPT.")
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/deeplogic_400m/best.pt"))
    parser.add_argument("--focus-adversarial", action="store_true", help="Run ONLY sections [2] & [3] (PvZ & Takeover) fast.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = PrimeWordleTokenizer(max_symbols=32)
    model = load_model(args.ckpt, device)
    print(f"Loaded {args.ckpt} ({model.num_parameters()/1e6:.2f}M params) on {device}\n", flush=True)

    report: dict[str, Any] = {"ckpt": str(args.ckpt)}

    # =========================================================================
    # 1. Fair-Turn Ceiling + Truly Raw Unconstrained Check
    # =========================================================================
    if not args.focus_adversarial:
        print("=== [1] Fair-Turn Ceiling (Batched KV-Cache) ===", flush=True)
        ceiling_presets = [
            ("K1_P6_L5", GameConfig(6, 1, 5, allow_underfill=False), 10_000, 12, 16, True),
            ("K1_P6_L5 [RAW]", GameConfig(6, 1, 5, allow_underfill=False), 10_000, 12, 16, False),
            ("K2_P5_L5_U", GameConfig(5, 2, 5, allow_underfill=True), 30_000, 15, 16, True),
            ("K2_P5_L5_U [RAW]", GameConfig(5, 2, 5, allow_underfill=True), 30_000, 15, 16, False),
            ("K3_P8_L5_U", GameConfig(8, 3, 5, allow_underfill=True), 50_000, 25, 12, True),
            ("K4_P16_L6_U", GameConfig(16, 4, 6, allow_underfill=True), 60_000, 28, 12, True),
            ("K4_P16_L6_U [RAW]", GameConfig(16, 4, 6, allow_underfill=True), 60_000, 28, 12, False),
            ("K6_P25_L8_U (38t)", GameConfig(25, 6, 8, allow_underfill=True), 70_000, 38, 8, True),
            ("K8_P32_L5_U (55t)", GameConfig(32, 8, 5, allow_underfill=True), 80_000, 55, 8, True),
        ]
        print(f"{'Preset':<20} | {'Win%':>6} | {'AvgT':>5} | {'SynErr':>6} | {'Wrong':>6} | {'ExDrop':>6} | {'Repeat':>6}")
        print("-" * 74)
        ceiling_out: dict[str, Any] = {}
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            for name, cfg, bseed, max_t, n_g, constrained in ceiling_presets:
                outs = play_games_batched(
                    model, cfg, seeds=[bseed + i for i in range(n_g)], device=device,
                    tokenizer=tok, max_turns=max_t, constrained_syntax=constrained,
                )
                win_r = sum(int(o.won) for o in outs) / n_g
                avg_t = sum(o.turns for o in outs) / n_g
                syn_crash = sum(int(o.syntax_errors > 0) for o in outs) / n_g
                wm = _rate(sum(o.wrong_mine_hits for o in outs), sum(o.wrong_mine_opps for o in outs))
                ed = _rate(sum(o.exact_drop_hits for o in outs), sum(o.exact_drop_opps for o in outs))
                rm = _rate(sum(o.repeat_mine_hits for o in outs), sum(o.repeat_mine_opps for o in outs))
                ceiling_out[name] = {
                    "win_rate": round(win_r, 3), "avg_turns": round(avg_t, 2),
                    "syn_crash": round(syn_crash, 3), "wrong_mine": wm, "exact_drop": ed, "repeat_mine": rm,
                }
                print(
                    f"{name:<20} | {win_r*100:5.1f}% | {avg_t:5.2f} | {syn_crash*100:5.1f}% | "
                    f"{_fmt(wm):>6} | {_fmt(ed):>6} | {_fmt(rm):>6}",
                    flush=True,
                )
        report["ceiling"] = ceiling_out

    # =========================================================================
    # 2. PvZ Plant Decoding (P=10, K=2, L=10, 50% Forbidden Pairs incl. p*p)
    #    Tested BOTH Raw-Logits AND Outer Anti-Repeat Guard!
    # =========================================================================
    print("\n=== [2] PvZ Plant Decoding (P=10, K=2, L=10, 50% Pairs + 50% Self p*p FORBIDDEN) ===", flush=True)
    pvz_report: dict[str, Any] = {}
    for mode_label, underfill_flag in [("UNDERFILL_ON", True), ("UNDERFILL_OFF (True PvZ 10x10)", False)]:
        pvz_cfg = GameConfig(num_primes=10, max_multiplier=2, code_length=10, allow_underfill=underfill_flag)
        for guard_flag in (False, True):
            guard_tag = "AntiRep=ON " if guard_flag else "AntiRep=OFF"
            results = []
            for i in range(10):
                allowed_pairs, legal_slots = make_forbidden_fusion_table(
                    10, forbid_ratio=0.50, allow_underfill=underfill_flag, seed=7000 + i, forbid_self_ratio=0.50
                )
                rng = random.Random(7100 + i)
                target = [list(rng.choice(legal_slots)) for _ in range(10)]
                sess = GameSession(pvz_cfg, seed=7100 + i, target_slots=target)
                results.append(
                    play_with_custom_constraints(
                        model, sess, device, tok, max_new_turns=25,
                        allowed_pairs=allowed_pairs, anti_repeat_guard=guard_flag,
                    )
                )
            agg = _agg_rates(results)
            key = f"{mode_label} | {guard_tag}"
            pvz_report[key] = agg
            print(
                f"  {key:<46} | Win={agg['wins']:2d}/10 ({agg['win_rate']*100:5.1f}%) | "
                f"AvgT={agg['avg_total_turns']:5.2f} | wrong={_fmt(agg['wrong_mine']):>5} | "
                f"ex_drop={_fmt(agg['exact_drop']):>5} | rep={_fmt(agg['repeat_mine']):>5}",
                flush=True,
            )
    report["pvz_forbidden"] = pvz_report

    # =========================================================================
    # 3. Mid-Game Takeover ("接着我的残局玩") — Disjoint Messy Human Seeds!
    # =========================================================================
    print("\n=== [3] Mid-Game Takeover (Raw vs Outer Anti-Repeat Guard) ===", flush=True)
    takeover_report: dict[str, Any] = {}

    # 3a: User's ACTUAL Phase-1 human game (game_id=6b263975d617, human won in 8 turns)
    human_file = Path("data/human_games.jsonl")
    if human_file.exists():
        for line in human_file.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("game_id") == "6b263975d617":
                for guard_flag in (False, True):
                    guard_tag = "AntiRep=ON " if guard_flag else "AntiRep=OFF"
                    for take_after in (2, 3, 4):
                        cfg_h = GameConfig.from_dict(rec["config"])
                        sess_h = GameSession(cfg_h, seed=rec["seed"], target_slots=rec["target"]["slots"])
                        for h in rec["history"][:take_after]:
                            sess_h.submit_guess(h["guess_slots"])
                        out_h = play_with_custom_constraints(
                            model, sess_h, device, tok, max_new_turns=12,
                            canonicalize_history=True, anti_repeat_guard=guard_flag,
                        )
                        takeover_report[f"real_human_T{take_after}_{guard_tag.strip()}"] = out_h
                        print(
                            f"  Real Human 6b263975d617 (8t) T{take_after} [{guard_tag}]: "
                            f"won={str(out_h['won']):5s} | added={out_h['added_turns']:2d}t (tot {out_h['total_turns']:2d}t) | "
                            f"wrong={_fmt(out_h['wrong_mine']):>5} | ex_drop={_fmt(out_h['exact_drop']):>5} | rep={_fmt(out_h['repeat_mine']):>5}",
                            flush=True,
                        )

    # 3b: Synthetic messy human prefixes (5 noisy turns using DISJOINT messy_rng seed!)
    messy_cases = [
        ("K2_P5_L5_U (5 messy)", GameConfig(5, 2, 5, allow_underfill=True), 0.0),
        ("PvZ 10x10 U_ON 50%-forbid (5 messy)", GameConfig(10, 2, 10, allow_underfill=True), 0.50),
        ("PvZ 10x10 U_OFF 50%-forbid (5 messy)", GameConfig(10, 2, 10, allow_underfill=False), 0.50),
    ]
    for label, cfg_m, forbid_r in messy_cases:
        for guard_flag in (False, True):
            guard_tag = "AntiRep=ON " if guard_flag else "AntiRep=OFF"
            results = []
            for i in range(10):
                target_rng = random.Random(8500 + i)
                messy_rng = random.Random(99_000 + i)  # Strictly disjoint from target seed!
                if forbid_r > 0:
                    allowed_pairs, legal_slots = make_forbidden_fusion_table(
                        10, forbid_r, allow_underfill=cfg_m.allow_underfill, seed=8600 + i, forbid_self_ratio=0.50
                    )
                    target = [list(target_rng.choice(legal_slots)) for _ in range(cfg_m.code_length)]
                else:
                    allowed_pairs = None
                    legal_slots = None
                    target = None
                sess = GameSession(cfg_m, seed=8500 + i, target_slots=target)
                for _t in range(5):
                    if sess.status != "playing":
                        break
                    if legal_slots is not None:
                        g = [list(messy_rng.choice(legal_slots)) for _ in range(cfg_m.code_length)]
                    else:
                        k_lo = 1 if cfg_m.allow_underfill else cfg_m.max_multiplier
                        g = [
                            [messy_rng.randrange(cfg_m.num_primes) for _ in range(messy_rng.randint(k_lo, cfg_m.max_multiplier))]
                            for _ in range(cfg_m.code_length)
                        ]
                    sess.submit_guess(g)
                results.append(
                    play_with_custom_constraints(
                        model, sess, device, tok, max_new_turns=20,
                        allowed_pairs=allowed_pairs, canonicalize_history=True, anti_repeat_guard=guard_flag,
                    )
                )
            agg = _agg_rates(results)
            key = f"{label} | {guard_tag}"
            takeover_report[key] = agg
            print(
                f"  {key:<51} | Win={agg['wins']:2d}/10 ({agg['win_rate']*100:5.1f}%) | "
                f"Add={agg['avg_added_turns']:4.1f}t | wrong={_fmt(agg['wrong_mine']):>5} | "
                f"ex_drop={_fmt(agg['exact_drop']):>5} | rep={_fmt(agg['repeat_mine']):>5}",
                flush=True,
            )
    report["takeover"] = takeover_report

    # =========================================================================
    # 4. Zero-Shot 26-Letter English Wordle via Outer Dictionary Trie Mask!
    # =========================================================================
    if not args.focus_adversarial:
        print(f"\n=== [4] Zero-Shot 26-Letter English Wordle (Trie over {len(ENGLISH_WORDLE_WORDS)} real words) ===", flush=True)
        trie = _build_word_trie(ENGLISH_WORDLE_WORDS)
        test_words = ("PLANT", "PRIME", "CRANE", "WORLD", "BRAIN", "MAGIC", "SOUND", "WATER", "STONE", "LIGHT")
        w_wins = 0
        w_turns = []
        w_details = []
        for w in test_words:
            res = play_english_wordle_trie(model, w, trie, device, tok, max_turns=10, beam_width=8)
            w_wins += int(res["won"])
            w_turns.append(res["turns"])
            w_details.append(res)
            print(f"  Secret={w} | won={str(res['won']):5s} | turns={res['turns']:2d} | path={' -> '.join(res['guesses'])}", flush=True)
        report["wordle"] = {
            "wins": w_wins, "games": len(test_words),
            "win_rate": round(w_wins / len(test_words), 3),
            "avg_turns": round(sum(w_turns) / len(w_turns), 2),
            "details": w_details,
        }
        print(f"  Wordle Summary: Win={w_wins}/{len(test_words)} ({w_wins*10}%) | AvgTurns={sum(w_turns)/len(w_turns):.2f}", flush=True)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\nTotal stress-test wall time: {time.perf_counter() - t0:.1f}s")
