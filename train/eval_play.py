"""Tiered batched autoregressive gameplay evaluator + X-ray constraint diagnostics.

Evaluates PrimeGPT live against `engine.game` across Small / Mid / Large difficulty
tiers using batched KV-Cache decoding (5x-8x faster than B=1 sequential play!),
measuring:
  - Win rate & average turns per tier
  - `raw_syntax_err`: fraction of turns where the model's unconstrained raw argmax
    would have emitted a grammatically illegal token (slot count, length, out-of-range S_i)
  - 3 sound mathematical "mine-stepping" (踩雷) rates via `BoardKnowledge`:
      * `wrong_mine`:  placing a prime already 100% excluded by a prior WRONG at that slot
      * `exact_drop`:  changing a slot that was already confirmed EXACT in an earlier turn
      * `repeat_mine`: re-guessing the exact same failed Gödel product at that slot
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from engine.game import GameConfig, GameSession
from engine.primes import encode_slot
from engine.tokenizer import PrimeWordleTokenizer
from model import PrimeGPT
from solver.constraints import BoardKnowledge, symbols_to_mask


@dataclass
class PlayOutcome:
    won: bool
    turns: int
    syntax_errors: int
    raw_syntax_turn_violations: int
    raw_syntax_tok_violations: int
    raw_syntax_tok_total: int
    wrong_mine_hits: int
    wrong_mine_opps: int
    exact_drop_hits: int
    exact_drop_opps: int
    repeat_mine_hits: int
    repeat_mine_opps: int
    total_tokens: int


def _parse_guess_tokens(
    tok_ids: list[int],
    tokenizer: PrimeWordleTokenizer,
    config: GameConfig,
) -> list[list[int]] | None:
    """Parse generated tokens between <TURN> and <FB> into validated guess_slots."""
    tokens = tokenizer.decode(tok_ids)
    if not tokens or tokens[0] != "<TURN>" or tokens[-1] != "<FB>":
        return None

    inner = tokens[1:-1]
    slots: list[list[int]] = []
    cur: list[int] | None = None

    for tok in inner:
        if tok == "<SLOT>":
            if cur is not None:
                slots.append(cur)
            cur = []
        elif tok.startswith("S_"):
            if cur is None:
                return None
            sym = int(tok[2:])
            if sym < 0 or sym >= config.num_primes:
                return None
            cur.append(sym)
        else:
            return None

    if cur is not None:
        slots.append(cur)

    if len(slots) != config.code_length:
        return None
    for s in slots:
        if len(s) < 1 or len(s) > config.max_multiplier:
            return None
        if not config.allow_underfill and len(s) != config.max_multiplier:
            return None
    return slots


def _allowed_next_tokens(
    step: int,
    slots_started: int,
    cur_slot_len: int,
    last_sym_idx: int,
    config: GameConfig,
    turn_id: int,
    slot_id: int,
    fb_id: int,
    sym_ids: list[int],
) -> list[int]:
    if step == 0:
        return [turn_id]
    if slots_started == 0:
        return [slot_id]
    allowed: list[int] = []
    can_add_sym = cur_slot_len < config.max_multiplier
    can_close_slot = (
        cur_slot_len >= 1
        if config.allow_underfill
        else cur_slot_len == config.max_multiplier
    )
    if can_add_sym:
        start_s = max(0, last_sym_idx)
        allowed.extend(sym_ids[start_s:])
    if can_close_slot:
        if slots_started < config.code_length:
            allowed.append(slot_id)
        else:
            allowed.append(fb_id)
    return allowed


@torch.no_grad()
def play_games_batched(
    model: PrimeGPT,
    config: GameConfig,
    seeds: Sequence[int],
    device: torch.device,
    tokenizer: PrimeWordleTokenizer | None = None,
    max_turns: int = 20,
    constrained_syntax: bool = True,
) -> list[PlayOutcome]:
    """Play `len(seeds)` games of the same config in parallel via batched KV-Cache."""
    bsz = len(seeds)
    if bsz == 0:
        return []

    tok = tokenizer if tokenizer is not None else PrimeWordleTokenizer(max_symbols=32)
    sessions = [GameSession(config, seed=s) for s in seeds]

    pad_id = tok.pad_id
    turn_id = tok.stoi["<TURN>"]
    slot_id = tok.stoi["<SLOT>"]
    fb_id = tok.stoi["<FB>"]
    sym_ids = [tok.stoi[f"S_{i}"] for i in range(config.num_primes)]
    sym_id_set = set(sym_ids)

    header = tok.encode_config(config)
    hdr_len = len(header)
    prompt = torch.tensor([header] * bsz, dtype=torch.long, device=device)
    cur_pos = [hdr_len] * bsz
    pos_init = torch.arange(hdr_len, dtype=torch.long, device=device)[None, :].expand(bsz, -1)
    valid_mask = torch.ones((bsz, hdr_len), dtype=torch.bool, device=device)

    logits, kv, _ = model(
        prompt, use_cache=True, position_ids=pos_init, key_padding_mask=valid_mask
    )

    syntax_errors = [0] * bsz
    raw_turn_viol = [0] * bsz
    raw_tok_viol = [0] * bsz
    raw_tok_total = [0] * bsz
    wrong_h = [0] * bsz
    wrong_o = [0] * bsz
    exact_h = [0] * bsz
    exact_o = [0] * bsz
    rep_h = [0] * bsz
    rep_o = [0] * bsz
    total_toks = [hdr_len] * bsz
    alive = [True] * bsz

    max_gen_per_turn = 2 + config.code_length * (config.max_multiplier + 1)

    for _turn_idx in range(max_turns):
        active_indices = [b for b in range(bsz) if alive[b] and sessions[b].status == "playing"]
        if not active_indices:
            break

        gen_ids: list[list[int]] = [[] for _ in range(bsz)]
        slots_started = [0] * bsz
        cur_slot_len = [0] * bsz
        last_sym_idx = [-1] * bsz
        turn_done = [not (alive[b] and sessions[b].status == "playing") for b in range(bsz)]
        turn_had_raw_viol = [False] * bsz

        for step in range(max_gen_per_turn):
            if all(turn_done):
                break

            step_logits_all = logits[:, -1, :].float()
            next_ids = [pad_id] * bsz
            next_pos = list(cur_pos)
            next_valid = [False] * bsz

            for b in range(bsz):
                if turn_done[b]:
                    continue
                allowed = _allowed_next_tokens(
                    step,
                    slots_started[b],
                    cur_slot_len[b],
                    last_sym_idx[b],
                    config,
                    turn_id,
                    slot_id,
                    fb_id,
                    sym_ids,
                )
                row_logits = step_logits_all[b]
                raw_pred = int(row_logits.argmax().item())
                raw_tok_total[b] += 1
                if raw_pred not in allowed:
                    raw_tok_viol[b] += 1
                    turn_had_raw_viol[b] = True

                if constrained_syntax:
                    best_id = allowed[0]
                    best_val = float(row_logits[best_id].item())
                    for cand_id in allowed[1:]:
                        v = float(row_logits[cand_id].item())
                        if v > best_val:
                            best_val = v
                            best_id = cand_id
                    chosen = best_id
                else:
                    chosen = raw_pred

                gen_ids[b].append(chosen)
                if chosen == slot_id:
                    slots_started[b] += 1
                    cur_slot_len[b] = 0
                    last_sym_idx[b] = -1
                elif chosen in sym_id_set:
                    cur_slot_len[b] += 1
                    last_sym_idx[b] = chosen - sym_ids[0]

                next_ids[b] = chosen
                next_pos[b] = cur_pos[b]
                next_valid[b] = True
                cur_pos[b] += 1
                total_toks[b] += 1

                if chosen == fb_id:
                    turn_done[b] = True

            inp = torch.tensor(next_ids, dtype=torch.long, device=device)[:, None]
            pos_t = torch.tensor(next_pos, dtype=torch.long, device=device)[:, None]
            val_col = torch.tensor(next_valid, dtype=torch.bool, device=device)[:, None]
            valid_mask = torch.cat([valid_mask, val_col], dim=1)

            logits, kv, _ = model(
                inp,
                past_kv=kv,
                use_cache=True,
                position_ids=pos_t,
                key_padding_mask=valid_mask,
            )

        # Submit parsed guesses and batch-feed L feedback tokens
        L = config.code_length
        fb_rows = [[pad_id] * L for _ in range(bsz)]
        fb_pos_rows = [[cur_pos[b]] * L for b in range(bsz)]
        fb_valid_rows = [[False] * L for _ in range(bsz)]
        any_submitted = False

        for b in active_indices:
            if turn_had_raw_viol[b]:
                raw_turn_viol[b] += 1
            parsed = _parse_guess_tokens(gen_ids[b], tok, config)
            if parsed is None:
                syntax_errors[b] += 1
                alive[b] = False
                continue

            sess = sessions[b]
            if sess.history:
                knowledge = BoardKnowledge(sess)
                for i, slot_syms in enumerate(parsed):
                    prod = encode_slot(slot_syms, sess.primes)
                    mask = symbols_to_mask(slot_syms)
                    if i in knowledge.confirmed_exact:
                        exact_o[b] += 1
                        if prod != knowledge.solved_products[i]:
                            exact_h[b] += 1
                    else:
                        if knowledge.excluded_mask[i] != 0:
                            wrong_o[b] += 1
                            if (mask & knowledge.excluded_mask[i]) != 0:
                                wrong_h[b] += 1
                        if knowledge.tried_at_slot[i]:
                            rep_o[b] += 1
                            if prod in knowledge.tried_at_slot[i]:
                                rep_h[b] += 1

            rec = sess.submit_guess(parsed)
            fb_rows[b] = [tok.stoi[fb.value] for fb in rec.feedback]
            fb_pos_rows[b] = [cur_pos[b] + k for k in range(L)]
            fb_valid_rows[b] = [True] * L
            cur_pos[b] += L
            total_toks[b] += L
            any_submitted = True

        if any_submitted:
            fb_t = torch.tensor(fb_rows, dtype=torch.long, device=device)
            fb_pos_t = torch.tensor(fb_pos_rows, dtype=torch.long, device=device)
            fb_val_t = torch.tensor(fb_valid_rows, dtype=torch.bool, device=device)
            valid_mask = torch.cat([valid_mask, fb_val_t], dim=1)
            logits, kv, _ = model(
                fb_t,
                past_kv=kv,
                use_cache=True,
                position_ids=fb_pos_t,
                key_padding_mask=valid_mask,
            )

    return [
        PlayOutcome(
            won=(sessions[b].status == "won"),
            turns=len(sessions[b].history),
            syntax_errors=syntax_errors[b],
            raw_syntax_turn_violations=raw_turn_viol[b],
            raw_syntax_tok_violations=raw_tok_viol[b],
            raw_syntax_tok_total=raw_tok_total[b],
            wrong_mine_hits=wrong_h[b],
            wrong_mine_opps=wrong_o[b],
            exact_drop_hits=exact_h[b],
            exact_drop_opps=exact_o[b],
            repeat_mine_hits=rep_h[b],
            repeat_mine_opps=rep_o[b],
            total_tokens=total_toks[b],
        )
        for b in range(bsz)
    ]


@torch.no_grad()
def play_game_with_model(
    model: PrimeGPT,
    config: GameConfig,
    seed: int,
    device: torch.device,
    tokenizer: PrimeWordleTokenizer | None = None,
    max_turns: int = 20,
    constrained_syntax: bool = True,
) -> PlayOutcome:
    """Single-game convenience wrapper over `play_games_batched`."""
    return play_games_batched(
        model,
        config,
        seeds=[seed],
        device=device,
        tokenizer=tokenizer,
        max_turns=max_turns,
        constrained_syntax=constrained_syntax,
    )[0]


def _safe_rate(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return round(num / den, 4)


def evaluate_model_play(
    model: PrimeGPT,
    device: torch.device,
    games_per_preset: int = 8,
) -> dict[str, Any]:
    """Run fast batched tiered live gameplay evaluation (Small / Mid / Large) + X-ray diagnostics."""
    was_training = model.training
    model.eval()
    tok = PrimeWordleTokenizer(max_symbols=32)

    presets = [
        # Small tier
        ("small", "K1_P6_L5", GameConfig(6, 1, 5, allow_underfill=False), 10_000, 12),
        ("small", "K2_P5_L5_U", GameConfig(5, 2, 5, allow_underfill=True), 30_000, 15),
        # Mid tier
        ("mid", "K3_P8_L5_U", GameConfig(8, 3, 5, allow_underfill=True), 50_000, 18),
        ("mid", "K4_P16_L6_U", GameConfig(16, 4, 6, allow_underfill=True), 60_000, 20),
        # Large tier
        ("large", "K6_P25_L8_U", GameConfig(25, 6, 8, allow_underfill=True), 70_000, 22),
    ]

    per_preset: dict[str, dict[str, Any]] = {}
    bucket_wins: dict[str, list[int]] = {"small": [], "mid": [], "large": []}

    tot_turns = 0
    tot_raw_turn_viol = 0
    tot_raw_tok_viol = 0
    tot_raw_tok = 0
    tot_wrong_h = 0
    tot_wrong_o = 0
    tot_exact_h = 0
    tot_exact_o = 0
    tot_rep_h = 0
    tot_rep_o = 0

    with torch.amp.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
    ):
        for bucket, name, cfg, base_seed, max_t in presets:
            seeds = [base_seed + i for i in range(games_per_preset)]
            outcomes = play_games_batched(
                model,
                cfg,
                seeds=seeds,
                device=device,
                tokenizer=tok,
                max_turns=max_t,
                constrained_syntax=True,
            )

            wins = sum(int(o.won) for o in outcomes)
            turns_sum = sum(o.turns for o in outcomes)
            p_raw_turn = sum(o.raw_syntax_turn_violations for o in outcomes)
            p_wrong_h = sum(o.wrong_mine_hits for o in outcomes)
            p_wrong_o = sum(o.wrong_mine_opps for o in outcomes)
            p_exact_h = sum(o.exact_drop_hits for o in outcomes)
            p_exact_o = sum(o.exact_drop_opps for o in outcomes)
            p_rep_h = sum(o.repeat_mine_hits for o in outcomes)
            p_rep_o = sum(o.repeat_mine_opps for o in outcomes)

            bucket_wins[bucket].extend(int(o.won) for o in outcomes)
            tot_turns += turns_sum
            tot_raw_turn_viol += p_raw_turn
            tot_raw_tok_viol += sum(o.raw_syntax_tok_violations for o in outcomes)
            tot_raw_tok += sum(o.raw_syntax_tok_total for o in outcomes)
            tot_wrong_h += p_wrong_h
            tot_wrong_o += p_wrong_o
            tot_exact_h += p_exact_h
            tot_exact_o += p_exact_o
            tot_rep_h += p_rep_h
            tot_rep_o += p_rep_o

            per_preset[name] = {
                "bucket": bucket,
                "win_rate": round(wins / games_per_preset, 3),
                "avg_turns": round(turns_sum / games_per_preset, 2),
                "raw_syntax_err": _safe_rate(p_raw_turn, turns_sum),
                "wrong_mine": _safe_rate(p_wrong_h, p_wrong_o),
                "exact_drop": _safe_rate(p_exact_h, p_exact_o),
                "repeat_mine": _safe_rate(p_rep_h, p_rep_o),
            }

    if was_training:
        model.train()

    return {
        "buckets": {
            b: round(sum(vals) / max(1, len(vals)), 3) for b, vals in bucket_wins.items()
        },
        "xray": {
            "raw_syntax_turn_err": _safe_rate(tot_raw_turn_viol, tot_turns),
            "raw_syntax_tok_err": _safe_rate(tot_raw_tok_viol, tot_raw_tok),
            "wrong_mine": _safe_rate(tot_wrong_h, tot_wrong_o),
            "exact_drop": _safe_rate(tot_exact_h, tot_exact_o),
            "repeat_mine": _safe_rate(tot_rep_h, tot_rep_o),
        },
        "presets": per_preset,
    }
