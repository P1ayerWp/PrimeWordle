"""Phase 4 Generalist GRPO v2 Trainer for DeepLogic PrimeGPT.

Upgrades over v1 (`train/grpo.py`) informed directly by the 310-game blind exam:
  1. Zero hardcoded test shapes — 100% continuous random (P, K, L) generalist pool:
       - 25% Gen_Mid_K24    (P=4..24, K=2..4, L=3..12, 45% dirty prefix 0..7t)
       - 25% Gen_Sparse_PvZ (P=6..18, K=2, L=5..16, 30..70% forbidden + p*p bans, 60% U_OFF)
       - 15% Gen_Wide_K1    (P=16..32, K=1, L=4..10 -> fortifies 5/6/7-letter English Wordle!)
       - 15% Gen_K58_Slice  (P=10..32, K=5..8, L=3..12 -> CPU solver lays first ~60%, GRPO finishes endgame!)
       - 10% Gen_UNK_CFG    (true hidden-rule <UNK_CFG> blind games: half ZeroPrefix, half WithPrefix)
       - 10% Gen_Small_K12  (P=3..14, K=1..2, L=3..10, 45% dirty prefix 1..5t)
  2. Stronger guardrails:
       - `kl_coef = 0.05` (tighter anchor to Adv100M Wordle / syntax circuits)
       - `wrong_mine` penalty increased `-2.0 -> -3.0` (crushes PvZ sparse exclusion mines)
       - Gentle Cosine LR decay: `1.5e-5 -> 3.0e-6`
       - Intermediate checkpoints saved every 25 steps (`step_0025.pt`, `step_0050.pt`, ...)
       - Strictly AntiRep=OFF during rollout + exact per-token allowed_masks alignment (`ratio==1.0`)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from engine.game import GameConfig, GameSession
from engine.primes import encode_slot
from engine.tokenizer import PrimeWordleTokenizer
from model import PrimeGPT, PrimeGPTConfig
from solver.adversarial_dataset import _play_messy_human_prefix
from solver.constraints import BoardKnowledge, count_slot_candidates, symbols_to_mask
from solver.hybrid import HybridSolver
from train.grpo import (
    RolloutEpisode,
    _set_windows_keep_awake,
    compute_group_advantages,
    grpo_update_step,
    save_grpo_checkpoint,
)
from train.stress_test import _slot_legal_extensions, make_forbidden_fusion_table


@dataclass
class GeneralistPromptSpec:
    category: str
    config: GameConfig
    target_slots: list[list[int]]
    prefix_guesses: list[list[list[int]]]
    allowed_pairs: dict[int, set[int]] | None
    max_new_turns: int
    reveal_k: bool = True
    reveal_underfill: bool = True
    reveal_dup: bool = True
    raw_unk_length: bool = False
    seed: int = 0


def sample_generalist_category(rng: random.Random) -> str:
    u = rng.random()
    if u < 0.25:
        return "Gen_Mid_K24"
    if u < 0.50:
        return "Gen_Sparse_PvZ"
    if u < 0.65:
        return "Gen_Wide_K1"
    if u < 0.80:
        return "Gen_K58_Slice"
    if u < 0.90:
        return "Gen_UNK_CFG"
    return "Gen_Small_K12"


def build_generalist_prompt(category: str, seed: int) -> GeneralistPromptSpec:
    rng = random.Random(seed)
    messy_rng = random.Random(seed * 11 + 777_773)

    if category == "Gen_Mid_K24":
        p = rng.randint(4, 24)
        k = rng.randint(2, 4)
        l = rng.randint(3, 12)
        allow_u = rng.random() < 0.60
        probe = GameConfig(p, k, l, allow_underfill=allow_u, allow_duplicate_slots=True)
        allow_dup = not (l <= count_slot_candidates(probe) and rng.random() < 0.30)
        cfg = GameConfig(p, k, l, allow_underfill=allow_u, allow_duplicate_slots=allow_dup)
        sess = GameSession(cfg, seed=seed)
        prefix: list[list[list[int]]] = []
        if rng.random() < 0.45:
            _play_messy_human_prefix(sess, messy_rng, rng.randint(1, 7))
            if sess.status == "playing":
                prefix = [[sorted(s) for s in r.guess_slots] for r in sess.history]
        return GeneralistPromptSpec(
            category=category,
            config=cfg,
            target_slots=[list(s) for s in sess.target_slots],
            prefix_guesses=prefix,
            allowed_pairs=None,
            max_new_turns=18,
            seed=seed,
        )

    if category == "Gen_Sparse_PvZ":
        p = rng.randint(6, 18)
        l = rng.randint(5, 16)
        allow_u = rng.random() < 0.40  # 60% UNDERFILL_OFF
        forbid_r = rng.uniform(0.30, 0.70)
        self_r = rng.uniform(0.30, 1.00)
        allowed_pairs, legal_slots = make_forbidden_fusion_table(
            p, forbid_ratio=forbid_r, allow_underfill=allow_u, seed=seed + 31, forbid_self_ratio=self_r
        )
        allow_dup = not (l <= len(legal_slots) and rng.random() < 0.25)
        cfg = GameConfig(p, 2, l, allow_underfill=allow_u, allow_duplicate_slots=allow_dup)
        if allow_dup:
            target = [list(rng.choice(legal_slots)) for _ in range(l)]
        else:
            target = [list(s) for s in rng.sample(legal_slots, l)]
        sess = GameSession(cfg, seed=seed, target_slots=target)
        prefix = []
        if rng.random() < 0.50:
            _play_messy_human_prefix(sess, messy_rng, rng.randint(2, 6), legal_slots=legal_slots)
            if sess.status == "playing":
                prefix = [[sorted(s) for s in r.guess_slots] for r in sess.history]
        return GeneralistPromptSpec(
            category=category,
            config=cfg,
            target_slots=[list(s) for s in sess.target_slots],
            prefix_guesses=prefix,
            allowed_pairs=allowed_pairs,
            max_new_turns=18,
            seed=seed,
        )

    if category == "Gen_Wide_K1":
        p = rng.randint(16, 32)
        l = rng.randint(4, 10)
        allow_dup = rng.random() < 0.65
        cfg = GameConfig(p, 1, l, allow_underfill=False, allow_duplicate_slots=allow_dup)
        sess = GameSession(cfg, seed=seed)
        prefix = []
        if rng.random() < 0.20:
            _play_messy_human_prefix(sess, messy_rng, rng.randint(1, 3))
            if sess.status == "playing":
                prefix = [[sorted(s) for s in r.guess_slots] for r in sess.history]
        return GeneralistPromptSpec(
            category=category,
            config=cfg,
            target_slots=[list(s) for s in sess.target_slots],
            prefix_guesses=prefix,
            allowed_pairs=None,
            max_new_turns=18,
            seed=seed,
        )

    if category == "Gen_K58_Slice":
        p = rng.randint(10, 32)
        k = rng.randint(5, 8)
        l = rng.randint(3, 10)
        allow_u = rng.random() < 0.70
        cfg = GameConfig(p, k, l, allow_underfill=allow_u, allow_duplicate_slots=True)
        sim = GameSession(cfg, seed=seed)
        solver = HybridSolver(seed=seed + 19, diversity=0.05)
        while sim.status == "playing" and len(sim.history) < 85:
            sim.submit_guess(solver.next_guess(sim))
        total_t = len(sim.history)
        # Leave roughly last 35%..45% (~7..14 turns) for GRPO to finish cleanly & fast!
        target_rem = rng.randint(7, 13)
        cut = max(1, total_t - target_rem)
        prefix = [[sorted(s) for s in r.guess_slots] for r in sim.history[:cut]]
        return GeneralistPromptSpec(
            category=category,
            config=cfg,
            target_slots=[list(s) for s in sim.target_slots],
            prefix_guesses=prefix,
            allowed_pairs=None,
            max_new_turns=min(18, target_rem + 5),
            seed=seed,
        )

    if category == "Gen_UNK_CFG":
        if rng.random() < 0.50:
            # UNK_ZeroPrefix: UNDERFILL_ON, hide K & DUP
            p = rng.randint(5, 16)
            k = rng.randint(2, 5)
            l = rng.randint(4, 8)
            allow_dup = rng.random() < 0.70
            cfg = GameConfig(p, k, l, allow_underfill=True, allow_duplicate_slots=allow_dup)
            sess = GameSession(cfg, seed=seed)
            return GeneralistPromptSpec(
                category="Gen_UNK_Zero",
                config=cfg,
                target_slots=[list(s) for s in sess.target_slots],
                prefix_guesses=[],
                allowed_pairs=None,
                max_new_turns=18,
                reveal_k=False,
                reveal_underfill=True,
                reveal_dup=False,
                raw_unk_length=True,
                seed=seed,
            )
        else:
            # UNK_WithPrefix: hide ALL THREE (K, UNDERFILL, DUP), give 2-3 solver prefix turns
            p = rng.randint(5, 16)
            k = rng.randint(1, 4)
            l = rng.randint(4, 8)
            allow_u = False if k == 1 else (rng.random() < 0.55)
            allow_dup = rng.random() < 0.70
            cfg = GameConfig(p, k, l, allow_underfill=allow_u, allow_duplicate_slots=allow_dup)
            sim = GameSession(cfg, seed=seed)
            solver = HybridSolver(seed=seed + 23)
            for _ in range(rng.randint(2, 3)):
                if sim.status != "playing":
                    break
                sim.submit_guess(solver.next_guess(sim))
            if sim.status != "playing":
                sim = GameSession(cfg, seed=seed + 91_000)
                sim.submit_guess(solver.next_guess(sim))
            prefix = [[sorted(s) for s in r.guess_slots] for r in sim.history]
            return GeneralistPromptSpec(
                category="Gen_UNK_Pref",
                config=cfg,
                target_slots=[list(s) for s in sim.target_slots],
                prefix_guesses=prefix,
                allowed_pairs=None,
                max_new_turns=16,
                reveal_k=False,
                reveal_underfill=False,
                reveal_dup=False,
                raw_unk_length=True,
                seed=seed,
            )

    # Gen_Small_K12
    p = rng.randint(3, 14)
    k = rng.randint(1, 2)
    l = rng.randint(3, 10)
    allow_u = False if k == 1 else (rng.random() < 0.60)
    probe = GameConfig(p, k, l, allow_underfill=allow_u, allow_duplicate_slots=True)
    allow_dup = not (l <= count_slot_candidates(probe) and rng.random() < 0.35)
    cfg = GameConfig(p, k, l, allow_underfill=allow_u, allow_duplicate_slots=allow_dup)
    sess = GameSession(cfg, seed=seed)
    prefix = []
    if rng.random() < 0.45:
        _play_messy_human_prefix(sess, messy_rng, rng.randint(1, 5))
        if sess.status == "playing":
            prefix = [[sorted(s) for s in r.guess_slots] for r in sess.history]
    return GeneralistPromptSpec(
        category="Gen_Small_K12",
        config=cfg,
        target_slots=[list(s) for s in sess.target_slots],
        prefix_guesses=prefix,
        allowed_pairs=None,
        max_new_turns=14,
        seed=seed,
    )


def compute_v2_reward(
    won: bool,
    rule_violated: bool,
    added_turns: int,
    max_turns: int,
    final_exact_ratio: float,
    wrong_hits: int,
    wrong_opps: int,
    exact_drop_hits: int,
    exact_drop_opps: int,
    repeat_hits: int,
    repeat_opps: int,
    wrong_penalty: float = 3.0,
) -> float:
    """v2 verifiable reward: stronger wrong_mine penalty (-3.0) + explicit UNK rule penalty."""
    if rule_violated:
        return -2.5

    if won:
        r = 2.0 + 1.0 * max(0.0, 1.0 - added_turns / max(1, max_turns))
    else:
        r = 0.8 * final_exact_ratio

    if repeat_opps > 0:
        r -= 2.5 * (repeat_hits / repeat_opps)
    if repeat_hits > 0:
        r -= 0.05 * min(10, repeat_hits)

    if wrong_opps > 0:
        r -= wrong_penalty * (wrong_hits / wrong_opps)

    if exact_drop_opps > 0:
        r -= 4.0 * (exact_drop_hits / exact_drop_opps)
    if exact_drop_hits > 0:
        r -= 0.20 * min(5, exact_drop_hits)

    return float(r)


@torch.no_grad()
def rollout_generalist_group(
    model: PrimeGPT,
    spec: GeneralistPromptSpec,
    prompt_idx: int,
    group_size: int,
    temperature: float,
    device: torch.device,
    tokenizer: PrimeWordleTokenizer,
    wrong_penalty: float = 3.0,
) -> list[RolloutEpisode]:
    """Batched G-rollout sampler supporting forbidden pairs, K5-8 slices, AND true raw <UNK_CFG>."""
    was_training = model.training
    model.eval()

    cfg = spec.config
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_id
    turn_id = tokenizer.stoi["<TURN>"]
    slot_id = tokenizer.stoi["<SLOT>"]
    fb_id = tokenizer.stoi["<FB>"]
    s0_id = tokenizer.stoi["S_0"]
    sym_ids = [s0_id + i for i in range(cfg.num_primes)]
    sym_id_set = set(sym_ids)

    sessions = [
        GameSession(cfg, seed=spec.seed, target_slots=spec.target_slots)
        for _ in range(group_size)
    ]
    for g_slots in spec.prefix_guesses:
        for s in sessions:
            if s.status == "playing":
                s.submit_guess(g_slots)

    prompt_ids = tokenizer.encode_config(
        cfg,
        reveal_k=spec.reveal_k,
        reveal_underfill=spec.reveal_underfill,
        reveal_dup=spec.reveal_dup,
    )
    for turn in sessions[0].history:
        prompt_ids.append(turn_id)
        for slot in turn.guess_slots:
            prompt_ids.append(slot_id)
            for sym in sorted(slot):
                prompt_ids.append(s0_id + sym)
        prompt_ids.append(fb_id)
        for fb in turn.feedback:
            prompt_ids.append(tokenizer.stoi[fb.value])

    prefix_len = len(prompt_ids)
    all_true_mask = [True] * vocab_size

    ep_ids: list[list[int]] = [list(prompt_ids) for _ in range(group_size)]
    ep_action_mask: list[list[int]] = [[0] * prefix_len for _ in range(group_size)]
    ep_allowed_masks: list[list[list[bool]]] = [
        [all_true_mask for _ in range(prefix_len)] for _ in range(group_size)
    ]

    wrong_h = [0] * group_size
    wrong_o = [0] * group_size
    exact_h = [0] * group_size
    exact_o = [0] * group_size
    rep_h = [0] * group_size
    rep_o = [0] * group_size
    added_turns = [0] * group_size
    rule_violated = [False] * group_size
    alive = [sessions[g].status == "playing" for g in range(group_size)]

    prompt_t = torch.tensor([prompt_ids] * group_size, dtype=torch.long, device=device)
    cur_pos = [prefix_len] * group_size
    pos_init = torch.arange(prefix_len, dtype=torch.long, device=device)[None, :].expand(
        group_size, -1
    )
    valid_mask = torch.ones((group_size, prefix_len), dtype=torch.bool, device=device)

    max_k_gen = 8 if spec.raw_unk_length else cfg.max_multiplier
    max_gen_per_turn = 2 + cfg.code_length * (max_k_gen + 1)
    inv_temp = 1.0 / max(1e-3, temperature)

    with torch.amp.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
    ):
        logits, kv, _ = model(
            prompt_t, use_cache=True, position_ids=pos_init, key_padding_mask=valid_mask
        )

        for _turn_i in range(spec.max_new_turns):
            active = [g for g in range(group_size) if alive[g] and sessions[g].status == "playing"]
            if not active:
                break

            gen_slots_ids: list[list[int]] = [[] for _ in range(group_size)]
            slots_started = [0] * group_size
            cur_slot_syms: list[list[int]] = [[] for _ in range(group_size)]
            turn_done = [not (alive[g] and sessions[g].status == "playing") for g in range(group_size)]

            for step in range(max_gen_per_turn):
                if all(turn_done):
                    break

                step_logits = logits[:, -1, :].float()
                next_ids = [pad_id] * group_size
                next_pos = list(cur_pos)
                next_valid = [False] * group_size

                for g in range(group_size):
                    if turn_done[g]:
                        continue

                    allowed: list[int] = []
                    if step == 0:
                        allowed = [turn_id]
                    elif slots_started[g] == 0:
                        allowed = [slot_id]
                    elif spec.raw_unk_length:
                        # True <UNK_CFG>: do NOT leak true K or underfill via mask!
                        cur_len = len(cur_slot_syms[g])
                        if cur_len < 8:
                            start_s = cur_slot_syms[g][-1] if cur_len > 0 else 0
                            allowed.extend(s0_id + s for s in range(start_s, cfg.num_primes))
                        if cur_len >= 1:
                            allowed.append(
                                slot_id if slots_started[g] < cfg.code_length else fb_id
                            )
                    else:
                        # Strictly AntiRep=OFF (forbidden_slot_prods=None)
                        next_syms, can_close = _slot_legal_extensions(
                            cur_slot_syms[g],
                            cfg,
                            sessions[g].primes,
                            spec.allowed_pairs,
                            forbidden_slot_prods=None,
                        )
                        allowed.extend(sym_ids[s] for s in next_syms)
                        if can_close:
                            allowed.append(
                                slot_id if slots_started[g] < cfg.code_length else fb_id
                            )

                    mask_vec = [False] * vocab_size
                    for aid in allowed:
                        mask_vec[aid] = True

                    if len(allowed) == 1:
                        chosen = allowed[0]
                    else:
                        row = step_logits[g]
                        cand_logits = row[allowed] * inv_temp
                        probs = F.softmax(cand_logits, dim=-1)
                        picked_idx = int(torch.multinomial(probs, num_samples=1).item())
                        chosen = allowed[picked_idx]

                    gen_slots_ids[g].append(chosen)
                    ep_ids[g].append(chosen)
                    ep_action_mask[g].append(1)
                    ep_allowed_masks[g].append(mask_vec)

                    if chosen == slot_id:
                        slots_started[g] += 1
                        cur_slot_syms[g] = []
                    elif chosen in sym_id_set:
                        cur_slot_syms[g].append(chosen - s0_id)

                    next_ids[g] = chosen
                    next_pos[g] = cur_pos[g]
                    next_valid[g] = True
                    cur_pos[g] += 1
                    if chosen == fb_id:
                        turn_done[g] = True

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

            L = cfg.code_length
            fb_rows = [[pad_id] * L for _ in range(group_size)]
            fb_pos_rows = [[cur_pos[g]] * L for _ in range(group_size)]
            fb_valid_rows = [[False] * L for _ in range(group_size)]
            any_submitted = False

            for g in active:
                slots: list[list[int]] = []
                cur: list[int] | None = None
                for tid in gen_slots_ids[g][1:-1]:
                    if tid == slot_id:
                        if cur is not None:
                            slots.append(cur)
                        cur = []
                    elif tid in sym_id_set and cur is not None:
                        cur.append(tid - s0_id)
                if cur is not None:
                    slots.append(cur)

                # Check hidden-rule length legality for <UNK_CFG>
                valid_len = len(slots) == cfg.code_length
                if valid_len:
                    for sl in slots:
                        if len(sl) < 1 or len(sl) > cfg.max_multiplier or (
                            not cfg.allow_underfill and len(sl) != cfg.max_multiplier
                        ):
                            valid_len = False
                            break

                if not valid_len:
                    rule_violated[g] = True
                    alive[g] = False
                    continue

                sess = sessions[g]
                if sess.history:
                    knowledge = BoardKnowledge(sess)
                    for i, slot_syms in enumerate(slots):
                        prod = encode_slot(slot_syms, sess.primes)
                        mask = symbols_to_mask(slot_syms)
                        if i in knowledge.confirmed_exact:
                            exact_o[g] += 1
                            if prod != knowledge.solved_products[i]:
                                exact_h[g] += 1
                        else:
                            if knowledge.excluded_mask[i] != 0:
                                wrong_o[g] += 1
                                if (mask & knowledge.excluded_mask[i]) != 0:
                                    wrong_h[g] += 1
                            if knowledge.tried_at_slot[i]:
                                rep_o[g] += 1
                                if prod in knowledge.tried_at_slot[i]:
                                    rep_h[g] += 1

                rec = sess.submit_guess(slots)
                added_turns[g] += 1
                fb_toks = [tokenizer.stoi[fb.value] for fb in rec.feedback]
                ep_ids[g].extend(fb_toks)
                ep_action_mask[g].extend([0] * L)
                ep_allowed_masks[g].extend([all_true_mask for _ in range(L)])

                fb_rows[g] = fb_toks
                fb_pos_rows[g] = [cur_pos[g] + j for j in range(L)]
                fb_valid_rows[g] = [True] * L
                cur_pos[g] += L
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

    if was_training:
        model.train()

    episodes: list[RolloutEpisode] = []
    for g in range(group_size):
        sess = sessions[g]
        won = (sess.status == "won") and (not rule_violated[g])
        if won:
            final_exact_ratio = 1.0
        elif sess.history:
            k_final = BoardKnowledge(sess)
            final_exact_ratio = len(k_final.confirmed_exact) / max(1, cfg.code_length)
        else:
            final_exact_ratio = 0.0

        rew = compute_v2_reward(
            won=won,
            rule_violated=rule_violated[g],
            added_turns=added_turns[g],
            max_turns=spec.max_new_turns,
            final_exact_ratio=final_exact_ratio,
            wrong_hits=wrong_h[g],
            wrong_opps=wrong_o[g],
            exact_drop_hits=exact_h[g],
            exact_drop_opps=exact_o[g],
            repeat_hits=rep_h[g],
            repeat_opps=rep_o[g],
            wrong_penalty=wrong_penalty,
        )
        episodes.append(
            RolloutEpisode(
                category=spec.category,
                prompt_idx=prompt_idx,
                input_ids=ep_ids[g],
                action_mask=ep_action_mask[g],
                allowed_masks=ep_allowed_masks[g],
                reward=rew,
                won=won,
                added_turns=added_turns[g],
                max_turns=spec.max_new_turns,
                final_exact_ratio=final_exact_ratio,
                wrong_mine_hits=wrong_h[g],
                wrong_mine_opps=wrong_o[g],
                exact_drop_hits=exact_h[g],
                exact_drop_opps=exact_o[g],
                repeat_mine_hits=rep_h[g],
                repeat_mine_opps=rep_o[g],
            )
        )
    return episodes


def _cosine_lr(step: int, planned_steps: int, max_lr: float, min_lr: float) -> float:
    if step >= planned_steps:
        return min_lr
    progress = max(0.0, (step - 1) / max(1, planned_steps - 1))
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def run_grpo_v2(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tok = PrimeWordleTokenizer(max_symbols=32)

    ckpt = torch.load(args.init_from, map_location=device, weights_only=True)
    cfg = PrimeGPTConfig.from_dict(ckpt["model_config"])
    model = PrimeGPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    ref_model = PrimeGPT(cfg).to(device=device, dtype=torch.bfloat16)
    ref_model.load_state_dict(ckpt["model_state"])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=(device.type == "cuda"),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "grpo_v2_log.jsonl"

    rng = random.Random(args.seed)
    step = 0
    best_reward = -1e9
    step_times: list[float] = []
    history_logs: list[dict[str, Any]] = []

    print(
        f"=== PrimeGPT Generalist GRPO v2 on {device} ({model.num_parameters()/1e6:.2f}M params) ===\n"
        f"Init/Ref: {args.init_from} | Cosine LR: {args.lr:.1e} -> {args.min_lr:.1e} (over {args.planned_steps} steps)\n"
        f"kl_coef={args.kl_coef} | wrong_pen={args.wrong_penalty} | clip={args.clip_eps} | "
        f"prompts/step={args.prompts_per_step} x G={args.group_size} (= {args.prompts_per_step*args.group_size} rollouts/step)\n"
        f"max_steps={args.max_steps} | max_minutes={args.max_minutes} | milestone_every={args.milestone_every}",
        flush=True,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    _set_windows_keep_awake(True)
    t_run_start = time.perf_counter()

    try:
        while True:
            if args.max_steps > 0 and step >= args.max_steps:
                break
            if args.max_minutes > 0 and ((time.perf_counter() - t_run_start) / 60.0) >= args.max_minutes:
                break

            step += 1
            cur_lr = _cosine_lr(step, args.planned_steps, args.lr, args.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            t_step = time.perf_counter()
            t_roll = t_step
            all_episodes: list[RolloutEpisode] = []
            cat_counter: Counter[str] = Counter()

            for p_i in range(args.prompts_per_step):
                cat = sample_generalist_category(rng)
                p_seed = args.seed * 100_000 + step * 100 + p_i
                spec = build_generalist_prompt(cat, seed=p_seed)
                cat_counter[spec.category] += 1
                group_eps = rollout_generalist_group(
                    model=model,
                    spec=spec,
                    prompt_idx=p_i,
                    group_size=args.group_size,
                    temperature=args.temperature,
                    device=device,
                    tokenizer=tok,
                    wrong_penalty=args.wrong_penalty,
                )
                all_episodes.extend(group_eps)
            roll_sec = time.perf_counter() - t_roll

            t_upd = time.perf_counter()
            advantages = compute_group_advantages(all_episodes, clip_adv=3.0)
            upd = grpo_update_step(
                model=model,
                ref_model=ref_model,
                optimizer=optimizer,
                episodes=all_episodes,
                advantages=advantages,
                device=device,
                kl_coef=args.kl_coef,
                clip_eps=args.clip_eps,
                ppo_epochs=args.ppo_epochs,
                mini_batch_size=args.mini_batch_size,
                grad_clip=args.grad_clip,
                ckpt_threshold=args.ckpt_threshold,
            )
            upd_sec = time.perf_counter() - t_upd
            step_sec = time.perf_counter() - t_step
            step_times.append(step_sec)

            n_eps = len(all_episodes)
            avg_rew = sum(e.reward for e in all_episodes) / n_eps
            win_r = sum(int(e.won) for e in all_episodes) / n_eps
            avg_t = sum(e.added_turns for e in all_episodes) / n_eps
            wm_o = sum(e.wrong_mine_opps for e in all_episodes)
            ed_o = sum(e.exact_drop_opps for e in all_episodes)
            rm_o = sum(e.repeat_mine_opps for e in all_episodes)
            wm = sum(e.wrong_mine_hits for e in all_episodes) / max(1, wm_o) if wm_o > 0 else 0.0
            ed = sum(e.exact_drop_hits for e in all_episodes) / max(1, ed_o) if ed_o > 0 else 0.0
            rm = sum(e.repeat_mine_hits for e in all_episodes) / max(1, rm_o) if rm_o > 0 else 0.0
            vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0

            rec = {
                "step": step,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "lr": round(cur_lr, 8),
                "episodes": n_eps,
                "active_episodes": int(upd["active_episodes"]),
                "avg_reward": round(avg_rew, 4),
                "win_rate": round(win_r, 4),
                "avg_turns": round(avg_t, 2),
                "wrong_mine": round(wm, 4),
                "exact_drop": round(ed, 4),
                "repeat_mine": round(rm, 4),
                "kl": round(upd["kl"], 5),
                "pg_loss": round(upd["pg_loss"], 5),
                "clip_frac": round(upd["clip_frac"], 4),
                "max_ratio_err_e0": round(upd["max_ratio_err_e0"], 6),
                "roll_sec": round(roll_sec, 2),
                "upd_sec": round(upd_sec, 2),
                "step_sec": round(step_sec, 2),
                "vram_mb": round(vram_mb, 1),
                "categories": dict(cat_counter),
            }
            history_logs.append(rec)
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(
                f"step {step:3d} | lr {cur_lr:.2e} | rew {avg_rew:+6.3f} | win {win_r*100:5.1f}% ({avg_t:4.1f}t) | "
                f"w={wm*100:4.1f}% ex={ed*100:4.1f}% rep={rm*100:4.1f}% | "
                f"kl={upd['kl']:.4f} clip={upd['clip_frac']*100:4.1f}% | "
                f"roll={roll_sec:4.1f}s upd={upd_sec:3.1f}s | VRAM {vram_mb:5.0f}MB",
                flush=True,
            )

            if step % args.save_every == 0:
                save_grpo_checkpoint(out_dir / "latest.pt", model, optimizer, step, rec)
                if avg_rew > best_reward:
                    best_reward = avg_rew
                    save_grpo_checkpoint(out_dir / "best.pt", model, optimizer, step, rec)
            if args.milestone_every > 0 and step % args.milestone_every == 0:
                save_grpo_checkpoint(out_dir / f"step_{step:04d}.pt", model, optimizer, step, rec)

        if history_logs:
            save_grpo_checkpoint(out_dir / "latest.pt", model, optimizer, step, history_logs[-1])
    finally:
        _set_windows_keep_awake(False)

    total_sec = time.perf_counter() - t_run_start
    avg_step_s = sum(step_times) / max(1, len(step_times))
    summary = {
        "steps_run": len(step_times),
        "final_step": step,
        "elapsed_sec": round(total_sec, 1),
        "avg_step_sec": round(avg_step_s, 2),
        "est_steps_in_50m": int(3000 / max(1.0, avg_step_s)),
        "peak_vram_mb": round(
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0, 1
        ),
        "last_step": history_logs[-1] if history_logs else None,
        "out_dir": str(out_dir),
    }
    print("\n=== GRPO v2 Session Summary ===", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalist GRPO v2 Trainer for DeepLogic PrimeGPT.")
    parser.add_argument("--init-from", type=Path, default=Path("checkpoints/deeplogic_adv100m/latest.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/deeplogic_grpo_v2"))
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.80)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--wrong-penalty", type=float, default=3.0)
    parser.add_argument("--clip-eps", type=float, default=0.20)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--mini-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--min-lr", type=float, default=3.0e-6)
    parser.add_argument("--planned-steps", type=int, default=80)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ckpt-threshold", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-minutes", type=float, default=50.0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--milestone-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7070)
    args = parser.parse_args()
    run_grpo_v2(args)


if __name__ == "__main__":
    main()
