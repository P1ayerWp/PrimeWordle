"""Phase 4 Online GRPO (Group Relative Policy Optimization) Trainer for PrimeGPT.

Implements all 4 stability & correctness guardrails:
  1. AntiRep=OFF during rollout (with temperature exploration T=0.75..0.85) so the
     model genuinely triggers `repeat_mine` and gets penalized to cure greedy loops.
  2. Frozen `ref_model` (~48 MB bf16) + Schulman k3 non-negative KL penalty
     (`kl_coef ~ 0.03`) + PPO clip (`0.2`) + low LR (`1.5e-5`) to protect 0.02% syntax
     and 10/10 English Wordle / K4 / K8 circuits.
  3. Exact per-token `allowed_masks` alignment across `old_log_prob`, `ref_log_prob`,
     and `new_log_prob` (header, env feedbacks, and dirty prefixes strictly `mask=0`),
     plus progress-floor reward (`final_exact_ratio`) on timeouts to prevent zero-variance groups.
  4. Fast targeted problem pool (K3_P8_L5, True PvZ 10x10 U_OFF/U_ON 50%-forbid fresh+dirty,
     K2, K4, and K1 anchor) using batched KV-Cache rollouts (B = prompts_in_group * G).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from engine.game import GameConfig, GameSession
from engine.primes import encode_slot
from engine.tokenizer import PrimeWordleTokenizer
from model import PrimeGPT, PrimeGPTConfig
from solver.adversarial_dataset import _play_messy_human_prefix
from solver.constraints import BoardKnowledge, symbols_to_mask
from train.stress_test import _slot_legal_extensions, make_forbidden_fusion_table


def _set_windows_keep_awake(enable: bool) -> None:
    if sys.platform != "win32":
        return
    try:
        flags = (0x80000000 | 0x00000001) if enable else 0x80000000
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


@dataclass
class GRPOPromptSpec:
    category: str
    config: GameConfig
    target_slots: list[list[int]]
    prefix_history_guesses: list[list[list[int]]]  # Dirty human prefix turns (already played)
    allowed_pairs: dict[int, set[int]] | None
    max_new_turns: int
    seed: int


@dataclass
class RolloutEpisode:
    category: str
    prompt_idx: int
    input_ids: list[int]          # Full token sequence (header + prefix + model turns + fb [+ EOS])
    action_mask: list[int]        # 1 ONLY on model-generated decision tokens (<TURN>...<FB>), 0 elsewhere
    allowed_masks: list[list[bool]]  # Per-position legal token mask of length vocab_size (aligned to input_ids)
    reward: float
    won: bool
    added_turns: int
    max_turns: int
    final_exact_ratio: float
    wrong_mine_hits: int
    wrong_mine_opps: int
    exact_drop_hits: int
    exact_drop_opps: int
    repeat_mine_hits: int
    repeat_mine_opps: int


def sample_grpo_prompt_category(rng: random.Random) -> str:
    """Focused fast problem pool targeting K3 & PvZ 10x10 while anchoring K1/K2/K4."""
    u = rng.random()
    if u < 0.30:
        return "K3_P8_L5_U"
    if u < 0.55:
        return "PvZ_10x10_UOFF"
    if u < 0.70:
        return "PvZ_10x10_UON"
    if u < 0.85:
        return "K2_P5_L5_U"
    if u < 0.95:
        return "K4_P16_L6_U"
    return "K1_P6_L5"


def build_prompt_spec(category: str, seed: int) -> GRPOPromptSpec:
    rng = random.Random(seed)
    messy_rng = random.Random(seed * 7 + 999_983)

    allowed_pairs: dict[int, set[int]] | None = None
    legal_slots: list[list[int]] | None = None
    target_slots: list[list[int]] | None = None
    dirty_prob = 0.0
    dirty_range = (1, 4)

    if category == "K3_P8_L5_U":
        cfg = GameConfig(num_primes=8, max_multiplier=3, code_length=5, allow_underfill=True)
        max_new = 18
        dirty_prob = 0.30
        dirty_range = (1, 4)
    elif category == "PvZ_10x10_UOFF":
        cfg = GameConfig(num_primes=10, max_multiplier=2, code_length=10, allow_underfill=False)
        forbid_r = rng.uniform(0.35, 0.65)
        self_r = rng.uniform(0.35, 0.90)
        allowed_pairs, legal_slots = make_forbidden_fusion_table(
            10, forbid_ratio=forbid_r, allow_underfill=False, seed=seed + 17, forbid_self_ratio=self_r
        )
        target_slots = [list(rng.choice(legal_slots)) for _ in range(10)]
        max_new = 16
        dirty_prob = 0.50
        dirty_range = (2, 5)
    elif category == "PvZ_10x10_UON":
        cfg = GameConfig(num_primes=10, max_multiplier=2, code_length=10, allow_underfill=True)
        forbid_r = rng.uniform(0.35, 0.65)
        self_r = rng.uniform(0.35, 0.90)
        allowed_pairs, legal_slots = make_forbidden_fusion_table(
            10, forbid_ratio=forbid_r, allow_underfill=True, seed=seed + 29, forbid_self_ratio=self_r
        )
        target_slots = [list(rng.choice(legal_slots)) for _ in range(10)]
        max_new = 16
        dirty_prob = 0.50
        dirty_range = (2, 5)
    elif category == "K2_P5_L5_U":
        cfg = GameConfig(num_primes=5, max_multiplier=2, code_length=5, allow_underfill=True)
        max_new = 12
        dirty_prob = 0.45
        dirty_range = (2, 5)
    elif category == "K4_P16_L6_U":
        cfg = GameConfig(num_primes=16, max_multiplier=4, code_length=6, allow_underfill=True)
        max_new = 20
        dirty_prob = 0.25
        dirty_range = (1, 4)
    elif category == "K1_P6_L5":
        cfg = GameConfig(num_primes=6, max_multiplier=1, code_length=5, allow_underfill=False)
        max_new = 10
        dirty_prob = 0.20
        dirty_range = (1, 3)
    else:
        raise ValueError(f"Unknown GRPO category: {category}")

    session = GameSession(cfg, seed=seed, target_slots=target_slots)
    prefix_guesses: list[list[list[int]]] = []
    if rng.random() < dirty_prob:
        n_m = rng.randint(dirty_range[0], dirty_range[1])
        _play_messy_human_prefix(session, messy_rng, n_m, legal_slots=legal_slots)
        if session.status == "playing":
            prefix_guesses = [
                [sorted(slot) for slot in rec.guess_slots] for rec in session.history
            ]

    return GRPOPromptSpec(
        category=category,
        config=cfg,
        target_slots=[list(s) for s in session.target_slots],
        prefix_history_guesses=prefix_guesses,
        allowed_pairs=allowed_pairs,
        max_new_turns=max_new,
        seed=seed,
    )


def compute_episode_reward(
    won: bool,
    added_turns: int,
    max_turns: int,
    final_exact_ratio: float,
    wrong_hits: int,
    wrong_opps: int,
    exact_drop_hits: int,
    exact_drop_opps: int,
    repeat_hits: int,
    repeat_opps: int,
) -> float:
    """Verifiable dense + terminal reward function (Guardrail #3).

    - Win: +2.0 + 1.0 * (1 - added_turns / max_turns) in [2.0, 3.0]
    - Timeout progress floor: +0.8 * final_exact_ratio in [0.0, 0.8)
    - Penalties:
        * repeat_mine: -2.5 * rate - 0.05 * min(10, hits)   (directly cures greedy loops!)
        * wrong_mine:  -2.0 * rate
        * exact_drop:  -4.0 * rate - 0.20 * min(5, hits)    (keeps exact_drop locked at 0%!)
    """
    if won:
        r = 2.0 + 1.0 * max(0.0, 1.0 - added_turns / max(1, max_turns))
    else:
        r = 0.8 * final_exact_ratio

    if repeat_opps > 0:
        r -= 2.5 * (repeat_hits / repeat_opps)
    if repeat_hits > 0:
        r -= 0.05 * min(10, repeat_hits)

    if wrong_opps > 0:
        r -= 2.0 * (wrong_hits / wrong_opps)

    if exact_drop_opps > 0:
        r -= 4.0 * (exact_drop_hits / exact_drop_opps)
    if exact_drop_hits > 0:
        r -= 0.20 * min(5, exact_drop_hits)

    return float(r)


@torch.no_grad()
def rollout_group_for_prompt(
    model: PrimeGPT,
    prompt_spec: GRPOPromptSpec,
    prompt_idx: int,
    group_size: int,
    temperature: float,
    device: torch.device,
    tokenizer: PrimeWordleTokenizer,
) -> list[RolloutEpisode]:
    """Generate `group_size` parallel rollouts for ONE prompt via batched KV-Cache.

    STRICTLY AntiRep=OFF (Guardrail #1): never masks out tried products so the
    model authentically hits `repeat_mine` when inclined and learns to eliminate it.
    Records exact per-position `allowed_masks` for Guardrail #3 log-prob alignment.
    """
    was_training = model.training
    model.eval()

    cfg = prompt_spec.config
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_id
    turn_id = tokenizer.stoi["<TURN>"]
    slot_id = tokenizer.stoi["<SLOT>"]
    fb_id = tokenizer.stoi["<FB>"]
    sym_ids = [tokenizer.stoi[f"S_{i}"] for i in range(cfg.num_primes)]
    sym_id_set = set(sym_ids)

    # Reconstruct identical starting session state for all G rollouts
    sessions = [
        GameSession(cfg, seed=prompt_spec.seed, target_slots=prompt_spec.target_slots)
        for _ in range(group_size)
    ]
    for g_slots in prompt_spec.prefix_history_guesses:
        for s in sessions:
            s.submit_guess(g_slots)

    # Encode shared prompt (header + any dirty prefix turns) — ALL action_mask = 0!
    prompt_ids = tokenizer.encode_config(cfg)
    for turn in sessions[0].history:
        prompt_ids.append(turn_id)
        for slot in turn.guess_slots:
            prompt_ids.append(slot_id)
            for sym in sorted(slot):
                prompt_ids.append(sym_ids[sym])
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

    prompt_t = torch.tensor([prompt_ids] * group_size, dtype=torch.long, device=device)
    cur_pos = [prefix_len] * group_size
    pos_init = torch.arange(prefix_len, dtype=torch.long, device=device)[None, :].expand(
        group_size, -1
    )
    valid_mask = torch.ones((group_size, prefix_len), dtype=torch.bool, device=device)

    max_gen_per_turn = 2 + cfg.code_length * (cfg.max_multiplier + 1)
    inv_temp = 1.0 / max(1e-3, temperature)

    with torch.amp.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
    ):
        logits, kv, _ = model(
            prompt_t, use_cache=True, position_ids=pos_init, key_padding_mask=valid_mask
        )

        for _turn_i in range(prompt_spec.max_new_turns):
            active = [g for g in range(group_size) if sessions[g].status == "playing"]
            if not active:
                break

            gen_slots_ids: list[list[int]] = [[] for _ in range(group_size)]
            slots_started = [0] * group_size
            cur_slot_syms: list[list[int]] = [[] for _ in range(group_size)]
            turn_done = [sessions[g].status != "playing" for g in range(group_size)]

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
                    else:
                        # STRICTLY AntiRep=OFF (forbidden_slot_prods=None) per Guardrail #1!
                        next_syms, can_close = _slot_legal_extensions(
                            cur_slot_syms[g],
                            cfg,
                            sessions[g].primes,
                            prompt_spec.allowed_pairs,
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
                        cur_slot_syms[g].append(chosen - sym_ids[0])

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

            # Audit mine-stepping BEFORE submit, then submit and feed L feedback tokens
            L = cfg.code_length
            fb_rows = [[pad_id] * L for _ in range(group_size)]
            fb_pos_rows = [[cur_pos[g]] * L for _ in range(group_size)]
            fb_valid_rows = [[False] * L for _ in range(group_size)]

            for g in active:
                # Parse slots directly from gen_slots_ids[g]
                slots: list[list[int]] = []
                cur: list[int] | None = None
                for tid in gen_slots_ids[g][1:-1]:
                    if tid == slot_id:
                        if cur is not None:
                            slots.append(cur)
                        cur = []
                    elif tid in sym_id_set and cur is not None:
                        cur.append(tid - sym_ids[0])
                if cur is not None:
                    slots.append(cur)

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
                fb_pos_rows[g] = [cur_pos[g] + k for k in range(L)]
                fb_valid_rows[g] = [True] * L
                cur_pos[g] += L

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
        won = sess.status == "won"
        if won:
            final_exact_ratio = 1.0
        else:
            k_final = BoardKnowledge(sess)
            final_exact_ratio = len(k_final.confirmed_exact) / max(1, cfg.code_length)

        rew = compute_episode_reward(
            won=won,
            added_turns=added_turns[g],
            max_turns=prompt_spec.max_new_turns,
            final_exact_ratio=final_exact_ratio,
            wrong_hits=wrong_h[g],
            wrong_opps=wrong_o[g],
            exact_drop_hits=exact_h[g],
            exact_drop_opps=exact_o[g],
            repeat_hits=rep_h[g],
            repeat_opps=rep_o[g],
        )
        episodes.append(
            RolloutEpisode(
                category=prompt_spec.category,
                prompt_idx=prompt_idx,
                input_ids=ep_ids[g],
                action_mask=ep_action_mask[g],
                allowed_masks=ep_allowed_masks[g],
                reward=rew,
                won=won,
                added_turns=added_turns[g],
                max_turns=prompt_spec.max_new_turns,
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


def collate_rollout_batch(
    episodes: Sequence[RolloutEpisode],
    vocab_size: int = 80,
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate episodes into (input_ids [B, T], action_mask [B, T], allowed_masks [B, T, V])."""
    bsz = len(episodes)
    raw_max = max(len(ep.input_ids) for ep in episodes)
    max_len = ((raw_max + 7) // 8) * 8

    ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    act = torch.zeros((bsz, max_len), dtype=torch.bool)
    allowed = torch.ones((bsz, max_len, vocab_size), dtype=torch.bool)

    for i, ep in enumerate(episodes):
        n = len(ep.input_ids)
        ids[i, :n] = torch.tensor(ep.input_ids, dtype=torch.long)
        act[i, :n] = torch.tensor(ep.action_mask, dtype=torch.bool)
        allowed[i, :n, :] = torch.tensor(ep.allowed_masks, dtype=torch.bool)

    return ids, act, allowed


def compute_masked_action_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    action_mask: torch.Tensor,
    allowed_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact per-token log pi(a_t | s_t) over the legal masked action space.

    Shift alignment:
      - `logits[:, :-1, :]` predicts `target_ids = input_ids[:, 1:]`
      - Legal mask for `target_ids` is `target_allowed = allowed_masks[:, 1:, :]`
      - Active decision positions are `target_act = action_mask[:, 1:]`

    Returns:
      selected_logp: (B, T-1) float32 log probabilities (0.0 on non-action positions)
      target_act:    (B, T-1) bool mask of active decision positions
    """
    shift_logits = logits[:, :-1, :].float()
    target_ids = input_ids[:, 1:]
    target_act = action_mask[:, 1:]
    target_allowed = allowed_masks[:, 1:, :]

    masked_logits = shift_logits.masked_fill(~target_allowed, -1e9)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    selected_logp = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    selected_logp = selected_logp * target_act.float()
    return selected_logp, target_act


def compute_group_advantages(
    episodes: Sequence[RolloutEpisode], clip_adv: float = 3.0
) -> list[float]:
    """Compute within-prompt normalized GRPO advantages (0.0 for zero-variance groups)."""
    by_prompt: dict[int, list[int]] = {}
    for idx, ep in enumerate(episodes):
        by_prompt.setdefault(ep.prompt_idx, []).append(idx)

    advantages = [0.0] * len(episodes)
    for idxs in by_prompt.values():
        rews = [episodes[i].reward for i in idxs]
        mean_r = sum(rews) / len(rews)
        var_r = sum((r - mean_r) ** 2 for r in rews) / len(rews)
        std_r = math.sqrt(var_r)
        if std_r < 1e-6:
            for i in idxs:
                advantages[i] = 0.0
        else:
            for i in idxs:
                a = (episodes[i].reward - mean_r) / (std_r + 1e-6)
                advantages[i] = max(-clip_adv, min(clip_adv, a))
    return advantages


def grpo_update_step(
    model: PrimeGPT,
    ref_model: PrimeGPT,
    optimizer: torch.optim.Optimizer,
    episodes: Sequence[RolloutEpisode],
    advantages: Sequence[float],
    device: torch.device,
    kl_coef: float = 0.03,
    clip_eps: float = 0.2,
    ppo_epochs: int = 2,
    mini_batch_size: int = 8,
    grad_clip: float = 1.0,
    ckpt_threshold: int = 1024,
) -> dict[str, float]:
    """Perform `ppo_epochs` of clipped GRPO + Schulman k3 KL update on collected rollouts."""
    vocab_size = model.cfg.vocab_size
    n_ep = len(episodes)

    # Filter out zero-advantage episodes (from zero-variance groups) cleanly
    active_indices = [i for i in range(n_ep) if abs(advantages[i]) > 1e-7]
    if not active_indices:
        return {
            "active_episodes": 0.0,
            "pg_loss": 0.0,
            "kl": 0.0,
            "clip_frac": 0.0,
            "max_ratio_err_e0": 0.0,
        }

    act_eps = [episodes[i] for i in active_indices]
    act_adv = torch.tensor([advantages[i] for i in active_indices], dtype=torch.float32, device=device)
    n_act = len(act_eps)

    # Precompute frozen `old_logp` (from current policy before update) and `ref_logp` in mini-batches
    model.eval()
    cached_batches: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for st in range(0, n_act, mini_batch_size):
            sub_eps = act_eps[st : st + mini_batch_size]
            sub_adv = act_adv[st : st + mini_batch_size]
            ids, act_mask, allowed = collate_rollout_batch(sub_eps, vocab_size=vocab_size)
            ids = ids.to(device)
            act_mask = act_mask.to(device)
            allowed = allowed.to(device)

            with torch.amp.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
            ):
                old_logits, _, _ = model(ids)
                ref_logits, _, _ = ref_model(ids)

            old_logp, target_act = compute_masked_action_log_probs(
                old_logits, ids, act_mask, allowed
            )
            ref_logp, _ = compute_masked_action_log_probs(
                ref_logits, ids, act_mask, allowed
            )
            cached_batches.append(
                {
                    "ids": ids,
                    "act_mask": act_mask,
                    "allowed": allowed,
                    "target_act": target_act,
                    "old_logp": old_logp.detach(),
                    "ref_logp": ref_logp.detach(),
                    "adv": sub_adv,
                }
            )

    # Total action tokens across active episodes (for exact normalized gradient accumulation)
    total_act_tokens = sum(int(b["target_act"].sum().item()) for b in cached_batches)
    if total_act_tokens == 0:
        return {
            "active_episodes": 0.0,
            "pg_loss": 0.0,
            "kl": 0.0,
            "clip_frac": 0.0,
            "max_ratio_err_e0": 0.0,
        }

    model.train()
    total_pg = 0.0
    total_kl = 0.0
    total_clipped = 0.0
    total_tok_counted = 0
    max_ratio_err_e0 = 0.0

    for ep_i in range(ppo_epochs):
        optimizer.zero_grad(set_to_none=True)
        for batch in cached_batches:
            ids = batch["ids"]
            act_mask = batch["act_mask"]
            allowed = batch["allowed"]
            target_act = batch["target_act"]
            old_logp = batch["old_logp"]
            ref_logp = batch["ref_logp"]
            sub_adv = batch["adv"][:, None]  # (B, 1)

            model.gradient_checkpointing = ids.size(1) > ckpt_threshold
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
            ):
                new_logits, _, _ = model(ids)

            new_logp, _ = compute_masked_action_log_probs(new_logits, ids, act_mask, allowed)
            mask_f = target_act.float()

            log_ratio = (new_logp - old_logp).clamp(-10.0, 10.0)
            ratio = torch.exp(log_ratio)
            if ep_i == 0:
                err = ((ratio - 1.0).abs() * mask_f).max().item()
                if err > max_ratio_err_e0:
                    max_ratio_err_e0 = err

            surr1 = ratio * sub_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * sub_adv
            pg_obj = torch.minimum(surr1, surr2)

            # Schulman k3 unbiased non-negative KL estimator: exp(ref - new) - (ref - new) - 1
            kl_delta = (ref_logp - new_logp).clamp(-10.0, 10.0)
            kl_per_tok = torch.exp(kl_delta) - kl_delta - 1.0

            loss_per_tok = (-pg_obj + kl_coef * kl_per_tok) * mask_f
            loss = loss_per_tok.sum() / float(total_act_tokens)
            loss.backward()

            with torch.no_grad():
                n_t = int(target_act.sum().item())
                total_pg += float((-pg_obj * mask_f).sum().item())
                total_kl += float((kl_per_tok * mask_f).sum().item())
                clipped = ((ratio - 1.0).abs() > clip_eps) & target_act
                total_clipped += float(clipped.sum().item())
                total_tok_counted += n_t

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    denom = max(1, total_tok_counted)
    return {
        "active_episodes": float(n_act),
        "pg_loss": total_pg / denom,
        "kl": total_kl / denom,
        "clip_frac": total_clipped / denom,
        "max_ratio_err_e0": max_ratio_err_e0,
    }


def save_grpo_checkpoint(
    path: Path,
    model: PrimeGPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model_config": model.cfg.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "metrics": metrics,
        },
        tmp,
    )
    tmp.replace(path)


def run_grpo(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tok = PrimeWordleTokenizer(max_symbols=32)

    # Load policy model + frozen bf16 reference model (~48 MB)
    ckpt = torch.load(args.init_from, map_location=device, weights_only=True)
    cfg = PrimeGPTConfig.from_dict(ckpt["model_config"])
    model = PrimeGPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    ref_model = PrimeGPT(cfg).to(device=device, dtype=torch.bfloat16)
    ref_model.load_state_dict(ckpt["model_state"])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    use_fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=use_fused,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "grpo_log.jsonl"

    rng = random.Random(args.seed)
    start_step = 0
    best_reward = -1e9

    if args.resume is not None and Path(args.resume).exists():
        r_ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(r_ckpt["model_state"])
        optimizer.load_state_dict(r_ckpt["optimizer_state"])
        start_step = int(r_ckpt.get("step", 0))
        print(f"Resumed GRPO policy from {args.resume} (step={start_step})", flush=True)

    print(
        f"=== PrimeGPT Phase 4 GRPO on {device} ({model.num_parameters()/1e6:.2f}M params) ===\n"
        f"Init/Ref: {args.init_from} | lr={args.lr:.1e} | kl_coef={args.kl_coef} | clip={args.clip_eps}\n"
        f"prompts/step={args.prompts_per_step} x G={args.group_size} (= {args.prompts_per_step*args.group_size} rollouts/step) | "
        f"temp={args.temperature} | ppo_epochs={args.ppo_epochs} | max_steps={args.max_steps} | max_minutes={args.max_minutes}",
        flush=True,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    _set_windows_keep_awake(True)
    t_run_start = time.perf_counter()
    step = start_step
    step_times: list[float] = []
    history_logs: list[dict[str, Any]] = []

    try:
        while True:
            if args.max_steps > 0 and (step - start_step) >= args.max_steps:
                break
            elapsed_min = (time.perf_counter() - t_run_start) / 60.0
            if args.max_minutes > 0 and elapsed_min >= args.max_minutes:
                break

            step += 1
            t_step = time.perf_counter()

            # 1. Online Rollout Phase (AntiRep=OFF!)
            t_roll = time.perf_counter()
            all_episodes: list[RolloutEpisode] = []
            cat_counter: Counter[str] = Counter()

            for p_i in range(args.prompts_per_step):
                cat = sample_grpo_prompt_category(rng)
                cat_counter[cat] += 1
                p_seed = args.seed * 100_000 + step * 100 + p_i
                spec = build_prompt_spec(cat, seed=p_seed)
                group_eps = rollout_group_for_prompt(
                    model,
                    spec,
                    prompt_idx=p_i,
                    group_size=args.group_size,
                    temperature=args.temperature,
                    device=device,
                    tokenizer=tok,
                )
                all_episodes.extend(group_eps)
            roll_sec = time.perf_counter() - t_roll

            # 2. GRPO Advantage + Clipped Policy + KL Update Phase
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

            # Aggregate step diagnostics
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
                f"step {step:4d} | rew {avg_rew:+6.3f} | win {win_r*100:5.1f}% ({avg_t:4.1f}t) | "
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

        # Always save final latest.pt
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
        "est_steps_in_20m": int(1200 / max(1.0, avg_step_s)),
        "peak_vram_mb": round(
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0, 1
        ),
        "last_step": history_logs[-1] if history_logs else None,
        "out_dir": str(out_dir),
    }
    print("\n=== GRPO Session Summary ===", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 Online GRPO RL for DeepLogic PrimeGPT.")
    parser.add_argument("--init-from", type=Path, default=Path("checkpoints/deeplogic_adv100m/latest.pt"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/deeplogic_grpo"))
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.80)
    parser.add_argument("--kl-coef", type=float, default=0.03)
    parser.add_argument("--clip-eps", type=float, default=0.20)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--mini-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ckpt-threshold", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-minutes", type=float, default=20.0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=4040)
    args = parser.parse_args()
    run_grpo(args)


if __name__ == "__main__":
    main()
