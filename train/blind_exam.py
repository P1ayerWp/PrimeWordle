"""Independent 310-Game Generalist Blind Exam (`blind_exam.py`) for PrimeGPT.

Evaluates any checkpoint (or multiple checkpoints side-by-side) on 310 strictly
unseen, out-of-distribution games using fast batched KV-Cache inference:
  1. 120 Random Unseen (P, K, L) Generalist Games (strictly AntiRep=OFF bare!):
       - 30 OOD_Small      (random P, K=1..2, L; excludes fixed stress_test shapes)
       - 30 OOD_Mid_K34    (random P, K=3..4, L; excludes fixed stress_test shapes)
       - 30 OOD_PvZ_Sparse (random P=6..16, K=2, L=5..16, 30..70% forbidden pairs + p*p)
       - 30 OOD_Hard_K58   (random P=10..32, K=5..8, L=3..16, mix fresh & mid-game slice)
  2. 40 <UNK_CFG> Hidden-Rule Blind Games (truly raw slot-length decoding, zero leakage!):
       - 20 UNK_ZeroPrefix (UNDERFILL_ON, K & DUP hidden as <UNK_CFG>, no prefix)
       - 20 UNK_WithPrefix (K, UNDERFILL & DUP all hidden as <UNK_CFG>, 2-3 prefix turns)
  3. 150 Real Large-Dictionary Zero-Shot English Wordle Games (batched Trie Beam Search):
       - 100 x 5-letter words (2,587-word clean dictionary)
       -  25 x 6-letter words (2,963-word clean dictionary)
       -  25 x 7-letter words (2,977-word clean dictionary)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from engine.game import GameConfig, GameSession
from engine.primes import encode_slot
from engine.tokenizer import PrimeWordleTokenizer
from model import PrimeGPT, PrimeGPTConfig
from solver.adversarial_dataset import _play_messy_human_prefix
from solver.constraints import BoardKnowledge, count_slot_candidates, symbols_to_mask
from solver.hybrid import HybridSolver
from train.stress_test import _slot_legal_extensions, make_forbidden_fusion_table


@dataclass
class BlindGameSpec:
    bucket: str
    config: GameConfig
    target_slots: list[list[int]]
    prefix_guesses: list[list[list[int]]]
    allowed_pairs: dict[int, set[int]] | None
    max_new_turns: int
    reveal_k: bool = True
    reveal_underfill: bool = True
    reveal_dup: bool = True
    raw_unk_length: bool = False  # If True, do NOT leak true K/underfill via syntax mask!
    seed: int = 0


def _build_120_ood_specs(seed_base: int = 555_000_000) -> list[BlindGameSpec]:
    """Generate 120 deterministic unseen OOD (P, K, L) specs avoiding fixed stress_test shapes."""
    excluded_exact = {(6, 1, 5), (5, 2, 5), (8, 3, 5), (16, 4, 6), (10, 2, 10)}
    rng = random.Random(seed_base)
    specs: list[BlindGameSpec] = []

    def _draw_non_excluded(p_lo: int, p_hi: int, k_lo: int, k_hi: int, l_lo: int, l_hi: int) -> tuple[int, int, int]:
        for _ in range(100):
            p = rng.randint(p_lo, p_hi)
            k = rng.randint(k_lo, k_hi)
            l = rng.randint(l_lo, l_hi)
            if (p, k, l) not in excluded_exact:
                return p, k, l
        return p_lo, k_lo, l_lo

    # 1. 30 OOD_Small (P in 3..14, K in 1..2, L in 3..10)
    for i in range(30):
        g_seed = seed_base + 1_000 + i
        m_rng = random.Random(g_seed * 13 + 7)
        p, k, l = _draw_non_excluded(3, 14, 1, 2, 3, 10)
        allow_u = False if k == 1 else (rng.random() < 0.60)
        probe_cfg = GameConfig(p, k, l, allow_underfill=allow_u, allow_duplicate_slots=True)
        allow_dup = not (l <= count_slot_candidates(probe_cfg) and rng.random() < 0.35)
        cfg = GameConfig(p, k, l, allow_underfill=allow_u, allow_duplicate_slots=allow_dup)
        sess = GameSession(cfg, seed=g_seed)
        prefix: list[list[list[int]]] = []
        if rng.random() < 0.50:
            _play_messy_human_prefix(sess, m_rng, rng.randint(1, 5))
            if sess.status == "playing":
                prefix = [[sorted(s) for s in r.guess_slots] for r in sess.history]
        specs.append(
            BlindGameSpec(
                bucket="OOD_Small",
                config=cfg,
                target_slots=[list(s) for s in sess.target_slots],
                prefix_guesses=prefix,
                allowed_pairs=None,
                max_new_turns=16,
                seed=g_seed,
            )
        )

    # 2. 30 OOD_Mid_K34 (P in 5..24, K in 3..4, L in 3..10)
    for i in range(30):
        g_seed = seed_base + 2_000 + i
        m_rng = random.Random(g_seed * 13 + 11)
        p, k, l = _draw_non_excluded(5, 24, 3, 4, 3, 10)
        allow_u = rng.random() < 0.65
        allow_dup = rng.random() < 0.70
        cfg = GameConfig(p, k, l, allow_underfill=allow_u, allow_duplicate_slots=allow_dup)
        sess = GameSession(cfg, seed=g_seed)
        prefix = []
        if rng.random() < 0.45:
            _play_messy_human_prefix(sess, m_rng, rng.randint(1, 6))
            if sess.status == "playing":
                prefix = [[sorted(s) for s in r.guess_slots] for r in sess.history]
        specs.append(
            BlindGameSpec(
                bucket="OOD_Mid_K34",
                config=cfg,
                target_slots=[list(s) for s in sess.target_slots],
                prefix_guesses=prefix,
                allowed_pairs=None,
                max_new_turns=24,
                seed=g_seed,
            )
        )

    # 3. 30 OOD_PvZ_Sparse (random P=6..16, K=2, L=5..16, 30..70% forbidden + p*p bans)
    for i in range(30):
        g_seed = seed_base + 3_000 + i
        m_rng = random.Random(g_seed * 13 + 19)
        p, k, l = _draw_non_excluded(6, 16, 2, 2, 5, 16)
        allow_u = rng.random() < 0.40  # 60% UNDERFILL_OFF
        forbid_r = rng.uniform(0.30, 0.70)
        self_r = rng.uniform(0.30, 1.00)
        allowed_pairs, legal_slots = make_forbidden_fusion_table(
            p, forbid_ratio=forbid_r, allow_underfill=allow_u, seed=g_seed + 77, forbid_self_ratio=self_r
        )
        allow_dup = not (l <= len(legal_slots) and rng.random() < 0.25)
        cfg = GameConfig(p, 2, l, allow_underfill=allow_u, allow_duplicate_slots=allow_dup)
        if allow_dup:
            target = [list(rng.choice(legal_slots)) for _ in range(l)]
        else:
            target = [list(s) for s in rng.sample(legal_slots, l)]
        sess = GameSession(cfg, seed=g_seed, target_slots=target)
        prefix = []
        if rng.random() < 0.50:
            _play_messy_human_prefix(sess, m_rng, rng.randint(2, 6), legal_slots=legal_slots)
            if sess.status == "playing":
                prefix = [[sorted(s) for s in r.guess_slots] for r in sess.history]
        specs.append(
            BlindGameSpec(
                bucket="OOD_PvZ_Sparse",
                config=cfg,
                target_slots=[list(s) for s in sess.target_slots],
                prefix_guesses=prefix,
                allowed_pairs=allowed_pairs,
                max_new_turns=22,
                seed=g_seed,
            )
        )

    # 4. 30 OOD_Hard_K58 (P=10..32, K=5..8, L=3..14; 15 fresh + 15 mid-game endgame slices!)
    for i in range(30):
        g_seed = seed_base + 4_000 + i
        p = rng.randint(10, 32)
        k = rng.randint(5, 8)
        l = rng.randint(3, 12)
        allow_u = rng.random() < 0.70
        cfg = GameConfig(p, k, l, allow_underfill=allow_u, allow_duplicate_slots=True)
        sess = GameSession(cfg, seed=g_seed)
        prefix = []
        if i >= 15:
            # Solver plays ~55% of the way to test deep endgame deduction cleanly
            sim = GameSession(cfg, seed=g_seed, target_slots=[list(s) for s in sess.target_slots])
            solver = HybridSolver(seed=g_seed + 5)
            while sim.status == "playing" and len(sim.history) < 80:
                sim.submit_guess(solver.next_guess(sim))
            cut = max(1, int(len(sim.history) * 0.55))
            if cut < len(sim.history):
                prefix = [[sorted(s) for s in r.guess_slots] for r in sim.history[:cut]]
        specs.append(
            BlindGameSpec(
                bucket="OOD_Hard_K58",
                config=cfg,
                target_slots=[list(s) for s in sess.target_slots],
                prefix_guesses=prefix,
                allowed_pairs=None,
                max_new_turns=32 if i < 15 else 20,
                seed=g_seed,
            )
        )

    return specs


def _build_40_unk_specs(seed_base: int = 556_000_000) -> list[BlindGameSpec]:
    """Generate 40 honest <UNK_CFG> blind-rule games (20 ZeroPrefix + 20 WithPrefix)."""
    rng = random.Random(seed_base)
    specs: list[BlindGameSpec] = []

    # 20 UNK_ZeroPrefix: UNDERFILL_ON, hide K & DUP (model must infer K <= K_true without mask leakage!)
    for i in range(20):
        g_seed = seed_base + 1_000 + i
        p = rng.randint(5, 16)
        k = rng.randint(2, 5)
        l = rng.randint(4, 8)
        allow_dup = rng.random() < 0.70
        cfg = GameConfig(p, k, l, allow_underfill=True, allow_duplicate_slots=allow_dup)
        sess = GameSession(cfg, seed=g_seed)
        specs.append(
            BlindGameSpec(
                bucket="UNK_ZeroPrefix",
                config=cfg,
                target_slots=[list(s) for s in sess.target_slots],
                prefix_guesses=[],
                allowed_pairs=None,
                max_new_turns=22,
                reveal_k=False,
                reveal_underfill=True,
                reveal_dup=False,
                raw_unk_length=True,
                seed=g_seed,
            )
        )

    # 20 UNK_WithPrefix: hide ALL THREE (K, UNDERFILL, DUP), give 2-3 solver prefix turns so rules are inferable!
    for i in range(20):
        g_seed = seed_base + 2_000 + i
        p = rng.randint(5, 16)
        k = rng.randint(1, 4)
        l = rng.randint(4, 8)
        allow_u = False if k == 1 else (rng.random() < 0.55)
        allow_dup = rng.random() < 0.70
        cfg = GameConfig(p, k, l, allow_underfill=allow_u, allow_duplicate_slots=allow_dup)
        sim = GameSession(cfg, seed=g_seed)
        solver = HybridSolver(seed=g_seed + 3)
        n_pre = rng.randint(2, 3)
        for _ in range(n_pre):
            if sim.status != "playing":
                break
            sim.submit_guess(solver.next_guess(sim))
        if sim.status != "playing":
            # Retry with fresh seed if solved in <=3 turns
            sim = GameSession(cfg, seed=g_seed + 50_000)
            sim.submit_guess(solver.next_guess(sim))
            if sim.status == "playing":
                sim.submit_guess(solver.next_guess(sim))
        prefix = [[sorted(s) for s in r.guess_slots] for r in sim.history]
        specs.append(
            BlindGameSpec(
                bucket="UNK_WithPrefix",
                config=cfg,
                target_slots=[list(s) for s in sim.target_slots],
                prefix_guesses=prefix,
                allowed_pairs=None,
                max_new_turns=20,
                reveal_k=False,
                reveal_underfill=False,
                reveal_dup=False,
                raw_unk_length=True,
                seed=g_seed,
            )
        )

    return specs


@torch.no_grad()
def eval_heterogeneous_specs(
    model: PrimeGPT,
    specs: Sequence[BlindGameSpec],
    device: torch.device,
    tokenizer: PrimeWordleTokenizer,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Run arbitrary heterogeneous (P, K, L, forbidden, UNK_CFG) specs in lock-step KV-Cache batches.

    Uses a per-game `pending_fb` queue so games of completely different L and K step
    1 token per GPU call with ZERO cross-game synchronization barriers!
    Strictly AntiRep=OFF (bare model!).
    """
    pad_id = tokenizer.pad_id
    turn_id = tokenizer.stoi["<TURN>"]
    slot_id = tokenizer.stoi["<SLOT>"]
    fb_id = tokenizer.stoi["<FB>"]
    s0_id = tokenizer.stoi["S_0"]
    all_sym_ids = [s0_id + i for i in range(32)]
    all_sym_set = set(all_sym_ids)

    all_results: list[dict[str, Any]] = []

    with torch.amp.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
    ):
        for chunk_start in range(0, len(specs), batch_size):
            chunk = specs[chunk_start : chunk_start + batch_size]
            bsz = len(chunk)

            sessions: list[GameSession] = []
            prompts: list[list[int]] = []
            for sp in chunk:
                sess = GameSession(sp.config, seed=sp.seed, target_slots=sp.target_slots)
                for g_slots in sp.prefix_guesses:
                    if sess.status == "playing":
                        sess.submit_guess(g_slots)
                sessions.append(sess)

                p_ids = tokenizer.encode_config(
                    sp.config,
                    reveal_k=sp.reveal_k,
                    reveal_underfill=sp.reveal_underfill,
                    reveal_dup=sp.reveal_dup,
                )
                for rec in sess.history:
                    p_ids.append(turn_id)
                    for slot in rec.guess_slots:
                        p_ids.append(slot_id)
                        for s in sorted(slot):
                            p_ids.append(s0_id + s)
                    p_ids.append(fb_id)
                    for fb in rec.feedback:
                        p_ids.append(tokenizer.stoi[fb.value])
                prompts.append(p_ids)

            max_pref = max(len(p) for p in prompts)
            pref_tensor = torch.full((bsz, max_pref), pad_id, dtype=torch.long, device=device)
            pref_pos = torch.zeros((bsz, max_pref), dtype=torch.long, device=device)
            valid_mask = torch.zeros((bsz, max_pref), dtype=torch.bool, device=device)
            cur_pos = [0] * bsz

            for b, p_ids in enumerate(prompts):
                n = len(p_ids)
                pref_tensor[b, :n] = torch.tensor(p_ids, dtype=torch.long, device=device)
                pref_pos[b, :n] = torch.arange(n, dtype=torch.long, device=device)
                valid_mask[b, :n] = True
                cur_pos[b] = n

            # Prefill prompt batch; note each game's initial decision logits sit at index len(p_ids)-1!
            logits_pref, kv, _ = model(
                pref_tensor, use_cache=True, position_ids=pref_pos, key_padding_mask=valid_mask
            )
            cur_logits = torch.stack(
                [logits_pref[b, len(prompts[b]) - 1, :] for b in range(bsz)], dim=0
            )  # (B, V)

            # Per-game streaming state machine
            # phase: "turn" -> "slot" -> "in_slot" -> ... -> "fb_queue" -> "turn" / "done"
            pending_fb: list[deque[int]] = [deque() for _ in range(bsz)]
            done = [sessions[b].status != "playing" for b in range(bsz)]
            added_turns = [0] * bsz
            rule_violations = [0] * bsz  # For raw_unk_length illegal length guesses

            in_turn_step = [0] * bsz  # 0=need <TURN>, 1=need first <SLOT>, 2=inside slots
            slots_started = [0] * bsz
            cur_slot_syms: list[list[int]] = [[] for _ in range(bsz)]
            built_slots: list[list[list[int]]] = [[] for _ in range(bsz)]

            wrong_h = [0] * bsz
            wrong_o = [0] * bsz
            exact_h = [0] * bsz
            exact_o = [0] * bsz
            rep_h = [0] * bsz
            rep_o = [0] * bsz

            # Hard upper bound on total single-token steps for this chunk
            max_total_steps = max(
                sp.max_new_turns * (2 + sp.config.code_length * (sp.config.max_multiplier + 2))
                for sp in chunk
            ) + 64

            for _global_step in range(max_total_steps):
                if all(done):
                    break

                logits_f = cur_logits.float()
                next_ids = [pad_id] * bsz
                next_pos = list(cur_pos)
                next_valid = [False] * bsz

                for b in range(bsz):
                    if done[b]:
                        continue
                    sp = chunk[b]
                    cfg = sp.config
                    sess = sessions[b]

                    # If feedback tokens are queued, stream the next feedback token
                    if pending_fb[b]:
                        tid = pending_fb[b].popleft()
                        next_ids[b] = tid
                        next_pos[b] = cur_pos[b]
                        next_valid[b] = True
                        cur_pos[b] += 1
                        if not pending_fb[b]:
                            # Ready for next turn (or finished!)
                            if sess.status != "playing" or added_turns[b] >= sp.max_new_turns:
                                done[b] = True
                            else:
                                in_turn_step[b] = 0
                                slots_started[b] = 0
                                cur_slot_syms[b] = []
                                built_slots[b] = []
                        continue

                    # Model decision step!
                    row = logits_f[b]
                    if in_turn_step[b] == 0:
                        chosen = turn_id
                        in_turn_step[b] = 1
                    elif in_turn_step[b] == 1:
                        chosen = slot_id
                        in_turn_step[b] = 2
                        slots_started[b] = 1
                        cur_slot_syms[b] = []
                    else:
                        if sp.raw_unk_length:
                            # Truly blind to true K and UNDERFILL: allow 1..8 symbols!
                            allowed_tids: list[int] = []
                            cur_len = len(cur_slot_syms[b])
                            if cur_len < 8:
                                start_s = cur_slot_syms[b][-1] if cur_len > 0 else 0
                                allowed_tids.extend(s0_id + s for s in range(start_s, cfg.num_primes))
                            if cur_len >= 1:
                                allowed_tids.append(
                                    slot_id if slots_started[b] < cfg.code_length else fb_id
                                )
                        else:
                            # Standard legal syntax + optional forbidden-pair mask, STRICTLY AntiRep=OFF!
                            next_syms, can_close = _slot_legal_extensions(
                                cur_slot_syms[b],
                                cfg,
                                sess.primes,
                                sp.allowed_pairs,
                                forbidden_slot_prods=None,
                            )
                            allowed_tids = [s0_id + s for s in next_syms]
                            if can_close:
                                allowed_tids.append(
                                    slot_id if slots_started[b] < cfg.code_length else fb_id
                                )

                        chosen = max(allowed_tids, key=lambda idx: float(row[idx].item()))

                        if chosen == slot_id:
                            built_slots[b].append(list(cur_slot_syms[b]))
                            slots_started[b] += 1
                            cur_slot_syms[b] = []
                        elif chosen in all_sym_set:
                            cur_slot_syms[b].append(chosen - s0_id)
                        elif chosen == fb_id:
                            built_slots[b].append(list(cur_slot_syms[b]))
                            guess = built_slots[b]

                            # Validate against engine (matters for raw_unk_length!)
                            valid_len = True
                            for s in guess:
                                if len(s) > cfg.max_multiplier or (
                                    not cfg.allow_underfill and len(s) != cfg.max_multiplier
                                ):
                                    valid_len = False
                                    break

                            if not valid_len:
                                rule_violations[b] += 1
                                done[b] = True
                            else:
                                # Audit mine-stepping BEFORE submitting
                                if sess.history:
                                    k_now = BoardKnowledge(sess)
                                    for i, slot_syms in enumerate(guess):
                                        prod = encode_slot(slot_syms, sess.primes)
                                        mask = symbols_to_mask(slot_syms)
                                        if i in k_now.confirmed_exact:
                                            exact_o[b] += 1
                                            if prod != k_now.solved_products[i]:
                                                exact_h[b] += 1
                                        else:
                                            if k_now.excluded_mask[i] != 0:
                                                wrong_o[b] += 1
                                                if (mask & k_now.excluded_mask[i]) != 0:
                                                    wrong_h[b] += 1
                                            if k_now.tried_at_slot[i]:
                                                rep_o[b] += 1
                                                if prod in k_now.tried_at_slot[i]:
                                                    rep_h[b] += 1

                                rec = sess.submit_guess(guess)
                                added_turns[b] += 1
                                for fb in rec.feedback:
                                    pending_fb[b].append(tokenizer.stoi[fb.value])

                    next_ids[b] = chosen
                    next_pos[b] = cur_pos[b]
                    next_valid[b] = True
                    cur_pos[b] += 1

                if all(done):
                    break

                inp = torch.tensor(next_ids, dtype=torch.long, device=device)[:, None]
                pos_t = torch.tensor(next_pos, dtype=torch.long, device=device)[:, None]
                val_col = torch.tensor(next_valid, dtype=torch.bool, device=device)[:, None]
                valid_mask = torch.cat([valid_mask, val_col], dim=1)

                step_out, kv, _ = model(
                    inp,
                    past_kv=kv,
                    use_cache=True,
                    position_ids=pos_t,
                    key_padding_mask=valid_mask,
                )
                cur_logits = step_out[:, -1, :]

            for b, sp in enumerate(chunk):
                sess = sessions[b]
                all_results.append(
                    {
                        "bucket": sp.bucket,
                        "won": sess.status == "won",
                        "added_turns": added_turns[b],
                        "rule_violation": rule_violations[b] > 0,
                        "wrong_hits": wrong_h[b],
                        "wrong_opps": wrong_o[b],
                        "exact_hits": exact_h[b],
                        "exact_opps": exact_o[b],
                        "rep_hits": rep_h[b],
                        "rep_opps": rep_o[b],
                    }
                )

    return all_results


def _build_word_trie(words: Sequence[str]) -> dict[Any, Any]:
    root: dict[Any, Any] = {}
    for w in words:
        node = root
        for ch in w:
            idx = ord(ch) - ord("A")
            node = node.setdefault(idx, {})
    return root


def _gather_kv(kv: list[tuple[torch.Tensor, torch.Tensor]], idx: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(k.index_select(0, idx), v.index_select(0, idx)) for k, v in kv]


@torch.no_grad()
def eval_wordle_batched(
    model: PrimeGPT,
    secret_words: Sequence[str],
    dictionary: Sequence[str],
    word_len: int,
    device: torch.device,
    tokenizer: PrimeWordleTokenizer,
    max_turns: int = 10,
    beam_width: int = 6,
    batch_games: int = 10,
) -> dict[str, Any]:
    """Batched Trie Beam Search for 26-letter English Wordle (zero external word elimination!).

    Runs `batch_games * beam_width` beams in parallel on GPU -> ~15x faster than serial beam search!
    """
    trie = _build_word_trie(dictionary)
    cfg = GameConfig(num_primes=26, max_multiplier=1, code_length=word_len, allow_underfill=False)
    turn_id = tokenizer.stoi["<TURN>"]
    slot_id = tokenizer.stoi["<SLOT>"]
    fb_id = tokenizer.stoi["<FB>"]
    s0_id = tokenizer.stoi["S_0"]
    header = tokenizer.encode_config(cfg)

    wins = 0
    turns_list: list[int] = []
    wins_within_6 = 0

    with torch.amp.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
    ):
        for st in range(0, len(secret_words), batch_games):
            chunk = secret_words[st : st + batch_games]
            bsz = len(chunk)
            sessions = [
                GameSession(cfg, seed=1, target_slots=[[ord(ch) - ord("A")] for ch in w])
                for w in chunk
            ]
            tried: list[set[str]] = [set() for _ in range(bsz)]
            active_games = list(range(bsz))

            prompt_t = torch.tensor([header] * bsz, dtype=torch.long, device=device)
            _, kv_games, _ = model(prompt_t, use_cache=True)

            for _t in range(max_turns):
                if not active_games:
                    break
                n_act = len(active_games)

                # Advance <TURN> for all active games
                t_in = torch.full((n_act, 1), turn_id, dtype=torch.long, device=device)
                _, kv_turn, _ = model(t_in, past_kv=kv_games, use_cache=True)

                # Beam state per active game: list of (score, letters, trie_node)
                # Initially 1 beam per active game; expands up to `beam_width` per game!
                beam_scores: list[list[float]] = [[0.0] for _ in range(n_act)]
                beam_letters: list[list[list[int]]] = [[[]] for _ in range(n_act)]
                beam_nodes: list[list[dict[Any, Any]]] = [[trie] for _ in range(n_act)]
                kv_beams = kv_turn  # batch size = sum(len(beam_scores[g]))

                for pos in range(word_len):
                    total_b = sum(len(bs) for bs in beam_scores)
                    # Feed <SLOT> to ALL active beams in ONE GPU call!
                    s_in = torch.full((total_b, 1), slot_id, dtype=torch.long, device=device)
                    s_logits, kv_after_slot, _ = model(s_in, past_kv=kv_beams, use_cache=True)
                    logp_all = torch.log_softmax(s_logits[:, -1, :].float(), dim=-1).cpu()

                    parent_indices: list[int] = []
                    chosen_letter_tids: list[int] = []
                    new_scores: list[list[float]] = []
                    new_letters: list[list[list[int]]] = []
                    new_nodes: list[list[dict[Any, Any]]] = []

                    offset = 0
                    for g_local, g_orig in enumerate(active_games):
                        n_b = len(beam_scores[g_local])
                        cands: list[tuple[float, int, list[int], dict[Any, Any], int]] = []
                        fallback_cands: list[tuple[float, int, list[int], dict[Any, Any], int]] = []
                        for bi in range(n_b):
                            row_lp = logp_all[offset + bi]
                            sc = beam_scores[g_local][bi]
                            lets = beam_letters[g_local][bi]
                            node = beam_nodes[g_local][bi]
                            for c, child in node.items():
                                item = (sc + float(row_lp[s0_id + c].item()), offset + bi, [*lets, c], child, c)
                                if pos == word_len - 1:
                                    w_str = "".join(chr(ord("A") + x) for x in (*lets, c))
                                    if w_str in tried[g_orig]:
                                        fallback_cands.append(item)
                                        continue
                                cands.append(item)
                        offset += n_b

                        if not cands:
                            cands = fallback_cands
                        cands.sort(key=lambda x: -x[0])
                        top_c = cands[:beam_width]
                        new_scores.append([x[0] for x in top_c])
                        new_letters.append([x[2] for x in top_c])
                        new_nodes.append([x[3] for x in top_c])
                        for x in top_c:
                            parent_indices.append(x[1])
                            chosen_letter_tids.append(s0_id + x[4])

                    idx_t = torch.tensor(parent_indices, dtype=torch.long, device=device)
                    kv_sel = _gather_kv(kv_after_slot, idx_t)
                    c_in = torch.tensor(chosen_letter_tids, dtype=torch.long, device=device)[:, None]
                    _, kv_beams, _ = model(c_in, past_kv=kv_sel, use_cache=True)

                    beam_scores = new_scores
                    beam_letters = new_letters
                    beam_nodes = new_nodes

                # Pick top-1 beam per active game and submit guesses
                best_beam_global_idx: list[int] = []
                fb_rows: list[list[int]] = []
                next_active_local: list[int] = []
                next_active_orig: list[int] = []

                offset = 0
                for g_local, g_orig in enumerate(active_games):
                    best_beam_global_idx.append(offset)  # Sorted descending, index 0 is best!
                    best_lets = beam_letters[g_local][0]
                    word = "".join(chr(ord("A") + x) for x in best_lets)
                    tried[g_orig].add(word)
                    rec = sessions[g_orig].submit_guess([[x] for x in best_lets])
                    fb_rows.append([fb_id] + [tokenizer.stoi[fb.value] for fb in rec.feedback])
                    if sessions[g_orig].status == "playing":
                        next_active_local.append(g_local)
                        next_active_orig.append(g_orig)
                    offset += len(beam_scores[g_local])

                if next_active_local:
                    keep_beam_idx = torch.tensor(
                        [best_beam_global_idx[i] for i in next_active_local],
                        dtype=torch.long,
                        device=device,
                    )
                    kv_kept = _gather_kv(kv_beams, keep_beam_idx)
                    fb_t = torch.tensor(
                        [fb_rows[i] for i in next_active_local], dtype=torch.long, device=device
                    )
                    _, kv_games, _ = model(fb_t, past_kv=kv_kept, use_cache=True)
                active_games = next_active_orig

            for sess in sessions:
                if sess.status == "won":
                    wins += 1
                    if len(sess.history) <= 6:
                        wins_within_6 += 1
                turns_list.append(len(sess.history))

    n = len(secret_words)
    return {
        "games": n,
        "dict_size": len(dictionary),
        "win_rate": round(wins / n, 4),
        "win_le6_rate": round(wins_within_6 / n, 4),
        "avg_turns": round(sum(turns_list) / n, 2),
    }


def _summarize_bucket_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_b: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_b.setdefault(r["bucket"], []).append(r)

    def _agg(sub: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(sub)
        wins = sum(int(r["won"]) for r in sub)
        avg_t = sum(r["added_turns"] for r in sub) / max(1, n)
        rule_v = sum(int(r["rule_violation"]) for r in sub) / max(1, n)
        wh = sum(r["wrong_hits"] for r in sub)
        wo = sum(r["wrong_opps"] for r in sub)
        eh = sum(r["exact_hits"] for r in sub)
        eo = sum(r["exact_opps"] for r in sub)
        rh = sum(r["rep_hits"] for r in sub)
        ro = sum(r["rep_opps"] for r in sub)
        return {
            "games": n,
            "win_rate": round(wins / max(1, n), 4),
            "avg_added_turns": round(avg_t, 2),
            "rule_viol_rate": round(rule_v, 4),
            "wrong_mine": round(wh / wo, 4) if wo > 0 else None,
            "exact_drop": round(eh / eo, 4) if eo > 0 else None,
            "repeat_mine": round(rh / ro, 4) if ro > 0 else None,
        }

    out = {b: _agg(sub) for b, sub in by_b.items()}
    out["OVERALL"] = _agg(rows)
    return out


def load_model(ckpt_path: Path, device: torch.device) -> PrimeGPT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = PrimeGPTConfig.from_dict(ckpt["model_config"])
    model = PrimeGPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _fmt_p(v: float | None) -> str:
    return "  n/a" if v is None else f"{v*100:5.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent 310-game generalist blind exam.")
    parser.add_argument(
        "--ckpts",
        nargs="+",
        required=True,
        help="List of label=path pairs, e.g. 400M=checkpoints/deeplogic_400m/best.pt ...",
    )
    parser.add_argument("--dict-dir", type=Path, default=Path("data/wordle_dicts"))
    parser.add_argument("--json-out", type=Path, default=Path("checkpoints/blind_exam_310.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = PrimeWordleTokenizer(max_symbols=32)

    print("Building deterministic 310-game blind exam suite...", flush=True)
    ood_120 = _build_120_ood_specs(seed_base=555_000_000)
    unk_40 = _build_40_unk_specs(seed_base=556_000_000)

    dict_5 = [w.strip() for w in (args.dict_dir / "words_5.txt").read_text(encoding="utf-8").splitlines() if w.strip()]
    dict_6 = [w.strip() for w in (args.dict_dir / "words_6.txt").read_text(encoding="utf-8").splitlines() if w.strip()]
    dict_7 = [w.strip() for w in (args.dict_dir / "words_7.txt").read_text(encoding="utf-8").splitlines() if w.strip()]

    w_rng = random.Random(557_000_000)
    secrets_5 = w_rng.sample(dict_5, 100)
    secrets_6 = w_rng.sample(dict_6, 25)
    secrets_7 = w_rng.sample(dict_7, 25)
    print(
        f"Exam ready: 120 OOD + 40 UNK_CFG + 100 Wordle-5 ({len(dict_5)} dict) + "
        f"25 Wordle-6 ({len(dict_6)} dict) + 25 Wordle-7 ({len(dict_7)} dict) = 310 games!\n",
        flush=True,
    )

    all_reports: dict[str, Any] = {}

    for item in args.ckpts:
        label, path_str = item.split("=", 1)
        ckpt_path = Path(path_str)
        t0 = time.perf_counter()
        print(f"================ Evaluating [{label}] ({ckpt_path}) ================", flush=True)
        model = load_model(ckpt_path, device)

        t1 = time.perf_counter()
        ood_rows = eval_heterogeneous_specs(model, ood_120, device, tok, batch_size=args.batch_size)
        ood_sum = _summarize_bucket_rows(ood_rows)
        print(f"  [1/3] 120 OOD games done in {time.perf_counter()-t1:.1f}s:", flush=True)
        for bname in ("OOD_Small", "OOD_Mid_K34", "OOD_PvZ_Sparse", "OOD_Hard_K58", "OVERALL"):
            s = ood_sum[bname]
            print(
                f"    {bname:<15} | Win={s['win_rate']*100:5.1f}% | AddT={s['avg_added_turns']:5.2f} | "
                f"wrong={_fmt_p(s['wrong_mine'])} | ex_drop={_fmt_p(s['exact_drop'])} | rep={_fmt_p(s['repeat_mine'])}",
                flush=True,
            )

        t2 = time.perf_counter()
        unk_rows = eval_heterogeneous_specs(model, unk_40, device, tok, batch_size=args.batch_size)
        unk_sum = _summarize_bucket_rows(unk_rows)
        print(f"  [2/3] 40 <UNK_CFG> games done in {time.perf_counter()-t2:.1f}s:", flush=True)
        for bname in ("UNK_ZeroPrefix", "UNK_WithPrefix", "OVERALL"):
            s = unk_sum[bname]
            print(
                f"    {bname:<15} | Win={s['win_rate']*100:5.1f}% | AddT={s['avg_added_turns']:5.2f} | "
                f"ruleV={s['rule_viol_rate']*100:5.1f}% | wrong={_fmt_p(s['wrong_mine'])} | "
                f"ex_drop={_fmt_p(s['exact_drop'])} | rep={_fmt_p(s['repeat_mine'])}",
                flush=True,
            )

        t3 = time.perf_counter()
        w5 = eval_wordle_batched(model, secrets_5, dict_5, 5, device, tok, max_turns=10, beam_width=6, batch_games=10)
        w6 = eval_wordle_batched(model, secrets_6, dict_6, 6, device, tok, max_turns=10, beam_width=6, batch_games=10)
        w7 = eval_wordle_batched(model, secrets_7, dict_7, 7, device, tok, max_turns=10, beam_width=6, batch_games=10)
        print(f"  [3/3] 150 Large-Dict Wordle games done in {time.perf_counter()-t3:.1f}s:", flush=True)
        print(
            f"    Wordle-5 (100g, {len(dict_5)}w) | Win10={w5['win_rate']*100:5.1f}% | Win<=6={w5['win_le6_rate']*100:5.1f}% | AvgT={w5['avg_turns']:.2f}\n"
            f"    Wordle-6 ( 25g, {len(dict_6)}w) | Win10={w6['win_rate']*100:5.1f}% | Win<=6={w6['win_le6_rate']*100:5.1f}% | AvgT={w6['avg_turns']:.2f}\n"
            f"    Wordle-7 ( 25g, {len(dict_7)}w) | Win10={w7['win_rate']*100:5.1f}% | Win<=6={w7['win_le6_rate']*100:5.1f}% | AvgT={w7['avg_turns']:.2f}",
            flush=True,
        )

        elapsed = round(time.perf_counter() - t0, 1)
        print(f"  -> [{label}] total wall time: {elapsed}s\n", flush=True)
        all_reports[label] = {
            "ckpt": str(ckpt_path),
            "elapsed_sec": elapsed,
            "ood_120": ood_sum,
            "unk_40": unk_sum,
            "wordle_150": {"wordle_5": w5, "wordle_6": w6, "wordle_7": w7},
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(all_reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved full 310-game blind exam report -> {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
