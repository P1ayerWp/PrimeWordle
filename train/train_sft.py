"""Streaming 1-Epoch Multi-Shard RTX 4060 bf16 SFT trainer for DeepLogic PrimeGPT.

Features:
  - Pure 1-Epoch streaming across numbered `.pt` shards (tiny RAM footprint even for 400M+ tokens)
  - Token-budget Cosine LR schedule calibrated to exact total tokens across all shards
  - Split accuracy metrics: `struct_acc` (<TURN>/<SLOT>/<FB>/<EOS>) vs `sym_acc` (S_0..S_31 only!)
  - Token-interval evaluation (`--eval-every-tokens`, e.g. 40,000,000) with tiered live gameplay
    + raw syntax violation rate + 3 mine-stepping X-ray metrics (wrong_mine / exact_drop / repeat_mine)
  - Adaptive gradient checkpointing (fast on short batches, safe 8GB VRAM up to 32K)
  - Exact shard + token checkpointing & seamless `--resume` support + `--max-minutes` pause mode
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from engine.tokenizer import PrimeWordleTokenizer
from model import PRESETS, PrimeGPT, compute_masked_lm_loss

from .dataset import ShardEpisodeDataset, TokenBudgetBatchSampler, collate_episodes
from .eval_play import evaluate_model_play


def resolve_shard_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pat in patterns:
        matched = sorted(glob.glob(pat))
        if matched:
            paths.extend(Path(m) for m in matched)
        else:
            p = Path(pat)
            if p.exists():
                paths.append(p)
            else:
                raise FileNotFoundError(f"Shard path/pattern matched nothing: {pat}")
    # Preserve order, deduplicate
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def build_optimizer(
    model: PrimeGPT, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    seen: set[int] = set()

    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim >= 2 and "tok_emb" not in name and "lm_head" not in name:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    use_fused = torch.cuda.is_available()
    return torch.optim.AdamW(
        groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
    )


def cosine_lr_by_tokens(
    tokens_seen: int,
    total_tokens: int,
    warmup_tokens: int,
    max_lr: float,
    min_lr: float,
) -> float:
    if tokens_seen < warmup_tokens:
        return max_lr * (tokens_seen + 1) / max(1, warmup_tokens)
    if tokens_seen >= total_tokens:
        return min_lr
    progress = (tokens_seen - warmup_tokens) / max(1, total_tokens - warmup_tokens)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _split_action_accuracy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    sym_lo: int,
    sym_hi: int,
) -> tuple[int, int, int, int]:
    """Return (sym_correct, sym_total, struct_correct, struct_total)."""
    shift_preds = logits[:, :-1, :].argmax(dim=-1)
    shift_targets = input_ids[:, 1:]
    act_mask = loss_mask[:, 1:].bool()

    sym_mask = act_mask & (shift_targets >= sym_lo) & (shift_targets <= sym_hi)
    struct_mask = act_mask & ~sym_mask

    correct = shift_preds == shift_targets
    sym_c = int((correct & sym_mask).sum().item())
    sym_n = int(sym_mask.sum().item())
    str_c = int((correct & struct_mask).sum().item())
    str_n = int(struct_mask.sum().item())
    return sym_c, sym_n, str_c, str_n


@torch.no_grad()
def evaluate_shard(
    model: PrimeGPT,
    loader: DataLoader,
    device: torch.device,
    sym_lo: int,
    sym_hi: int,
) -> dict[str, float]:
    """Return val_loss, val_sym_acc, val_struct_acc, val_action_acc."""
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_act = 0
    sym_c = sym_n = str_c = str_n = 0

    for input_ids, loss_mask, _ in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        loss_mask = loss_mask.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
        ):
            logits, _, _ = model(input_ids)
            loss = compute_masked_lm_loss(logits, input_ids, loss_mask)

        sc, sn, tc, tn = _split_action_accuracy(logits, input_ids, loss_mask, sym_lo, sym_hi)
        n_act = sn + tn
        total_loss += float(loss.item()) * n_act
        total_act += n_act
        sym_c += sc
        sym_n += sn
        str_c += tc
        str_n += tn

    if was_training:
        model.train()

    return {
        "val_loss": total_loss / max(1, total_act),
        "val_sym_acc": sym_c / max(1, sym_n),
        "val_struct_acc": str_c / max(1, str_n),
        "val_action_acc": (sym_c + str_c) / max(1, total_act),
    }


def save_checkpoint(
    path: Path,
    model: PrimeGPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    shard_idx: int,
    batch_idx_in_shard: int,
    tokens_seen: int,
    next_eval_tokens: int,
    best_val_loss: float,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model_config": model.cfg.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "shard_idx": shard_idx,
            "batch_idx_in_shard": batch_idx_in_shard,
            "tokens_seen": tokens_seen,
            "next_eval_tokens": next_eval_tokens,
            "best_val_loss": best_val_loss,
            "metrics": metrics,
        },
        tmp_path,
    )
    tmp_path.replace(path)


def _fmt_pct(val: float | None) -> str:
    return "n/a" if val is None else f"{val * 100:.1f}%"


def _run_full_eval(
    model: PrimeGPT,
    val_loader: DataLoader | None,
    device: torch.device,
    sym_lo: int,
    sym_hi: int,
    play_eval_games: int,
    step: int,
    tokens_seen: int,
    eval_log_path: Path,
) -> dict[str, Any]:
    val_stats: dict[str, float] = {}
    if val_loader is not None:
        val_stats = evaluate_shard(model, val_loader, device, sym_lo, sym_hi)

    play_stats: dict[str, Any] = {}
    if play_eval_games > 0:
        play_stats = evaluate_model_play(model, device, games_per_preset=play_eval_games)

    parts = [f"  [EVAL step={step} | tok={tokens_seen/1e6:.2f}M]"]
    if val_stats:
        parts.append(
            f"val_loss={val_stats['val_loss']:.4f} | "
            f"sym_acc={val_stats['val_sym_acc']*100:.2f}% | "
            f"struct_acc={val_stats['val_struct_acc']*100:.2f}%"
        )
    if play_stats:
        b = play_stats["buckets"]
        x = play_stats["xray"]
        parts.append(
            f"WIN(S/M/L)={b['small']*100:.0f}%/{b['mid']*100:.0f}%/{b['large']*100:.0f}% | "
            f"XRAY(syn={_fmt_pct(x['raw_syntax_turn_err'])}, "
            f"wrong={_fmt_pct(x['wrong_mine'])}, "
            f"ex_drop={_fmt_pct(x['exact_drop'])}, "
            f"rep={_fmt_pct(x['repeat_mine'])})"
        )
    print(" | ".join(parts), flush=True)

    record = {
        "step": step,
        "tokens_seen": tokens_seen,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "val": {k: round(v, 5) for k, v in val_stats.items()},
        "play": play_stats,
    }
    with eval_log_path.open("a", encoding="utf-8") as ef:
        ef.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def train_sft(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tok = PrimeWordleTokenizer(max_symbols=32)
    sym_lo = tok.stoi["S_0"]
    sym_hi = tok.stoi["S_31"]

    train_shard_paths = resolve_shard_paths(args.train_shards)
    val_shard_paths = resolve_shard_paths(args.val_shards) if args.val_shards else []

    # Probe exact token totals across all shards cleanly without keeping them all in RAM
    shard_token_counts: list[int] = []
    shard_game_counts: list[int] = []
    for sp in train_shard_paths:
        meta_file = sp.with_suffix(".meta.json")
        if meta_file.exists():
            m = json.loads(meta_file.read_text(encoding="utf-8"))
            shard_token_counts.append(int(m["total_tokens"]))
            shard_game_counts.append(int(m["accepted_games"]))
        else:
            d = torch.load(sp, map_location="cpu", weights_only=True)
            shard_token_counts.append(int(d["input_ids"].numel()))
            shard_game_counts.append(int(d["num_games"]))
            del d

    dataset_total_tokens = sum(shard_token_counts) * args.epochs
    total_tokens_budget = (
        args.total_tokens_budget if args.total_tokens_budget > 0 else dataset_total_tokens
    )
    warmup_tokens = (
        args.warmup_tokens
        if args.warmup_tokens > 0
        else min(8_000_000, max(200_000, int(total_tokens_budget * 0.02)))
    )

    val_loader = None
    val_games = 0
    if val_shard_paths:
        val_ds = ShardEpisodeDataset(val_shard_paths)
        val_games = len(val_ds)
        val_sampler = TokenBudgetBatchSampler(
            val_ds.lengths,
            max_tokens=args.micro_tokens,
            max_batch_size=args.max_batch_size,
            shuffle=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=val_sampler,
            collate_fn=collate_episodes,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    cfg = PRESETS[args.preset]
    model = PrimeGPT(cfg).to(device)
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    start_epoch = 0
    resume_shard_idx = 0
    resume_batch_idx = 0
    tokens_seen = 0
    next_eval_tokens = args.eval_every_tokens if args.eval_every_tokens > 0 else 0
    best_val_loss = float("inf")

    if args.resume is not None and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        global_step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", 0))
        resume_shard_idx = int(ckpt.get("shard_idx", 0))
        resume_batch_idx = int(ckpt.get("batch_idx_in_shard", 0))
        tokens_seen = int(ckpt.get("tokens_seen", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        saved_next_eval = int(ckpt.get("next_eval_tokens", 0))
        if args.eval_every_tokens > 0:
            next_eval_tokens = (
                saved_next_eval
                if saved_next_eval > tokens_seen
                else ((tokens_seen // args.eval_every_tokens) + 1) * args.eval_every_tokens
            )
        print(
            f"Resumed from {args.resume} | step={global_step} | epoch={start_epoch} | "
            f"shard={resume_shard_idx} batch={resume_batch_idx} | "
            f"tokens_seen={tokens_seen:,} | best_val={best_val_loss:.4f}",
            flush=True,
        )
    elif getattr(args, "init_from", None) is not None and Path(args.init_from).exists():
        ckpt = torch.load(args.init_from, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        print(
            f"Initialized weights from {args.init_from} (fresh optimizer & fresh LR schedule)",
            flush=True,
        )

    print(
        f"=== PrimeGPT Streaming SFT ({args.preset}, {model.num_parameters()/1e6:.2f}M params) on {device} ===\n"
        f"Shards: {len(train_shard_paths)} ({sum(shard_game_counts):,} games, {sum(shard_token_counts):,} tok/ep) | "
        f"Val: {val_games:,} games | LR Budget: {total_tokens_budget:,} tok (warmup {warmup_tokens:,})\n"
        f"micro_tokens={args.micro_tokens:,} x grad_accum={args.grad_accum} | "
        f"eval_every_tokens={args.eval_every_tokens:,} | max_steps={args.max_steps} | max_minutes={args.max_minutes}",
        flush=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = out_dir / "train_log.jsonl"
    eval_log_path = out_dir / "eval_log.jsonl"

    model.train()
    optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    t_window = t_start
    run_tokens_start = tokens_seen
    run_steps_start = global_step

    running_loss = 0.0
    running_act_toks = 0
    running_sym_c = running_sym_n = 0
    running_str_c = running_str_n = 0
    window_real_toks = 0
    window_padded_toks = 0
    micro_count = 0

    last_eval_record: dict[str, Any] = {}
    stop_reason = "completed"
    cur_epoch = start_epoch
    cur_shard_idx = resume_shard_idx
    cur_batch_idx = resume_batch_idx

    stop_training = False

    for epoch in range(start_epoch, args.epochs):
        cur_epoch = epoch
        shard_start = resume_shard_idx if epoch == start_epoch else 0

        for s_idx in range(shard_start, len(train_shard_paths)):
            cur_shard_idx = s_idx
            shard_path = train_shard_paths[s_idx]
            shard_ds = ShardEpisodeDataset([shard_path])
            shard_sampler = TokenBudgetBatchSampler(
                shard_ds.lengths,
                max_tokens=args.micro_tokens,
                max_batch_size=args.max_batch_size,
                shuffle=True,
                seed=args.seed + s_idx * 97,
            )
            shard_sampler.set_epoch(epoch)
            shard_loader = DataLoader(
                shard_ds,
                batch_sampler=shard_sampler,
                collate_fn=collate_episodes,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
            )

            skip_batches = (
                resume_batch_idx
                if (epoch == start_epoch and s_idx == resume_shard_idx)
                else 0
            )
            print(
                f"-> Loading shard {s_idx + 1}/{len(train_shard_paths)}: {shard_path} "
                f"({len(shard_ds):,} games, {shard_ds.total_tokens:,} tok, skip_batches={skip_batches})",
                flush=True,
            )

            for b_idx, (input_ids, loss_mask, real_toks) in enumerate(shard_loader):
                if b_idx < skip_batches:
                    continue
                cur_batch_idx = b_idx + 1

                input_ids = input_ids.to(device, non_blocking=True)
                loss_mask = loss_mask.to(device, non_blocking=True)
                seq_len = input_ids.size(1)

                model.gradient_checkpointing = seq_len > args.ckpt_threshold

                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=(device.type == "cuda"),
                ):
                    logits, _, _ = model(input_ids)
                    loss = compute_masked_lm_loss(logits, input_ids, loss_mask)
                    scaled_loss = loss / args.grad_accum

                scaled_loss.backward()

                with torch.no_grad():
                    sc, sn, tc, tn = _split_action_accuracy(
                        logits, input_ids, loss_mask, sym_lo, sym_hi
                    )
                    n_act = sn + tn
                    running_loss += float(loss.item()) * n_act
                    running_act_toks += n_act
                    running_sym_c += sc
                    running_sym_n += sn
                    running_str_c += tc
                    running_str_n += tn
                    window_real_toks += real_toks
                    window_padded_toks += int(input_ids.numel())
                    tokens_seen += real_toks

                micro_count += 1
                if micro_count % args.grad_accum == 0:
                    lr = cosine_lr_by_tokens(
                        tokens_seen,
                        total_tokens_budget,
                        warmup_tokens,
                        args.lr,
                        args.min_lr,
                    )
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr

                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    elapsed_now = time.perf_counter() - t_start

                    # Periodic train log
                    if (
                        global_step % args.log_every == 0
                        or global_step == run_steps_start + 1
                    ):
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        now = time.perf_counter()
                        dt = max(1e-6, now - t_window)
                        tps = window_real_toks / dt
                        pad_eff = window_real_toks / max(1, window_padded_toks)
                        avg_loss = running_loss / max(1, running_act_toks)
                        sym_acc = running_sym_c / max(1, running_sym_n)
                        str_acc = running_str_c / max(1, running_str_n)
                        vram_mb = (
                            torch.cuda.max_memory_allocated() / 1024**2
                            if device.type == "cuda"
                            else 0.0
                        )
                        run_tps = (tokens_seen - run_tokens_start) / max(1e-6, now - t_start)
                        rem_toks = max(0, total_tokens_budget - tokens_seen)
                        eta_min = (rem_toks / max(1.0, run_tps)) / 60.0

                        print(
                            f"step {global_step:5d} | sh {s_idx+1}/{len(train_shard_paths)} | "
                            f"tok {tokens_seen/1e6:6.2f}M/{total_tokens_budget/1e6:.0f}M | "
                            f"loss {avg_loss:.4f} | sym {sym_acc*100:5.1f}% | str {str_acc*100:5.1f}% | "
                            f"lr {lr:.2e} | {tps:7,.0f} tok/s (eff {pad_eff*100:4.1f}%) | "
                            f"VRAM {vram_mb:5.0f}MB | ETA {eta_min:5.1f}m",
                            flush=True,
                        )
                        with train_log_path.open("a", encoding="utf-8") as lf:
                            lf.write(
                                json.dumps(
                                    {
                                        "step": global_step,
                                        "epoch": epoch + 1,
                                        "shard_idx": s_idx,
                                        "tokens_seen": tokens_seen,
                                        "train_loss": round(avg_loss, 5),
                                        "train_sym_acc": round(sym_acc, 4),
                                        "train_struct_acc": round(str_acc, 4),
                                        "lr": lr,
                                        "tok_per_sec": round(tps, 1),
                                        "pad_eff": round(pad_eff, 4),
                                        "vram_mb": round(vram_mb, 1),
                                    }
                                )
                                + "\n"
                            )

                        running_loss = 0.0
                        running_act_toks = 0
                        running_sym_c = running_sym_n = 0
                        running_str_c = running_str_n = 0
                        window_real_toks = 0
                        window_padded_toks = 0
                        t_window = time.perf_counter()

                    # Periodic token-based evaluation + checkpointing
                    trigger_eval = False
                    if args.eval_every_tokens > 0 and tokens_seen >= next_eval_tokens:
                        trigger_eval = True
                        while next_eval_tokens <= tokens_seen:
                            next_eval_tokens += args.eval_every_tokens
                    elif args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                        trigger_eval = True

                    if trigger_eval:
                        last_eval_record = _run_full_eval(
                            model,
                            val_loader,
                            device,
                            sym_lo,
                            sym_hi,
                            args.play_eval_games,
                            global_step,
                            tokens_seen,
                            eval_log_path,
                        )
                        v_loss = last_eval_record.get("val", {}).get("val_loss", float("inf"))
                        if v_loss < best_val_loss:
                            best_val_loss = v_loss
                            save_checkpoint(
                                out_dir / "best.pt",
                                model,
                                optimizer,
                                global_step,
                                epoch,
                                s_idx,
                                cur_batch_idx,
                                tokens_seen,
                                next_eval_tokens,
                                best_val_loss,
                                last_eval_record,
                            )
                        save_checkpoint(
                            out_dir / "latest.pt",
                            model,
                            optimizer,
                            global_step,
                            epoch,
                            s_idx,
                            cur_batch_idx,
                            tokens_seen,
                            next_eval_tokens,
                            best_val_loss,
                            last_eval_record,
                        )
                        # Reset window timer so eval time doesn't skew tok/s
                        t_window = time.perf_counter()

                    # Check stop limits (for smoke tests or 2-minute pause-and-verify)
                    if args.max_steps > 0 and (global_step - run_steps_start) >= args.max_steps:
                        stop_reason = f"max_steps({args.max_steps})"
                        stop_training = True
                        break
                    if args.max_minutes > 0 and (elapsed_now / 60.0) >= args.max_minutes:
                        stop_reason = f"max_minutes({args.max_minutes})"
                        stop_training = True
                        break

            # Advance resume pointer cleanly to next shard if this shard finished
            if not stop_training:
                cur_shard_idx = s_idx + 1
                cur_batch_idx = 0
            del shard_ds, shard_loader
            if stop_training:
                break

        if stop_training:
            break
        cur_epoch = epoch + 1
        cur_shard_idx = 0
        cur_batch_idx = 0

    # Final evaluation + final checkpoint save
    if not last_eval_record or last_eval_record.get("step") != global_step:
        last_eval_record = _run_full_eval(
            model,
            val_loader,
            device,
            sym_lo,
            sym_hi,
            args.play_eval_games,
            global_step,
            tokens_seen,
            eval_log_path,
        )
        v_loss = last_eval_record.get("val", {}).get("val_loss", float("inf"))
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            save_checkpoint(
                out_dir / "best.pt",
                model,
                optimizer,
                global_step,
                cur_epoch,
                cur_shard_idx,
                cur_batch_idx,
                tokens_seen,
                next_eval_tokens,
                best_val_loss,
                last_eval_record,
            )

    save_checkpoint(
        out_dir / "latest.pt",
        model,
        optimizer,
        global_step,
        cur_epoch,
        cur_shard_idx,
        cur_batch_idx,
        tokens_seen,
        next_eval_tokens,
        best_val_loss,
        last_eval_record,
    )

    total_time = time.perf_counter() - t_start
    run_toks = tokens_seen - run_tokens_start
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0

    summary = {
        "stop_reason": stop_reason,
        "preset": args.preset,
        "params": model.num_parameters(),
        "step": global_step,
        "tokens_seen_total": tokens_seen,
        "tokens_seen_this_run": run_toks,
        "elapsed_sec": round(total_time, 1),
        "avg_tok_per_sec": round(run_toks / max(1e-6, total_time), 1),
        "peak_vram_mb": round(peak_vram, 1),
        "best_val_loss": round(best_val_loss, 4) if math.isfinite(best_val_loss) else None,
        "last_eval": last_eval_record,
        "checkpoint_dir": str(out_dir),
    }
    print("\n=== Training Session Summary ===", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming 1-Epoch SFT trainer for PrimeGPT.")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="deeplogic_25m")
    parser.add_argument("--train-shards", nargs="+", required=True, help="Shard paths or glob patterns.")
    parser.add_argument("--val-shards", nargs="*", default=[])
    parser.add_argument("--out-dir", type=str, default="checkpoints/deeplogic_400m")
    parser.add_argument("--init-from", type=str, default=None, help="Load weights only (fresh optimizer & LR schedule).")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--total-tokens-budget", type=int, default=0, help="Override total token budget for Cosine LR.")
    parser.add_argument("--warmup-tokens", type=int, default=0, help="Override LR warmup tokens (default 2% capped at 8M).")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after N new optimizer steps (0 = full run).")
    parser.add_argument("--max-minutes", type=float, default=0.0, help="Pause cleanly after M minutes (0 = no limit).")
    parser.add_argument("--micro-tokens", type=int, default=8192)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--ckpt-threshold", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every-tokens", type=int, default=40_000_000)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--play-eval-games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_sft(args)


if __name__ == "__main__":
    main()
