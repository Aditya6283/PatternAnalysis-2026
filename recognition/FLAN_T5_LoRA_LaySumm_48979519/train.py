"""
train.py - Fine-tune FLAN-T5 for layperson radiology summaries (LoRA or full fine-tuning).

Project : Expert-to-layperson translation of radiology reports with FLAN-T5,
          LoRA vs. full fine-tuning (COMP3710 Pattern Analysis, Project 2.6, Hard).

Usage (from this folder, on a GPU node; data and weights fetched first with download.py):
    python train.py --mode lora --out runs/lora        # LoRA adapters on q, v (r=16)
    python train.py --mode full --out runs/full        # every weight trained
    torchrun --standalone --nproc_per_node=2 train.py --mode full --out runs/full_ddp   # multi-GPU

What happens:
  * data: dataset.prepare_splits (dedup, report-grouped dev split); prompts + targets tokenised;
    batches of similar length (less padding);
  * objective: token cross-entropy of the reference lay summary (teacher forcing), bfloat16
    autocast on the GPU, AdamW, linear warm-up then linear decay, gradient clipping at 1.0;
  * every --eval-every optimiser steps: dev loss (all dev rows) and dev ROUGE (greedy decoding
    on --dev-rouge-samples rows); the checkpoint with the lowest dev loss is kept and training
    stops early after --patience evaluations without improvement;
  * outputs in --out: best.pt (LoRA: adapters only, ~7 MB; full: all weights, ~1 GB),
    steps.csv (loss every --log-every steps), evals.csv, curves.png, summary.json
    (parameters, trainable parameters, peak VRAM, time, throughput, best dev metrics).
The official validation split is never used here; evaluate.py uses it once at the end.
"""

import argparse
import os
import time
from collections import defaultdict

import torch
import torch.distributed as dist

from dataset import (DEFAULT_DATA_DIR, DEFAULT_MODEL, LaySummDataset, load_tokenizer,
                     make_eval_loader, make_train_loader, prepare_splits)
from modules import IGNORE_INDEX, LaySummariser, count_parameters
from utils import (autocast_ctx, get_device, library_versions, peak_memory_mb,
                   plot_training, reset_peak_memory, rouge_per_example, save_json, set_seed,
                   write_csv)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune FLAN-T5 (LoRA or full) on BioLaySumm LaymanRRG")
    p.add_argument("--mode", choices=["lora", "full"], required=True)
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out", default=None, help="output folder (default runs/<mode>)")
    # data
    p.add_argument("--dev-size", type=int, default=1000, help="rows held out of train for model selection")
    p.add_argument("--max-train-samples", type=int, default=None, help="subsample training rows (budget/smoke tests)")
    p.add_argument("--no-dedup", action="store_true", help="keep duplicate (report, lay) pairs")
    p.add_argument("--max-source-len", type=int, default=384)
    p.add_argument("--max-target-len", type=int, default=320)
    # optimisation
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=32, help="max examples per batch (per GPU)")
    p.add_argument("--max-tokens", type=int, default=5120,
                   help="max padded tokens (source + target) per batch; long reports get smaller batches "
                        "so GPU memory stays bounded (0 = no limit)")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=None,
                   help="default 1e-3 (lora) / 1e-4 (full): the new LoRA adapters start at zero and "
                        "need larger steps than the pre-trained weights")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", nargs="+", default=["q", "v"],
                   help="T5 linear layers to adapt: q k v o (attention), wi_0 wi_1 wo (feed-forward)")
    # evaluation / logging
    p.add_argument("--eval-every", type=int, default=1000, help="optimiser steps between dev evaluations (0 = per epoch)")
    p.add_argument("--dev-rouge-samples", type=int, default=300, help="dev rows decoded for ROUGE at each evaluation")
    p.add_argument("--gen-max-new-tokens", type=int, default=320)
    p.add_argument("--patience", type=int, default=4, help="evaluations without dev-loss improvement before stopping")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--max-minutes", type=float, default=0, help="stop cleanly after this many minutes (0 = no limit)")
    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu"])
    p.add_argument("--no-amp", action="store_true", help="disable bfloat16 autocast")
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()
    args.out = args.out or os.path.join("runs", args.mode)
    args.lr = args.lr or (1e-3 if args.mode == "lora" else 1e-4)
    return args


# ---------------------------------------------------------------------------
# Distributed set-up (torchrun); a plain `python train.py` runs as a single process
# ---------------------------------------------------------------------------
def setup_distributed(prefer: str):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return False, 0, 1, get_device(prefer)
    use_cuda = torch.cuda.is_available() and prefer != "cpu"
    dist.init_process_group("nccl" if use_cuda else "gloo")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return True, dist.get_rank(), dist.get_world_size(), device


def param_groups(model, weight_decay: float):
    """AdamW groups: no weight decay on biases and layer-norm scales (standard practice)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if ("layer_norm" in name or name.endswith("bias")) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def make_optimizer(groups, lr: float, device: torch.device):
    try:
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                                 fused=device.type == "cuda")
    except (TypeError, RuntimeError):  # older PyTorch without the fused kernel
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)


# ---------------------------------------------------------------------------
# Dev evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_dev(model: LaySummariser, dev_loader, dev_df, tokenizer, device, args, amp: bool):
    """Token-weighted dev cross-entropy on all dev rows + ROUGE on the first n dev rows."""
    model.eval()
    total, n_tok = 0.0, 0
    for batch in dev_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with autocast_ctx(device, amp):
            loss = model(**batch).loss
        tokens = int((batch["labels"] != IGNORE_INDEX).sum())
        total += loss.item() * tokens
        n_tok += tokens
    metrics = {"dev_loss": total / max(1, n_tok)}
    if args.dev_rouge_samples > 0:
        sub = dev_df.head(args.dev_rouge_samples)
        outs = model.summarise(sub["radiology_report"].tolist(), tokenizer, device, batch_size=64,
                               max_source_len=args.max_source_len,
                               max_new_tokens=args.gen_max_new_tokens, amp=amp)
        scores = rouge_per_example([o["summary"] for o in outs], sub["layman_report"].tolist())
        metrics.update({f"dev_{k}": float(v.mean()) for k, v in scores.items()})
    model.train()
    return metrics


def save_checkpoint(path: str, model: LaySummariser, args, **extra) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"config": model.config, "mode": model.mode, "state_dict": model.trainable_state_dict(),
               "train_args": vars(args),
               "data": {"dev_size": args.dev_size, "seed": args.seed, "dedup": not args.no_dedup,
                        "max_train": args.max_train_samples}, **extra}
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    ddp, rank, world, device = setup_distributed(args.device)
    is_main = rank == 0
    amp = not args.no_amp and device.type == "cuda"
    set_seed(args.seed)
    if is_main:
        os.makedirs(args.out, exist_ok=True)
        print(f"[train] mode={args.mode} model={args.model} device={device} world={world} "
              f"bf16={amp} lr={args.lr}", flush=True)

    splits = prepare_splits(args.data_dir, dev_size=args.dev_size, seed=args.seed,
                            dedup=not args.no_dedup, max_train=args.max_train_samples, verbose=is_main)
    tokenizer = load_tokenizer(args.model)
    train_ds = LaySummDataset(splits["train"], tokenizer, args.max_source_len, args.max_target_len)
    dev_ds = LaySummDataset(splits["dev"], tokenizer, args.max_source_len, args.max_target_len)
    if is_main:
        print(f"[train] tokenised: train {len(train_ds)} (truncated sources {train_ds.truncated_sources}, "
              f"targets {train_ds.truncated_targets}), dev {len(dev_ds)}", flush=True)

    model = LaySummariser(args.model, args.mode, args.lora_r, args.lora_alpha, args.lora_dropout,
                          args.lora_targets).to(device)
    total_params, trainable_params = count_parameters(model)
    if is_main:
        print(f"[train] parameters: total {total_params:,}, trainable {trainable_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)"
              + (f"; LoRA on {len(model.lora_modules)} layers" if args.mode == "lora" else ""), flush=True)
    optimizer = make_optimizer(param_groups(model, args.weight_decay), args.lr, device)
    train_loader = make_train_loader(train_ds, args.batch_size, tokenizer.pad_token_id, args.seed,
                                     rank=rank, world_size=world, num_workers=args.num_workers,
                                     max_tokens=args.max_tokens or None)
    dev_loader = make_eval_loader(dev_ds, 64, tokenizer.pad_token_id, max_tokens=2 * args.max_tokens or None)
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup = max(1, int(args.warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:  # linear warm-up, then linear decay to zero
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    train_model = model
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        train_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    eval_every = args.eval_every if args.eval_every > 0 else steps_per_epoch
    if is_main:
        print(f"[train] {steps_per_epoch} optimiser steps/epoch x {args.epochs} epochs = {total_steps} steps; "
              f"warm-up {warmup}; dev evaluation every {eval_every} steps", flush=True)

    step_log, eval_log = [], []
    best_loss, best_step, best_metrics, bad_evals = float("inf"), 0, {}, 0
    window = defaultdict(float)
    global_step, stop = 0, False
    reset_peak_memory(device)
    start = time.time()
    last_log = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    def run_eval():
        nonlocal best_loss, best_step, best_metrics, bad_evals
        metrics = evaluate_dev(model, dev_loader, splits["dev"], tokenizer, device, args, amp)
        row = {"step": global_step, "epoch": round(global_step / steps_per_epoch, 4),
               "minutes": round((time.time() - start) / 60, 2), **metrics}
        eval_log.append(row)
        improved = metrics["dev_loss"] < best_loss - 1e-4
        if improved:
            best_loss, best_step, best_metrics, bad_evals = metrics["dev_loss"], global_step, row, 0
            save_checkpoint(os.path.join(args.out, "best.pt"), model, args, step=global_step,
                            epoch=row["epoch"], dev=metrics)
        else:
            bad_evals += 1
        rouge_txt = " ".join(f"{k[4:]} {v:.4f}" for k, v in metrics.items() if k.startswith("dev_rouge"))
        print(f"[eval] step {global_step} (epoch {row['epoch']:.2f}) dev loss {metrics['dev_loss']:.4f} "
              f"{rouge_txt} {'*best*' if improved else f'(no improvement {bad_evals}/{args.patience})'}",
              flush=True)
        return bad_evals >= args.patience

    epoch = 0
    while not stop and global_step < total_steps:
        train_loader.batch_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with autocast_ctx(device, amp):
                loss = train_model(**batch).loss / args.grad_accum
            loss.backward()
            window["loss"] += loss.item() * args.grad_accum
            window["batches"] += 1
            window["tokens"] += int(batch["attention_mask"].sum()) + int((batch["labels"] != IGNORE_INDEX).sum())
            if (i + 1) % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if is_main and global_step % args.log_every == 0:
                elapsed = time.time() - last_log
                row = {"step": global_step, "epoch": round(global_step / steps_per_epoch, 4),
                       "train_loss": window["loss"] / window["batches"], "lr": scheduler.get_last_lr()[0],
                       "tokens_per_s": window["tokens"] * world / max(elapsed, 1e-9)}
                step_log.append(row)
                print(f"[train] step {global_step}/{total_steps} epoch {row['epoch']:.2f} "
                      f"loss {row['train_loss']:.4f} lr {row['lr']:.2e} {row['tokens_per_s']:.0f} tok/s", flush=True)
                window.clear()
                last_log = time.time()

            # Every rank must take the same branch (collectives), so the decision points depend only
            # on global_step and rank 0 broadcasts its verdict: [patience exhausted, out of time].
            do_eval = global_step % eval_every == 0 or global_step >= total_steps
            check_time = bool(args.max_minutes) and global_step % args.log_every == 0
            if do_eval or check_time:
                flag = torch.zeros(2, device=device)
                if is_main:
                    out_of_time = bool(args.max_minutes) and (time.time() - start) / 60 > args.max_minutes
                    if do_eval or out_of_time:
                        flag[0] = float(run_eval())
                    flag[1] = float(out_of_time)
                if ddp:
                    dist.broadcast(flag, src=0)
                if flag[1] > 0:
                    if is_main:
                        print(f"[train] time limit of {args.max_minutes} min reached, stopping", flush=True)
                    stop = True
                elif flag[0] > 0:
                    if is_main:
                        print(f"[train] early stopping: dev loss did not improve for {args.patience} evaluations",
                              flush=True)
                    stop = True
            if stop or global_step >= total_steps:
                break
        epoch += 1

    if is_main:
        total_time = time.time() - start
        write_csv(step_log, os.path.join(args.out, "steps.csv"))
        write_csv(eval_log, os.path.join(args.out, "evals.csv"))
        plot_training(step_log, eval_log, os.path.join(args.out, "curves.png"),
                      f"{os.path.basename(args.model)} {'LoRA' if args.mode == 'lora' else 'full'} "
                      f"fine-tuning (best dev loss at step {best_step})")
        ckpt_mb = os.path.getsize(os.path.join(args.out, "best.pt")) / 2 ** 20 if best_step else float("nan")
        tok_rates = [r["tokens_per_s"] for r in step_log[1:]] or [r["tokens_per_s"] for r in step_log]
        summary = {"mode": args.mode, "model": args.model, "parameters_total": total_params,
                   "parameters_trainable": trainable_params,
                   "trainable_percent": 100 * trainable_params / total_params,
                   "train_rows": len(train_ds), "dev_rows": len(dev_ds), "world_size": world,
                   "steps_run": global_step, "steps_planned": total_steps, "steps_per_epoch": steps_per_epoch,
                   "epochs_run": round(global_step / steps_per_epoch, 3), "best_step": best_step,
                   "best_epoch": best_metrics.get("epoch"), "best_dev": best_metrics,
                   "train_time_min": round(total_time / 60, 2),
                   "train_tokens_per_s": sum(tok_rates) / len(tok_rates) if tok_rates else None,
                   "peak_train_vram_mb": round(peak_memory_mb(device), 1),
                   "checkpoint_mb": round(ckpt_mb, 2), "train_args": vars(args),
                   "versions": library_versions()}
        save_json(summary, os.path.join(args.out, "summary.json"))
        print(f"[train] done in {total_time / 60:.1f} min; best dev loss {best_loss:.4f} at step {best_step}; "
              f"peak VRAM {summary['peak_train_vram_mb']} MB; checkpoint {summary['checkpoint_mb']} MB", flush=True)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
