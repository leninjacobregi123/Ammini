"""Pretrain MalayaLM on the tokenized Malayalam corpus (data/prepare_pretrain.py
output). Equivalent role to ch05's gpt_train.py, but for our own architecture,
with grad accumulation + AMP + checkpointing added (ch05 + appendix-D's bonus
material combined) since a real run is many more steps than the book's demo.

Usage (inside the docker container, on Shannon):
    python train/pretrain.py \
        --config configs/shannon.yaml \
        --tokenizer tokenizer/malayalam_tokenizer.json \
        --data-dir data/prepared \
        --out-dir checkpoints/pretrain \
        --max-steps 20000 --batch-size 32 --grad-accum-steps 4
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.config import load_config, count_params  # noqa: E402
from model.model import MalayaLM  # noqa: E402
from train.lr_schedule import lr_at_step, set_lr  # noqa: E402
from tokenizer.special_tokens import EOS  # noqa: E402

from tokenizers import Tokenizer


def get_batch(data: np.memmap, batch_size: int, context_length: int, device: str):
    ix = torch.randint(0, len(data) - context_length - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + context_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + context_length].astype(np.int64)) for i in ix])
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_val_loss(model, val_data, batch_size, context_length, device, iters=20):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(val_data, batch_size, context_length, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/shannon.yaml")
    ap.add_argument("--tokenizer", default="tokenizer/malayalam_tokenizer.json")
    ap.add_argument("--data-dir", default="data/prepared")
    ap.add_argument("--out-dir", default="checkpoints/pretrain")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    ap.add_argument("--max-lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--ckpt-interval", type=int, default=1000)
    ap.add_argument("--resume", default=None, help="path to a checkpoint .pt to resume from")
    ap.add_argument("--patience", type=int, default=10,
                     help="stop after this many consecutive evals with no val-loss improvement "
                          "(0 disables early stopping and always runs to --max-steps)")
    ap.add_argument("--min-delta", type=float, default=1e-3,
                     help="minimum val-loss improvement to reset patience and save best.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    tok = Tokenizer.from_file(args.tokenizer)
    cfg = load_config(args.config, vocab_size=tok.get_vocab_size())
    eos_id = tok.token_to_id(EOS)

    model = MalayaLM(cfg).to(device)
    params = count_params(model)
    print(f"config: {cfg}")
    print(f"params: total={params['total']:,} active/token={params['active']:,}")

    data_dir = Path(args.data_dir)
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        print(f"resumed from {args.resume} at step {start_step}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    best_val_loss = float("inf")
    patience_counter = 0
    stopped_early = False

    model.train()
    t0 = time.time()
    for step in range(start_step, args.max_steps):
        lr = lr_at_step(step, warmup_steps=args.warmup_steps, max_steps=args.max_steps,
                         max_lr=args.max_lr, min_lr=args.min_lr)
        set_lr(optimizer, lr)

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accum_steps):
            x, y = get_batch(train_data, args.batch_size, cfg.context_length, device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / args.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % 50 == 0:
            dt = time.time() - t0
            print(f"step {step:6d} | loss {accum_loss:.4f} | lr {lr:.2e} | {dt:.1f}s")
            t0 = time.time()

        if step % args.eval_interval == 0 and step > 0:
            val_loss = estimate_val_loss(model, val_data, args.batch_size, cfg.context_length, device)
            improved = val_loss < best_val_loss - args.min_delta
            print(f"  -> val loss {val_loss:.4f}"
                  + (" (best so far, saving best.pt)" if improved else
                     f" (no improvement, patience {patience_counter + 1}/{args.patience})"))

            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                best_path = out_dir / "best.pt"
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "step": step, "val_loss": val_loss, "config": cfg.to_dict()}, best_path)
            elif args.patience > 0:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"early stopping at step {step}: val loss hasn't improved by >= "
                          f"{args.min_delta} for {args.patience} evals (best={best_val_loss:.4f})")
                    stopped_early = True
                    break

        if step % args.ckpt_interval == 0 and step > 0:
            ckpt_path = out_dir / f"step_{step}.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "config": cfg.to_dict()}, ckpt_path)
            print(f"  -> saved checkpoint {ckpt_path}")

    final_path = out_dir / "final.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "config": cfg.to_dict()}, final_path)
    print(f"training {'stopped early' if stopped_early else 'finished'} -> {final_path}")
    if (out_dir / "best.pt").exists():
        print(f"NOTE: best.pt (val loss {best_val_loss:.4f}) is usually what you want to hand to "
              f"finetune/instruction_finetune.py, not final.pt -- final.pt is just the last step "
              f"reached, which may be past the point where val loss stopped improving.")


if __name__ == "__main__":
    main()
