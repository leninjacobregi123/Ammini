"""LoRA instruction-finetune a pretrained MalayaLM checkpoint into a Malayalam
chat assistant, using the unified instruction dataset from
data/prepare_instruct.py and the chat template from tokenizer/special_tokens.py.

Equivalent role to ch07's gpt_instruction_finetuning.py, but LoRA-based (like
appendix-E) since we're adapting our own pretrained-from-scratch checkpoint
rather than full-finetuning a downloaded GPT-2.

Usage:
    python finetune/instruction_finetune.py \
        --pretrained checkpoints/pretrain/final.pt \
        --tokenizer tokenizer/malayalam_tokenizer.json \
        --data data/prepared/instruct.json \
        --out checkpoints/instruct/malayalam_assistant.pt \
        --lora-rank 16 --lora-alpha 16 --epochs 3
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.config import MalayaLMConfig  # noqa: E402
from model.model import MalayaLM  # noqa: E402
from finetune.lora import replace_linear_with_lora, freeze_base_model, count_trainable  # noqa: E402
from tokenizer.special_tokens import build_training_example, PAD, DEFAULT_SYSTEM_PROMPT  # noqa: E402
from train.lr_schedule import lr_at_step, set_lr  # noqa: E402

from tokenizers import Tokenizer


class InstructionDataset(Dataset):
    def __init__(self, records, tokenizer, system_prompt, mask_prompt_loss=True):
        self.examples = []
        pad_id = tokenizer.token_to_id(PAD)
        for rec in records:
            prompt_text, full_text = build_training_example(
                rec["instruction"], rec.get("input", ""), rec["output"], system_prompt,
            )
            full_ids = tokenizer.encode(full_text).ids
            if mask_prompt_loss:
                prompt_len = len(tokenizer.encode(prompt_text).ids)
            else:
                prompt_len = 0
            self.examples.append((full_ids, prompt_len))
        self.pad_id = pad_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def make_collate_fn(pad_id, context_length):
    def collate(batch):
        max_len = min(max(len(ids) for ids, _ in batch), context_length)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        targets = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, (ids, prompt_len) in enumerate(batch):
            ids = ids[:max_len]
            length = len(ids)
            input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            if length > 1:
                tgt = torch.tensor(ids[1:], dtype=torch.long)
                start = max(prompt_len - 1, 0)
                targets[i, start:length - 1] = tgt[start:]
        return input_ids, targets
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", required=True, help="checkpoint .pt from train/pretrain.py")
    ap.add_argument("--tokenizer", default="tokenizer/malayalam_tokenizer.json")
    ap.add_argument("--data", default="data/prepared/instruct.json")
    ap.add_argument("--out", default="checkpoints/instruct/malayalam_assistant.pt")
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--train-norms", action="store_true", default=True,
                     help="also unfreeze RMSNorm weights (cheap, helps adaptation)")
    ap.add_argument("--mask-prompt-loss", action="store_true", default=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    tok = Tokenizer.from_file(args.tokenizer)
    pad_id = tok.token_to_id(PAD)

    ckpt = torch.load(args.pretrained, map_location=device)
    cfg = MalayaLMConfig(**ckpt["config"])
    model = MalayaLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    freeze_base_model(model)
    replace_linear_with_lora(model, args.lora_rank, args.lora_alpha)
    if args.train_norms:
        for module in model.modules():
            if module.__class__.__name__ == "RMSNorm":
                for p in module.parameters():
                    p.requires_grad = True
    model.to(device)

    stats = count_trainable(model)
    print(f"trainable params: {stats['trainable']:,} / {stats['total']:,} ({stats['pct']:.2f}%)")

    with open(args.data, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"loaded {len(records)} instruction records")

    dataset = InstructionDataset(records, tok, args.system_prompt, args.mask_prompt_loss)
    collate_fn = make_collate_fn(pad_id, cfg.context_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    total_steps = args.epochs * len(loader)
    step = 0
    model.train()
    for epoch in range(args.epochs):
        for input_ids, targets in loader:
            input_ids, targets = input_ids.to(device), targets.to(device)
            lr = lr_at_step(step, warmup_steps=args.warmup_steps, max_steps=total_steps,
                             max_lr=args.lr, min_lr=args.lr * 0.1)
            set_lr(optimizer, lr)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(input_ids, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()

            if step % 20 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} | loss {loss.item():.4f} | lr {lr:.2e}")
            step += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": ckpt["config"],
                "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                "system_prompt": args.system_prompt}, out_path)
    print(f"saved instruction-tuned model -> {out_path}")


if __name__ == "__main__":
    main()
