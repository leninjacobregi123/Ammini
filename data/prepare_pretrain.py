"""Tokenize raw .txt shards (from download_corpus.py) into flat uint16 token-id
binaries (train.bin / val.bin), memmap-style like nanoGPT, so the pretraining
dataloader can just mmap the file instead of holding tokenized text in RAM.

Usage:
    python data/prepare_pretrain.py \
        --input data/raw/*.txt \
        --tokenizer tokenizer/malayalam_tokenizer.json \
        --out-dir data/prepared \
        --val-fraction 0.001
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokenizer.special_tokens import EOS  # noqa: E402

CHUNK_LINES = 2000  # how many lines to batch-encode at a time


def iter_chunks(path, chunk_lines=CHUNK_LINES):
    buf = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            buf.append(line)
            if len(buf) >= chunk_lines:
                yield buf
                buf = []
    if buf:
        yield buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/malayalam_tokenizer.json")
    ap.add_argument("--out-dir", default="data/prepared")
    ap.add_argument("--val-fraction", type=float, default=0.001)
    args = ap.parse_args()

    files = []
    for g in args.input:
        files.extend(sorted(glob.glob(g)))
    if not files:
        raise SystemExit(f"No input files matched: {args.input}")

    tok = Tokenizer.from_file(args.tokenizer)
    eos_id = tok.token_to_id(EOS)
    if eos_id is None:
        raise SystemExit(f"Tokenizer at {args.tokenizer} has no {EOS} token -- retrain with special_tokens.py")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_path = out_dir / "_all_tokens.bin"

    total_tokens = 0
    with open(scratch_path, "wb") as out_f:
        for path in files:
            print(f"Tokenizing {path} ...")
            for chunk in iter_chunks(path):
                encodings = tok.encode_batch(chunk)
                for enc in encodings:
                    ids = enc.ids
                    if not ids:
                        continue
                    ids.append(eos_id)
                    arr = np.array(ids, dtype=np.uint16)
                    out_f.write(arr.tobytes())
                    total_tokens += len(arr)

    print(f"Total tokens: {total_tokens:,}")

    all_tokens = np.memmap(scratch_path, dtype=np.uint16, mode="r", shape=(total_tokens,))
    val_tokens = max(1, int(total_tokens * args.val_fraction))
    split_idx = total_tokens - val_tokens

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(split_idx,))[:] = all_tokens[:split_idx]
    np.memmap(val_path, dtype=np.uint16, mode="w+", shape=(val_tokens,))[:] = all_tokens[split_idx:]

    del all_tokens
    scratch_path.unlink()

    print(f"train.bin: {split_idx:,} tokens -> {train_path}")
    print(f"val.bin:   {val_tokens:,} tokens -> {val_path}")


if __name__ == "__main__":
    main()
