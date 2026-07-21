"""Train a byte-level BPE tokenizer from scratch on Malayalam text.

This is the from-scratch counterpart to ch02/05_bpe-from-scratch in the book
(which builds a toy BPE tokenizer by hand to teach the algorithm) — here we
use the production `tokenizers` library to actually train a tokenizer over
raw UTF-8 bytes, which is what makes it robust to Malayalam's script without
falling back to a wall of <unk> tokens the way the book's GPT-2 (English-
centric) tokenizer would.

Usage:
    python tokenizer/train_tokenizer.py \
        --input data/raw/*.txt \
        --vocab-size 32000 \
        --out tokenizer/malayalam_tokenizer.json
"""
import argparse
import glob
import sys
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokenizer.special_tokens import SPECIAL_TOKENS  # noqa: E402


def train(input_globs, vocab_size, out_path, min_frequency=2):
    files = []
    for g in input_globs:
        files.extend(sorted(glob.glob(g)))
    if not files:
        raise SystemExit(f"No input files matched: {input_globs}")
    print(f"Training on {len(files)} file(s): {files[:5]}{' ...' if len(files) > 5 else ''}")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train(files, trainer)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    print(f"Saved tokenizer ({tokenizer.get_vocab_size()} tokens) -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True,
                     help="glob(s) of raw .txt files to train on, e.g. data/raw/*.txt")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--out", default="tokenizer/malayalam_tokenizer.json")
    args = ap.parse_args()
    train(args.input, args.vocab_size, args.out, args.min_frequency)
