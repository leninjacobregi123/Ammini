"""Build a unified Malayalam instruction dataset from verified public sources,
normalized to the same {instruction, input, output} shape the book's ch07
uses -- the chat template (system/user/assistant) is applied later, at
finetuning time, by finetune/instruction_finetune.py.

Sources (verified to exist on the HF Hub before writing this script):
  - Tensoic/GPTeacher-Malayalam       -- clean instruction/input/output fields
  - VishnuPJ/Alpaca_Instruct_Malayalam -- single combined "Prompt" field
    (Malayalam Alpaca template with English "### Instruction:" / "### Input:"
    / "### Response:" section markers), parsed out below.

Usage:
    python data/prepare_instruct.py --out data/prepared/instruct.json
"""
import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

ALPACA_RE = re.compile(
    r"###\s*Instruction:\s*\n(?P<instruction>.*?)\n\n###\s*Input:\s*\n(?P<input>.*?)\n\n###\s*Response:\s*\n(?P<output>.*)",
    re.DOTALL,
)


def load_gpteacher():
    ds = load_dataset("Tensoic/GPTeacher-Malayalam", split="train")
    records = []
    for row in ds:
        instruction = (row.get("instruction") or "").strip()
        output = (row.get("output") or "").strip()
        if not instruction or not output:
            continue
        records.append({
            "instruction": instruction,
            "input": (row.get("input") or "").strip(),
            "output": output,
            "source": "GPTeacher-Malayalam",
        })
    print(f"GPTeacher-Malayalam: {len(records)} usable records")
    return records


def load_alpaca_malayalam():
    ds = load_dataset("VishnuPJ/Alpaca_Instruct_Malayalam", split="train")
    records = []
    skipped = 0
    for row in ds:
        prompt = row.get("Prompt") or ""
        m = ALPACA_RE.search(prompt)
        if not m:
            skipped += 1
            continue
        instruction = m.group("instruction").strip()
        input_text = m.group("input").strip()
        output = m.group("output").strip()
        if not instruction or not output:
            skipped += 1
            continue
        records.append({
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "source": "Alpaca_Instruct_Malayalam",
        })
    print(f"Alpaca_Instruct_Malayalam: {len(records)} usable records, {skipped} skipped (unparseable)")
    return records


LOADERS = {
    "gpteacher": load_gpteacher,
    "alpaca": load_alpaca_malayalam,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", choices=list(LOADERS), default=["gpteacher", "alpaca"])
    ap.add_argument("--out", default="data/prepared/instruct.json")
    ap.add_argument("--max-per-source", type=int, default=None)
    args = ap.parse_args()

    all_records = []
    for name in args.sources:
        records = LOADERS[name]()
        if args.max_per_source:
            records = records[: args.max_per_source]
        all_records.extend(records)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=1)

    print(f"Wrote {len(all_records)} total instruction records -> {out_path}")


if __name__ == "__main__":
    main()
