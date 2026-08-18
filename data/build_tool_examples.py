"""Standalone dump of the synthetic tool-use instruction examples defined in
agent/tools.py, for inspection or a one-off file. data/prepare_instruct.py's
"tools" source calls agent.tools.build_tool_training_examples() directly and
merges it into the main instruct.json, so running this script by itself
isn't required for the normal pipeline -- it's here for quickly eyeballing
what the tool-use examples look like.

Usage:
    python data/build_tool_examples.py --out data/prepared/tool_examples.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.tools import build_tool_training_examples  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/prepared/tool_examples.json")
    args = ap.parse_args()

    records = build_tool_training_examples()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    print(f"Wrote {len(records)} tool-use examples -> {out_path}")


if __name__ == "__main__":
    main()
