"""Side-by-side diff of two eval/run_eval.py output files, matched by prompt
id -- for comparing a retrain against the pre-retrain baseline without
manually re-reading two full results.jsonl files.

Usage:
    python eval/compare_results.py \
        --before eval/results_v1_baseline.jsonl --after eval/results.jsonl
"""
import argparse
import json
from pathlib import Path


def load(path):
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec["id"]] = rec
    return records


def truncate(text, n=200):
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[: n - 3] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    args = ap.parse_args()

    before = load(args.before)
    after = load(args.after)

    ids = sorted(set(before) | set(after))
    missing_before = [i for i in ids if i not in before]
    missing_after = [i for i in ids if i not in after]
    if missing_before:
        print(f"NOTE: {len(missing_before)} id(s) only in --after (not in baseline): {missing_before}")
    if missing_after:
        print(f"NOTE: {len(missing_after)} id(s) only in --before (missing from --after): {missing_after}")

    rep_before = rep_after = low_ml_before = low_ml_after = 0
    for i in ids:
        b, a = before.get(i), after.get(i)
        if not b or not a:
            continue
        rep_before += b["repetition_flag"]
        rep_after += a["repetition_flag"]
        low_ml_before += b["malayalam_ratio"] < 0.5
        low_ml_after += a["malayalam_ratio"] < 0.5

        changed_flags = []
        if a["repetition_flag"] != b["repetition_flag"]:
            changed_flags.append(f"repetition {b['repetition_flag']}->{a['repetition_flag']}")
        if abs(a["malayalam_ratio"] - b["malayalam_ratio"]) > 0.1:
            changed_flags.append(f"malayalam_ratio {b['malayalam_ratio']}->{a['malayalam_ratio']}")
        flag_str = f"  [{', '.join(changed_flags)}]" if changed_flags else ""

        print(f"\n[{a['category']:>14}] {i}{flag_str}")
        print(f"  prompt : {truncate(a['prompt'])}")
        print(f"  before : {truncate(b['response'])}")
        print(f"  after  : {truncate(a['response'])}")

    common = len(ids) - len(missing_before) - len(missing_after)
    print(f"\n--- summary ({common} prompts compared) ---")
    print(f"repetition_flag:      before={rep_before}  after={rep_after}")
    print(f"malayalam_ratio<0.5:  before={low_ml_before}  after={low_ml_after}")
    print("(these two heuristics don't measure factual correctness -- still read the "
          "responses above for that, especially the factual_simple_* and edge_unknown_* categories)")


if __name__ == "__main__":
    main()
