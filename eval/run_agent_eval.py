"""Batch-eval for the tool-calling path specifically -- eval/run_eval.py only
exercises plain chat generation, never agent/orchestrator.py's tool-call/
tool-result loop, so it can't tell you whether instruct-finetuning on the
"tools" source (data/prepare_instruct.py) actually taught the model to use
<tool_call> correctly. This runs eval/agent_prompts.json through
agent.orchestrator.run_turn() for real (tools actually execute) and checks
which tool (if any) got called against each prompt's expected_tool.

Usage:
    python eval/run_agent_eval.py \
        --checkpoint checkpoints/instruct/malayalam_assistant.pt \
        --tokenizer tokenizer/malayalam_tokenizer.json \
        --prompts eval/agent_prompts.json --out eval/agent_results.jsonl
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.orchestrator import run_turn, TOOL_CALL_RE  # noqa: E402
from agent.tools import AGENT_SYSTEM_PROMPT  # noqa: E402
from model.checkpoint import load_model  # noqa: E402

from tokenizers import Tokenizer


def called_tool_name(raw_text):
    """None if no <tool_call> appeared, "<unparseable>" if it appeared but
    wasn't valid JSON, otherwise the "name" field (or "<no-name-field>")."""
    m = TOOL_CALL_RE.search(raw_text)
    if not m:
        return None
    try:
        call = json.loads(m.group(1))
    except json.JSONDecodeError:
        return "<unparseable>"
    return call.get("name", "<no-name-field>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/instruct/malayalam_assistant.pt")
    ap.add_argument("--tokenizer", default="tokenizer/malayalam_tokenizer.json")
    ap.add_argument("--prompts", default="eval/agent_prompts.json")
    ap.add_argument("--out", default="eval/agent_results.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    tok = Tokenizer.from_file(args.tokenizer)
    model, cfg, system_prompt = load_model(args.checkpoint, device=device)
    system_prompt = system_prompt or AGENT_SYSTEM_PROMPT

    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    print(f"loaded {len(prompts)} agent prompts from {args.prompts}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for entry in prompts:
            raw, display = run_turn(model, tok, cfg, system_prompt, [], entry["instruction"],
                                     args.max_new_tokens, args.temperature, args.top_k)
            got_tool = called_tool_name(raw)
            expected_tool = entry.get("expected_tool")
            match = got_tool == expected_tool

            record = {
                "id": entry["id"],
                "instruction": entry["instruction"],
                "expected_tool": expected_tool,
                "called_tool": got_tool,
                "match": match,
                "raw_response": raw,
                "display_response": display,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            correct += match
            status = "OK  " if match else "MISS"
            print(f"[{status}] {entry['id']:>20} | expected={expected_tool} got={got_tool}")

    print(f"\n{correct}/{len(prompts)} matched expected_tool -> {out_path}")
    print("A miss on a real tool call means either the wrong tool fired, the JSON didn't "
          "parse, or no tool_call appeared at all -- read raw_response in the output file "
          "to see which. A miss on an expected_tool=null prompt means the model called a "
          "tool it shouldn't have (over-triggering).")


if __name__ == "__main__":
    main()
