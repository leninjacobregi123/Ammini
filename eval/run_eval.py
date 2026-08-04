"""Batch-generate responses for a fixed prompt suite (eval/prompts.json) so the
model's qualitative behavior can be reviewed and compared across runs, instead
of only chatting with it one message at a time in app/streamlit_app.py.

Each result gets a couple of cheap automatic heuristic flags (repetition,
non-Malayalam character ratio) to help triage output at a glance -- these are
signal, not judgment; still read the actual responses.

Usage:
    python eval/run_eval.py \
        --checkpoint checkpoints/instruct/malayalam_assistant.pt \
        --tokenizer tokenizer/malayalam_tokenizer.json \
        --prompts eval/prompts.json --out eval/results.jsonl
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.checkpoint import load_model  # noqa: E402
from tokenizer.special_tokens import format_chat, ASSISTANT, END_TURN, DEFAULT_SYSTEM_PROMPT  # noqa: E402

from tokenizers import Tokenizer

# Malayalam Unicode block (U+0D00-U+0D7F) used to flag responses that lean
# heavily on English/other scripts instead of Malayalam.
MALAYALAM_RE = re.compile(r"[ഀ-ൿ]")
NON_SPACE_RE = re.compile(r"\S")


def repetition_flag(text: str, n: int = 3) -> bool:
    """True if any word n-gram repeats 3+ times -- a crude stand-in for the
    stuck-in-a-loop generation artifact small models are prone to."""
    words = text.split()
    if len(words) < n * 3:
        return False
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = {}
    for g in grams:
        counts[g] = counts.get(g, 0) + 1
        if counts[g] >= 3:
            return True
    return False


def malayalam_ratio(text: str) -> float:
    non_space = NON_SPACE_RE.findall(text)
    if not non_space:
        return 0.0
    malayalam = MALAYALAM_RE.findall(text)
    return len(malayalam) / len(non_space)


def generate_reply(model, tok, cfg, system_prompt, history, user_text, max_new_tokens, temperature, top_k):
    """history: list of (role, text) pairs already exchanged, role in
    {"user", "assistant"} -- mirrors app/streamlit_app.py::generate_reply."""
    device = next(model.parameters()).device
    turns = [("system", system_prompt)] + list(history) + [("user", user_text)]
    prompt = format_chat(turns) + f"{ASSISTANT}\n"

    ids = tok.encode(prompt).ids
    ids = ids[-(cfg.context_length - max_new_tokens):]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    eos_id = tok.token_to_id(END_TURN)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature,
                          top_k=top_k, eos_id=eos_id)
    new_ids = out[0, len(ids):].tolist()
    text = tok.decode(new_ids)
    return text.split(END_TURN)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/instruct/malayalam_assistant.pt")
    ap.add_argument("--tokenizer", default="tokenizer/malayalam_tokenizer.json")
    ap.add_argument("--prompts", default="eval/prompts.json")
    ap.add_argument("--out", default="eval/results.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    tok = Tokenizer.from_file(args.tokenizer)
    model, cfg, system_prompt = load_model(args.checkpoint, device=device)
    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    print(f"loaded {len(prompts)} prompts from {args.prompts}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for entry in prompts:
            category = entry["category"]
            if "turns" in entry:
                # Replay each turn for real, feeding the model's own actual
                # generated reply back in as history -- tests whether it
                # tracks context, rather than splicing raw prompts together.
                history = []
                transcript = []
                response = ""
                for user_text in entry["turns"]:
                    response = generate_reply(model, tok, cfg, system_prompt, history, user_text,
                                               args.max_new_tokens, args.temperature, args.top_k)
                    history.append(("user", user_text))
                    history.append(("assistant", response))
                    transcript.append({"user": user_text, "assistant": response})
                prompt_display = " -> ".join(entry["turns"])
            else:
                user_text = entry["instruction"]
                prompt_display = user_text
                transcript = None
                response = generate_reply(model, tok, cfg, system_prompt, [], user_text,
                                           args.max_new_tokens, args.temperature, args.top_k)

            record = {
                "id": entry["id"],
                "category": category,
                "prompt": prompt_display,
                "response": response,
                **({"transcript": transcript} if transcript is not None else {}),
                "temperature": args.temperature,
                "top_k": args.top_k,
                "response_len_chars": len(response),
                "malayalam_ratio": round(malayalam_ratio(response), 3),
                "repetition_flag": repetition_flag(response),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            flags = []
            if record["repetition_flag"]:
                flags.append("REPETITION")
            if record["malayalam_ratio"] < 0.5:
                flags.append("LOW-MALAYALAM")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            short_prompt = prompt_display if len(prompt_display) <= 60 else prompt_display[:57] + "..."
            print(f"[{category:>14}] {entry['id']:>16} | {short_prompt}{flag_str}")

    print(f"\nwrote {len(prompts)} results -> {out_path}")


if __name__ == "__main__":
    main()
