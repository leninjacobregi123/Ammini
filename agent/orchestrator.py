"""Interactive agent loop: chats with the instruction-tuned checkpoint like
app/streamlit_app.py and eval/run_eval.py do, but watches the model's output
for the <tool_call>{...}</tool_call> convention from agent/tools.py, executes
the matching tool, feeds the result back in as <tool_result>...</tool_result>,
and lets the model keep generating -- a minimal ReAct-style loop.

This only works well once the checkpoint's instruction-finetune data included
data/build_tool_examples.py's examples (see data/prepare_instruct.py's
"tools" source) -- run against an older checkpoint, the model simply won't
know the <tool_call> convention and will just answer in plain text.

No KV-cache reuse across generation calls, same as eval/run_eval.py -- fine
at this model size/interaction pace, not worth the complexity yet.

Usage (inside the docker container, on Shannon):
    python agent/orchestrator.py \
        --checkpoint checkpoints/instruct/malayalam_assistant.pt \
        --tokenizer tokenizer/malayalam_tokenizer.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.tools import AGENT_SYSTEM_PROMPT, execute_tool  # noqa: E402
from model.checkpoint import load_model  # noqa: E402
from tokenizer.special_tokens import format_chat, ASSISTANT, END_TURN  # noqa: E402

from tokenizers import Tokenizer

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
TOOL_RESULT_RE = re.compile(r"<tool_result>.*?</tool_result>", re.DOTALL)


def generate_once(model, tok, cfg, prompt_text, max_new_tokens, temperature, top_k, eos_id, device):
    ids = tok.encode(prompt_text).ids
    ids = ids[-(cfg.context_length - max_new_tokens):]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature,
                          top_k=top_k, eos_id=eos_id)
    new_ids = out[0, len(ids):].tolist()
    return tok.decode(new_ids)


def run_turn(model, tok, cfg, system_prompt, history, user_text,
             max_new_tokens=200, temperature=0.7, top_k=40, max_tool_calls=3):
    """Returns (raw_text, display_text). raw_text keeps the <tool_call>/
    <tool_result> markup (useful for debugging with --show-tool-calls);
    display_text has it stripped, ready to show the user and to store as
    this turn's assistant history for the next turn."""
    device = next(model.parameters()).device
    eos_id = tok.token_to_id(END_TURN)
    base_prompt = format_chat([("system", system_prompt)] + list(history) + [("user", user_text)])
    base_prompt += f"{ASSISTANT}\n"

    assistant_text = ""
    for _ in range(max_tool_calls + 1):
        chunk = generate_once(model, tok, cfg, base_prompt + assistant_text,
                               max_new_tokens, temperature, top_k, eos_id, device)
        assistant_text += chunk
        if END_TURN in assistant_text:
            assistant_text = assistant_text.split(END_TURN)[0]
            break
        m = TOOL_CALL_RE.search(chunk)
        if not m:
            break  # hit max_new_tokens without EOS or a tool call -- stop here
        try:
            call = json.loads(m.group(1))
        except json.JSONDecodeError:
            break
        result = execute_tool(call.get("name"), call.get("args"))
        assistant_text += f"\n<tool_result>{result}</tool_result>\n"

    raw_text = assistant_text.strip()
    display_text = TOOL_RESULT_RE.sub("", TOOL_CALL_RE.sub("", raw_text)).strip()
    return raw_text, display_text


_TOOL_CALL_OPEN = "<tool_call>"


def _split_safe(buffer):
    """Returns (safe_to_emit, held_back). held_back is the longest suffix of
    buffer that's still a proper prefix of "<tool_call>" -- i.e. it might be
    the start of a tool-call tag still growing token by token, so it isn't
    safe to show the user yet. Everything before that is safe to flush."""
    max_check = min(len(buffer), len(_TOOL_CALL_OPEN) - 1)
    for k in range(max_check, 0, -1):
        if _TOOL_CALL_OPEN.startswith(buffer[-k:]):
            return buffer[:-k], buffer[-k:]
    return buffer, ""


def run_turn_stream(model, tok, cfg, system_prompt, history, user_text,
                     max_new_tokens=200, temperature=0.7, top_k=40, max_tool_calls=3):
    """Generator version of run_turn() for server/app.py's SSE endpoint.
    Yields dicts as they happen:
        {"type": "token", "text": "..."}                        -- safe to show the user
        {"type": "tool_call", "name": "...", "args": {...}}      -- about to run a tool
        {"type": "tool_result", "result": "..."}                 -- what it returned
        {"type": "done", "text": "..."}                          -- final full display text

    Token text is held back from being yielded while it could still be the
    start of a forming "<tool_call>" tag (see _split_safe), so a client never
    sees tool-call markup flash on screen before being replaced by the
    "tool_call" event -- it just sees a brief pause instead.
    """
    device = next(model.parameters()).device
    eos_id = tok.token_to_id(END_TURN)
    base_prompt = format_chat([("system", system_prompt)] + list(history) + [("user", user_text)])
    base_prompt += f"{ASSISTANT}\n"

    assistant_text = ""
    display_text = ""

    for _ in range(max_tool_calls + 1):
        ids = tok.encode(base_prompt + assistant_text).ids
        ids = ids[-(cfg.context_length - max_new_tokens):]
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        buffer = ""
        match = None
        for token_id in model.generate_stream(idx, max_new_tokens=max_new_tokens,
                                               temperature=temperature, top_k=top_k, eos_id=eos_id):
            if token_id == eos_id:
                break
            piece = tok.decode([token_id])
            assistant_text += piece
            buffer += piece

            match = TOOL_CALL_RE.search(buffer)
            if match:
                break

            safe, buffer = _split_safe(buffer)
            if safe:
                display_text += safe
                yield {"type": "token", "text": safe}

        if match:
            try:
                call = json.loads(match.group(1))
            except json.JSONDecodeError:
                call = {}
            name, call_args = call.get("name"), call.get("args")
            result = execute_tool(name, call_args)
            yield {"type": "tool_call", "name": name, "args": call_args}
            yield {"type": "tool_result", "result": result}
            assistant_text += f"\n<tool_result>{result}</tool_result>\n"
            continue

        if buffer:
            display_text += buffer
            yield {"type": "token", "text": buffer}
        break

    yield {"type": "done", "text": display_text.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/instruct/malayalam_assistant.pt")
    ap.add_argument("--tokenizer", default="tokenizer/malayalam_tokenizer.json")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-tool-calls", type=int, default=3)
    ap.add_argument("--show-tool-calls", action="store_true",
                     help="also print the raw <tool_call>/<tool_result> markup, not just the final reply")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    tok = Tokenizer.from_file(args.tokenizer)
    model, cfg, system_prompt = load_model(args.checkpoint, device=device)
    system_prompt = system_prompt or AGENT_SYSTEM_PROMPT

    print("Ammini agent ready. Ctrl-C or Ctrl-D to quit.\n")
    history = []
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue

        raw, reply = run_turn(model, tok, cfg, system_prompt, history, user_text,
                               args.max_new_tokens, args.temperature, args.top_k, args.max_tool_calls)
        if args.show_tool_calls and raw != reply:
            print(f"[raw] {raw}")
        print(f"Ammini: {reply}\n")

        history.append(("user", user_text))
        history.append(("assistant", reply))


if __name__ == "__main__":
    main()
