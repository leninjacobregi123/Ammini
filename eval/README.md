# Evaluation harness

Batch-generates responses for a fixed prompt suite (`prompts.json`) so the model's
qualitative behavior can be reviewed and compared across runs, instead of only chatting
one message at a time in the Streamlit app.

## Run it

```bash
make evaluate
```

or the raw form:

```bash
DOCKER_UID=$(id -u) DOCKER_GID=$(id -g) docker compose run --rm gpu python eval/run_eval.py \
    --checkpoint checkpoints/instruct/malayalam_assistant.pt \
    --tokenizer tokenizer/malayalam_tokenizer.json \
    --prompts eval/prompts.json --out eval/results.jsonl
```

Same rules as every other step in this project: rebuild (`docker compose build gpu`) first
if you've pulled code changes, since source is baked into the image, not bind-mounted.

## `prompts.json`

A list of prompt records, each tagged with a `category`:

- `factual` / `factual_simple` — knowledge recall, open-ended vs. short/checkable
- `instruction` — explain/summarize/write-style, matching the instruction-tuning data shape
- `casual` — small talk, the "Jarvis" conversational feel
- `multi_turn` — entries have a `turns` list instead of `instruction`; the harness actually
  generates a reply to each turn and feeds that real reply back in as history before the
  next turn, so this tests whether the model tracks context across a follow-up
- `edge_case` — very short/long prompts, code-switched input, questions it shouldn't know

Add more entries any time — same shape, just needs a unique `id`.

## `results.jsonl`

One JSON object per prompt: `prompt`, `response` (and `transcript` for multi-turn entries,
showing every turn), plus generation params and two automatic heuristic flags to speed up
triage —

- `repetition_flag` — a word 3-gram repeats 3+ times (the "website website website"-style
  stutter)
- `malayalam_ratio` — fraction of non-whitespace characters that are Malayalam script; low
  values mean the response leaned on English or produced garbled/non-script output

These flags are triage aids, not judgment — they catch the obvious failure modes
automatically so you don't have to read every response equally closely, but they won't
catch "fluent but hollow" content (no real information despite being grammatically fine).
Still read the actual text for that.

This is meant to be pasted back (the stdout summary, or key lines from `results.jsonl`) for
help interpreting the pattern, same as every other step in this project.
