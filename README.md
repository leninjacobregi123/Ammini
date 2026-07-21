# Malayalam LLM — a from-scratch Malayalam assistant

A completely self-built language model: own tokenizer, own architecture, own
training loop — no pretrained GPT-2/Llama weights loaded from anywhere. The
goal is a small but genuinely capable Malayalam-speaking chat assistant
(a "Jarvis, but in Malayalam"), trainable on a single consumer GPU and usable
across devices afterward.

This project lives alongside the book's chapters and deliberately mirrors
their progression, but swaps the book's plain GPT-2 clone for the
architecture family actually used by current small open models:

| Book chapter | This project |
|---|---|
| ch02 (tokenization) | `tokenizer/` — byte-level BPE trained from scratch on Malayalam text |
| ch03 (attention) | `model/layers.py` — RoPE + Grouped-Query Attention instead of learned positions + plain MHA |
| ch04 (GPT architecture) | `model/model.py`, `model/moe.py` — RMSNorm + SwiGLU + sparse Mixture-of-Experts instead of LayerNorm + GELU-MLP |
| ch05 (pretraining) | `train/pretrain.py` — pretrains on raw Malayalam text (no OpenAI weight loading, unlike the book) |
| appendix-E (LoRA) | `finetune/lora.py` — same LoRA mechanics, reused on our own model |
| ch07 (instruction tuning) | `finetune/instruction_finetune.py` — chat-template instruction tuning instead of single-turn Alpaca format |
| ch05-07 bonus UIs (Chainlit) | `app/streamlit_app.py` — a Streamlit chat UI instead |

&nbsp;

## Architecture

**MalayaLM**: a decoder-only transformer, Llama-family style:
- RMSNorm (pre-norm)
- RoPE rotary position embeddings
- Grouped-Query Attention (fewer KV heads than query heads)
- Sparse Mixture-of-Experts SwiGLU feed-forward (top-2-of-N routing, with a
  load-balancing auxiliary loss so routing doesn't collapse onto a few experts)

Two configs are provided (`configs/`), same code, different scale:
- `smoke.yaml` — ~10-15M params, runs on a CPU laptop in seconds. Exists only
  to prove the pipeline (tokenizer → model → train step → generation) is
  wired correctly before spending real time on Shannon.
- `shannon.yaml` — ~586M total params, ~190M "active" per token (top-2 of 8
  experts). Sized to comfortably fit training (weights + gradients + AdamW
  optimizer state + activations) in the RTX 5090's 32GB VRAM with headroom
  for a reasonable batch size. Scale it up further once the pipeline is
  proven end-to-end.

&nbsp;

## Honest expectations

Training a genuinely fluent LLM from absolute scratch normally takes far more
data and compute than a single GPU provides, and **more training time on a
fixed corpus does not fix this** — past a certain number of epochs the model
just memorizes the training text and generalizes worse (overfitting), it
doesn't get smarter. The public Malayalam corpora this project pulls from
(`wikimedia/wikipedia` + `ai4bharat/sangraha` + `ai4bharat/IndicCorpV2`) total
in the tens-of-GB range — a real, meaningfully-sized corpus for a language
this underserved by open models, but nowhere near the trillions of tokens
frontier models train on. Expect a small model that's clearly Malayalam-
fluent at the sentence/paragraph level and useful for everyday conversation
and instructions, not a ChatGPT-class assistant on obscure facts or complex
reasoning. That's the honest tradeoff of "100% from scratch, on one GPU."

`train/pretrain.py` tracks validation loss and stops automatically once it
plateaus (`--patience`, default 10 evals with no improvement), saving the
best-validation checkpoint as `best.pt` alongside the final one — so training
runs until it's actually converged on the data available, not for an
arbitrary fixed number of steps, and doesn't need to be babysat to avoid
overfitting late in a long run.

&nbsp;

## Running it — everything happens in Docker on Shannon

Shannon has no sudo and no host-level installs; every step below runs inside
the container. `docker` and GPU passthrough are already set up there.

**Getting the code onto Shannon:** copy this whole `malayalam-llm/` folder
over — `rsync -avz malayalam-llm/ shannon:~/malayalam-llm/`, `scp -r`, or push
this repo somewhere and `git clone`/`git pull` on Shannon, whichever you
already use to get files onto it. Nothing besides `docker`/`docker compose`
needs to exist on the host.

**Root-owned files:** since there's no sudo on Shannon, every `make` target
runs the container as *your* UID/GID (via `DOCKER_UID`/`DOCKER_GID` in the
Makefile), not root — so anything written into `data/`, `checkpoints/`,
`tokenizer/`, `.cache/` comes back out owned by you and stays deletable/
editable without sudo. `make` also creates those folders on the host before
compose does, so Docker never auto-creates them (which would make them
root-owned). You shouldn't need to fight permissions at any point; if you
ever do see a permission error, it means something wrote as root before this
was in place — `ls -la` the offending folder and let me know before deleting
anything.

```bash
# 1. build the image and confirm the RTX 5090 is visible inside the container
make build
make verify-gpu

# 2. pull Malayalam text (streamed from HF, bounded so it doesn't fill disk —
#    raise --max-mb-per-source in the Makefile once this works)
make download-data

# 3. train a byte-level BPE tokenizer on that text (from scratch, not reused
#    from anywhere — this is what makes Malayalam script-native instead of
#    an English tokenizer's afterthought)
make train-tokenizer

# 4. tokenize the corpus into train.bin/val.bin
make prepare-pretrain-data

# 5. pretrain MalayaLM from random init on raw Malayalam text -- runs until
#    val loss plateaus (early stopping), saving checkpoints/pretrain/best.pt
make pretrain

# 6. build the instruction dataset (GPTeacher-Malayalam + Alpaca-Malayalam,
#    both real public Malayalam instruction datasets) and LoRA-finetune
#    best.pt (not final.pt -- see "Honest expectations" above) into a chat assistant
make prepare-instruct-data
make instruct-finetune

# 7. chat with it
make serve   # -> http://<shannon-host>:8501
```

Before any of this, run a local smoke test on whatever machine you're editing
code on (works fine on CPU, uses `configs/smoke.yaml`):

```bash
python3 -c "
from model.config import load_config
from model.model import MalayaLM
import torch
cfg = load_config('configs/smoke.yaml', vocab_size=500)
model = MalayaLM(cfg)
x = torch.randint(0, cfg.vocab_size, (2, 32))
logits, loss = model(x, x)
loss.backward()
print('ok', logits.shape, loss.item())
"
```

&nbsp;

## Data sources (verified to exist on the HF Hub)

- Pretraining text: `wikimedia/wikipedia` (config `20231101.ml`),
  `ai4bharat/sangraha` (config `verified`, split `mal`), `ai4bharat/IndicCorpV2`
  (config `indiccorp_v2`, split `mal_Mlym`). All three stream (no full
  download) and are bounded by `--max-mb-per-source` in `data/download_corpus.py`.
- Instruction tuning: `Tensoic/GPTeacher-Malayalam` (clean
  instruction/input/output fields), `VishnuPJ/Alpaca_Instruct_Malayalam`
  (Alpaca-template text, parsed by `data/prepare_instruct.py`)
- Deliberately excluded: `oscar-corpus/OSCAR-2301`'s Malayalam subset (gated,
  needs manual HF approval) and `rajeshradhakrishnan/malayalam_news` (uses a
  legacy dataset-loading script requiring `trust_remote_code=True` — not worth
  running arbitrary remote code for a ~100MB corpus next to the three sources above).

&nbsp;

## Next steps once the base pipeline works

- DPO-style preference tuning (mirrors ch07's bonus DPO chapter) once you
  have SFT working and want to further improve response quality/style.
- KV-cache for faster generation (`model/model.py`'s `generate()` currently
  recomputes the full sequence each step — fine for a demo, not optimal).
- Quantized export (e.g. int8/int4) once you want to run the assistant on
  something other than Shannon, matching the "usable on any device" goal.
