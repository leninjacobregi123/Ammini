.PHONY: init-dirs build verify-gpu shell shell-gpu download-data train-tokenizer prepare-pretrain-data \
        prepare-instruct-data pretrain instruct-finetune evaluate evaluate-agent serve agent

# Run every docker compose invocation as your own host user/group instead of
# root -- required since Shannon has no sudo, so anything the container
# writes into the bind-mounted data/tokenizer/checkpoints/.cache folders must
# come back out owned by you, not root.
COMPOSE = DOCKER_UID=$$(id -u) DOCKER_GID=$$(id -g) docker compose

# Create bind-mount targets on the host *before* compose does, so Docker
# never auto-creates them (which would make them root-owned). Run via a
# throwaway container (as your own UID, not root) rather than the host's own
# mkdir, so this step goes through Docker too.
init-dirs:
	docker run --rm -v "$$(pwd):/repo" -w /repo --user "$$(id -u):$$(id -g)" alpine \
		mkdir -p data/raw data/prepared tokenizer checkpoints/pretrain checkpoints/instruct .cache agent/state

build: init-dirs
	$(COMPOSE) build

# Only this target actually needs a GPU reservation -- everything else below
# uses the `cpu` service (no deploy/GPU block at all), so data prep can run
# regardless of GPU availability/policy, and only pretrain/instruct-finetune/
# serve ever touch the GPU-authorization question.
verify-gpu: init-dirs
	$(COMPOSE) run --rm gpu bash scripts/verify_gpu.sh

shell: init-dirs
	$(COMPOSE) run --rm cpu bash

shell-gpu: init-dirs
	$(COMPOSE) run --rm gpu bash

# ---- data (CPU-only, no GPU reservation) ----
# HF_INSECURE_SSL=1: see data/_ssl_workaround.py -- Shannon's campus network
# intercepts some Hugging Face CDN hosts with a Fortinet cert that nothing on
# Shannon (host or container) trusts, and the Docker authorization policy
# here blocks the bind-mount that would otherwise fix trust properly. Scoped
# to these two HF-fetching targets only.
download-data: init-dirs
	$(COMPOSE) run --rm -e HF_INSECURE_SSL=1 cpu python data/download_corpus.py --out-dir data/raw --max-mb-per-source 2000

prepare-pretrain-data: init-dirs
	$(COMPOSE) run --rm cpu python data/prepare_pretrain.py \
		--input 'data/raw/*.txt' --tokenizer tokenizer/malayalam_tokenizer.json --out-dir data/prepared

prepare-instruct-data: init-dirs
	$(COMPOSE) run --rm -e HF_INSECURE_SSL=1 cpu python data/prepare_instruct.py --out data/prepared/instruct.json

# ---- tokenizer (CPU-only, no GPU reservation) ----
train-tokenizer: init-dirs
	$(COMPOSE) run --rm cpu python tokenizer/train_tokenizer.py \
		--input 'data/raw/*.txt' --vocab-size 32000 --out tokenizer/malayalam_tokenizer.json

# ---- training (GPU) ----
pretrain: init-dirs
	$(COMPOSE) run --rm gpu python train/pretrain.py \
		--config configs/shannon.yaml --tokenizer tokenizer/malayalam_tokenizer.json \
		--data-dir data/prepared --out-dir checkpoints/pretrain

instruct-finetune: init-dirs
	$(COMPOSE) run --rm gpu python finetune/instruction_finetune.py \
		--pretrained checkpoints/pretrain/best.pt --tokenizer tokenizer/malayalam_tokenizer.json \
		--data data/prepared/instruct.json --out checkpoints/instruct/malayalam_assistant.pt

# eval/run_eval.py has no KV cache (same as model/model.py::generate used by
# the app), so generation is slow enough on CPU to keep this on the gpu
# service like training/serving, even though it's inference-only.
evaluate: init-dirs
	$(COMPOSE) run --rm gpu python eval/run_eval.py \
		--checkpoint checkpoints/instruct/malayalam_assistant.pt \
		--tokenizer tokenizer/malayalam_tokenizer.json \
		--prompts eval/prompts.json --out eval/results.jsonl

# eval/run_eval.py never exercises the <tool_call> loop -- this runs
# eval/agent_prompts.json through agent/orchestrator.py's real tool-execution
# path and checks the called tool against each prompt's expected_tool.
evaluate-agent: init-dirs
	$(COMPOSE) run --rm gpu python eval/run_agent_eval.py \
		--checkpoint checkpoints/instruct/malayalam_assistant.pt \
		--tokenizer tokenizer/malayalam_tokenizer.json \
		--prompts eval/agent_prompts.json --out eval/agent_results.jsonl

# ---- serving (GPU) ----
serve: init-dirs
	$(COMPOSE) up app

# Interactive agent loop (tool-calling ReAct-style, see agent/orchestrator.py
# and agent/tools.py) -- only useful once instruct-finetune ran on data that
# included the "tools" source from prepare-instruct-data.
agent: init-dirs
	$(COMPOSE) run --rm gpu python agent/orchestrator.py \
		--checkpoint checkpoints/instruct/malayalam_assistant.pt \
		--tokenizer tokenizer/malayalam_tokenizer.json
