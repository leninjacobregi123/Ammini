.PHONY: init-dirs build verify-gpu shell shell-gpu download-data train-tokenizer prepare-pretrain-data \
        prepare-instruct-data pretrain instruct-finetune serve

# Run every docker compose invocation as your own host user/group instead of
# root -- required since Shannon has no sudo, so anything the container
# writes into the bind-mounted data/tokenizer/checkpoints/.cache folders must
# come back out owned by you, not root.
COMPOSE = DOCKER_UID=$$(id -u) DOCKER_GID=$$(id -g) docker compose

# Create bind-mount targets on the host *before* compose does, so Docker
# never auto-creates them (which would make them root-owned).
init-dirs:
	mkdir -p data/raw data/prepared tokenizer checkpoints/pretrain checkpoints/instruct .cache

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
download-data: init-dirs
	$(COMPOSE) run --rm cpu python data/download_corpus.py --out-dir data/raw --max-mb-per-source 2000

prepare-pretrain-data: init-dirs
	$(COMPOSE) run --rm cpu python data/prepare_pretrain.py \
		--input 'data/raw/*.txt' --tokenizer tokenizer/malayalam_tokenizer.json --out-dir data/prepared

prepare-instruct-data: init-dirs
	$(COMPOSE) run --rm cpu python data/prepare_instruct.py --out data/prepared/instruct.json

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

# ---- serving (GPU) ----
serve: init-dirs
	$(COMPOSE) up app
