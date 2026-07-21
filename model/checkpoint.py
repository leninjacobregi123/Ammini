"""Single source of truth for loading a MalayaLM checkpoint, whether it's a
plain pretrained checkpoint (train/pretrain.py) or a LoRA instruction-tuned
one (finetune/instruction_finetune.py) -- the latter needs the exact same
LinearWithLoRA wrapping re-applied before load_state_dict, since that's what
changed the parameter names when it was saved.
"""
import torch

from model.config import MalayaLMConfig
from model.model import MalayaLM
from finetune.lora import replace_linear_with_lora


def load_model(ckpt_path: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    cfg = MalayaLMConfig(**ckpt["config"])
    model = MalayaLM(cfg)

    if "lora_rank" in ckpt:
        replace_linear_with_lora(model, ckpt["lora_rank"], ckpt["lora_alpha"])

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    system_prompt = ckpt.get("system_prompt")
    return model, cfg, system_prompt
