"""Linear warmup + cosine decay LR schedule -- the same recipe as appendix-D's
"bells and whistles" bonus chapter, factored out for reuse by both the
pretraining and finetuning loops."""
import math


def lr_at_step(step: int, *, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def set_lr(optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr
