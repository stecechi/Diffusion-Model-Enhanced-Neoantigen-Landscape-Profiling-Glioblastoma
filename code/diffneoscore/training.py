from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor, nn


class Phase(str, Enum):
    STRUCTURAL = "structural"
    BINDING = "binding"
    IMMUNOGENICITY = "immunogenicity"


@dataclass(frozen=True)
class PhaseConfig:
    phase: Phase
    epochs: int
    learning_rate: float
    flow: bool
    binding: bool
    presentation: bool
    immunogenicity: bool


PAPER_PHASES = (
    PhaseConfig(Phase.STRUCTURAL, 100, 3e-4, True, False, False, False),
    PhaseConfig(Phase.BINDING, 50, 3e-4, True, True, True, False),
    PhaseConfig(Phase.IMMUNOGENICITY, 30, 1e-5, True, True, True, True),
)


def adamw(model: nn.Module, learning_rate: float, weight_decay: float = 0.01) -> torch.optim.AdamW:
    return torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=learning_rate, weight_decay=weight_decay)


def cosine_scheduler(optimizer: torch.optim.Optimizer, epochs: int, steps_per_epoch: int) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)


def distributed_mean(value: Tensor) -> Tensor:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return value
    result = value.detach().clone()
    torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
    return result / torch.distributed.get_world_size()


def gradient_norm(parameters: Iterable[nn.Parameter]) -> Tensor:
    norms = [parameter.grad.detach().norm(2) for parameter in parameters if parameter.grad is not None]
    if not norms:
        return torch.tensor(0.0)
    return torch.stack(norms).norm(2)


class Trainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, precision: str = "bf16", gradient_accumulation: int = 1, gradient_clip: float | None = None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.precision = precision
        self.gradient_accumulation = gradient_accumulation
        self.gradient_clip = gradient_clip
        self.step = 0

    def backward(self, loss: Tensor) -> None:
        (loss / self.gradient_accumulation).backward()
        self.step += 1
        if self.step % self.gradient_accumulation:
            return
        if self.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
