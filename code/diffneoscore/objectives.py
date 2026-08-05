from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .architecture import Prediction


@dataclass(frozen=True)
class LossWeights:
    flow: float = 1.0
    binding: float = 0.5
    presentation: float = 0.3
    immunogenicity: float = 0.2


@dataclass(frozen=True)
class LossBundle:
    total: Tensor
    flow: Tensor
    binding: Tensor
    presentation: Tensor
    immunogenicity: Tensor


class MultiTaskObjective(nn.Module):
    def __init__(self, weights: LossWeights = LossWeights()) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, prediction: Prediction, predicted_velocity: Tensor, target_velocity: Tensor, log_ic50: Tensor, presentation: Tensor, immunogenicity: Tensor, binding_mask: Tensor | None = None, presentation_mask: Tensor | None = None, immunogenicity_mask: Tensor | None = None) -> LossBundle:
        flow = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)
        binding = masked_mse(prediction.log_ic50, log_ic50, binding_mask)
        presentation_loss = masked_binary_cross_entropy(prediction.presentation_probability, presentation, presentation_mask)
        immunogenicity_loss = masked_binary_cross_entropy(prediction.immunogenicity_probability, immunogenicity, immunogenicity_mask)
        total = self.weights.flow * flow + self.weights.binding * binding + self.weights.presentation * presentation_loss + self.weights.immunogenicity * immunogenicity_loss
        return LossBundle(total, flow, binding, presentation_loss, immunogenicity_loss)


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    loss = (prediction - target).square()
    if mask is None:
        return loss.mean()
    weights = mask.to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1)


def masked_binary_cross_entropy(prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    loss = torch.nn.functional.binary_cross_entropy(prediction, target, reduction="none")
    if mask is None:
        return loss.mean()
    weights = mask.to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1)


def differential_agretopicity(mutant_ic50: Tensor, wildtype_ic50: Tensor) -> Tensor:
    return torch.log2(wildtype_ic50.clamp_min(1e-8) / mutant_ic50.clamp_min(1e-8))
