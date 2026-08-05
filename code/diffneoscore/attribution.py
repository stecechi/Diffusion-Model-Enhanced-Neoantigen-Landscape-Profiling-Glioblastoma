from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class StructuralAttribution:
    atom_scores: Tensor
    tcr_surface_score: Tensor
    backbone_entropy: Tensor
    complex_stability: Tensor


def integrated_gradients(function: Callable[[Tensor], Tensor], inputs: Tensor, baseline: Tensor | None = None, steps: int = 64) -> Tensor:
    origin = torch.zeros_like(inputs) if baseline is None else baseline
    accumulated = torch.zeros_like(inputs)
    for alpha in torch.linspace(0.0, 1.0, steps, device=inputs.device, dtype=inputs.dtype):
        point = (origin + alpha * (inputs - origin)).detach().requires_grad_(True)
        output = function(point).sum()
        gradient = torch.autograd.grad(output, point)[0]
        accumulated = accumulated + gradient
    return (inputs - origin) * accumulated / steps


def backbone_dihedral(points: Tensor) -> Tensor:
    first = points[..., 1:-2, :] - points[..., :-3, :]
    second = points[..., 2:-1, :] - points[..., 1:-2, :]
    third = points[..., 3:, :] - points[..., 2:-1, :]
    normal1 = torch.linalg.cross(first, second, dim=-1)
    normal2 = torch.linalg.cross(second, third, dim=-1)
    normal1 = torch.nn.functional.normalize(normal1, dim=-1)
    normal2 = torch.nn.functional.normalize(normal2, dim=-1)
    tangent = torch.linalg.cross(normal1, torch.nn.functional.normalize(second, dim=-1), dim=-1)
    return torch.atan2((tangent * normal2).sum(-1), (normal1 * normal2).sum(-1))


def trajectory_entropy(trajectory: Tensor) -> Tensor:
    angles = backbone_dihedral(trajectory)
    return angles.var(dim=0, unbiased=False).mean(dim=-1)


def solvent_accessible_proxy(coordinates: Tensor, residue_mask: Tensor, radius: float = 6.0) -> Tensor:
    distances = torch.cdist(coordinates, coordinates)
    neighbors = ((distances < radius) & (distances > 0)).sum(dim=-1)
    exposure = torch.exp(-neighbors.to(coordinates.dtype) / 10.0)
    weights = residue_mask.to(exposure.dtype)
    return (exposure * weights).sum(-1) / weights.sum(-1).clamp_min(1)


def summarize_attribution(atom_scores: Tensor, coordinates: Tensor, tcr_mask: Tensor, trajectory: Tensor, complex_stability: Tensor) -> StructuralAttribution:
    magnitude = torch.linalg.vector_norm(atom_scores, dim=-1)
    surface = solvent_accessible_proxy(coordinates, tcr_mask)
    entropy = trajectory_entropy(trajectory)
    return StructuralAttribution(magnitude, surface, entropy, complex_stability)
