from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AtomGraph:
    coordinates: Tensor
    features: Tensor
    peptide_mask: Tensor
    edge_index: Tensor
    edge_features: Tensor


def pairwise_displacements(coordinates: Tensor) -> tuple[Tensor, Tensor]:
    delta = coordinates.unsqueeze(-2) - coordinates.unsqueeze(-3)
    distance = torch.linalg.vector_norm(delta, dim=-1)
    direction = delta / distance.clamp_min(1e-8).unsqueeze(-1)
    return distance, direction


def radius_graph(coordinates: Tensor, cutoff: float = 8.0) -> Tensor:
    distance, _ = pairwise_displacements(coordinates)
    batch, nodes, _ = distance.shape
    valid = (distance < cutoff) & (distance > 0)
    indices = valid.nonzero(as_tuple=False)
    source = indices[:, 0] * nodes + indices[:, 1]
    target = indices[:, 0] * nodes + indices[:, 2]
    return torch.stack((source, target), dim=0)


def radial_basis(distance: Tensor, bins: int = 32, cutoff: float = 8.0) -> Tensor:
    centers = torch.linspace(0.0, cutoff, bins, device=distance.device, dtype=distance.dtype)
    width = cutoff / bins
    return torch.exp(-((distance.unsqueeze(-1) - centers) / width).square())


class VectorLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, vectors: Tensor) -> Tensor:
        return torch.einsum("oi,...ic->...oc", self.weight, vectors)


class EquivariantLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels))

    def forward(self, vectors: Tensor) -> Tensor:
        magnitude = vectors.square().sum(dim=-1).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        return vectors / magnitude.unsqueeze(-1) * self.scale.view(1, 1, -1, 1)


class InvariantPointAttention(nn.Module):
    def __init__(self, hidden_dim: int = 128, heads: int = 8, point_count: int = 4) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.point_count = point_count
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.query_points = nn.Linear(hidden_dim, heads * point_count * 3)
        self.key_points = nn.Linear(hidden_dim, heads * point_count * 3)
        self.pair_bias = nn.Linear(32, heads)
        self.point_weight = nn.Parameter(torch.ones(heads))
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, scalar: Tensor, coordinates: Tensor, pair: Tensor, mask: Tensor | None = None) -> Tensor:
        batch, nodes, _ = scalar.shape
        query = self.query(scalar).view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        key = self.key(scalar).view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        value = self.value(scalar).view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        logits = torch.einsum("bhid,bhjd->bhij", query, key) / self.head_dim**0.5
        logits = logits + self.pair_bias(pair).permute(0, 3, 1, 2)
        qp = self.query_points(scalar).view(batch, nodes, self.heads, self.point_count, 3)
        kp = self.key_points(scalar).view(batch, nodes, self.heads, self.point_count, 3)
        qp = qp + coordinates[:, :, None, None, :]
        kp = kp + coordinates[:, :, None, None, :]
        point_distance = (qp[:, :, None] - kp[:, None, :]).square().sum(dim=(-1, -2)).permute(0, 3, 1, 2)
        logits = logits - torch.nn.functional.softplus(self.point_weight).view(1, -1, 1, 1) * point_distance
        if mask is not None:
            valid = mask[:, None, :, None] & mask[:, None, None, :]
            logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        attended = torch.einsum("bhij,bhjd->bhid", attention, value)
        attended = attended.transpose(1, 2).reshape(batch, nodes, -1)
        return self.norm(scalar + self.output(attended))


class EquivariantTransition(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.scalar = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.SiLU(), nn.Linear(hidden_dim * 4, hidden_dim))
        self.coordinate_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: Tensor, coordinates: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        features = self.norm(features + self.scalar(features))
        distance, direction = pairwise_displacements(coordinates)
        gate = self.coordinate_gate(features).squeeze(-1)
        weights = torch.softmax(-distance, dim=-1) * gate.unsqueeze(-2)
        if mask is not None:
            weights = weights * mask.unsqueeze(-2)
        update = (weights.unsqueeze(-1) * direction).sum(dim=-2)
        return features, coordinates + update


class SE3FlowNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, layers: int = 8, heads: int = 8) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.time = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.condition = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.ModuleList([InvariantPointAttention(hidden_dim, heads) for _ in range(layers)])
        self.transitions = nn.ModuleList([EquivariantTransition(hidden_dim) for _ in range(layers)])
        self.velocity_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.embedding = nn.Linear(hidden_dim, 256)

    def forward(self, coordinates: Tensor, atom_features: Tensor, time: Tensor, condition: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        scalar = self.input(atom_features)
        scalar = scalar + self.time(time.reshape(-1, 1)).unsqueeze(1) + self.condition(condition).unsqueeze(1)
        distance, _ = pairwise_displacements(coordinates)
        pair = radial_basis(distance)
        moved = coordinates
        for attention, transition in zip(self.attention, self.transitions, strict=True):
            scalar = attention(scalar, moved, pair, mask)
            scalar, moved = transition(scalar, moved, mask)
        displacement = moved.unsqueeze(-2) - moved.unsqueeze(-3)
        velocity = (torch.softmax(-distance, dim=-1).unsqueeze(-1) * displacement).sum(dim=-2)
        velocity = velocity * self.velocity_gate(scalar)
        return velocity, self.embedding(scalar)


def center_coordinates(coordinates: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is None:
        return coordinates - coordinates.mean(dim=-2, keepdim=True)
    weight = mask.to(coordinates.dtype).unsqueeze(-1)
    center = (coordinates * weight).sum(dim=-2, keepdim=True) / weight.sum(dim=-2, keepdim=True).clamp_min(1)
    return (coordinates - center) * weight


def interpolate_flow(noise: Tensor, target: Tensor, time: Tensor) -> Tensor:
    while time.ndim < noise.ndim:
        time = time.unsqueeze(-1)
    return time * target + (1.0 - time) * noise


def target_velocity(noise: Tensor, target: Tensor) -> Tensor:
    return target - noise
