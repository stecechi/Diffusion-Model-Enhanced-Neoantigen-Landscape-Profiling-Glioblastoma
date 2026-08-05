from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .geometry import SE3FlowNetwork
from .sequence import ProteinSequenceEncoder


@dataclass(frozen=True)
class DiffNeoScoreConfig:
    atom_feature_dim: int = 48
    hidden_dim: int = 128
    ipa_layers: int = 8
    attention_heads: int = 8
    sequence_width: int = 1280
    embedding_dim: int = 256
    lora_rank: int = 4
    dropout: float = 0.2
    flow_time: float = 0.1
    ode_steps: int = 20


@dataclass(frozen=True)
class Prediction:
    log_ic50: Tensor
    presentation_probability: Tensor
    immunogenicity_probability: Tensor
    structural_embedding: Tensor
    sequence_embedding: Tensor
    fused_embedding: Tensor


class BidirectionalFusion(nn.Module):
    def __init__(self, dimension: int = 256, heads: int = 8) -> None:
        super().__init__()
        self.structure_to_sequence = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.sequence_to_structure = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.structure_norm = nn.LayerNorm(dimension)
        self.sequence_norm = nn.LayerNorm(dimension)

    def forward(self, structure: Tensor, sequence: Tensor) -> Tensor:
        structure_token = structure.unsqueeze(1)
        sequence_token = sequence.unsqueeze(1)
        first, _ = self.structure_to_sequence(structure_token, sequence_token, sequence_token, need_weights=False)
        second, _ = self.sequence_to_structure(sequence_token, structure_token, structure_token, need_weights=False)
        first = self.structure_norm(first + structure_token)
        second = self.sequence_norm(second + sequence_token)
        return torch.cat((first[:, 0], second[:, 0]), dim=-1)


class DiffNeoScore(nn.Module):
    def __init__(self, config: DiffNeoScoreConfig = DiffNeoScoreConfig()) -> None:
        super().__init__()
        self.config = config
        self.structural = SE3FlowNetwork(config.atom_feature_dim, config.hidden_dim, config.ipa_layers, config.attention_heads)
        self.sequence = ProteinSequenceEncoder(width=config.sequence_width, projection=config.embedding_dim, rank=config.lora_rank)
        self.fusion = BidirectionalFusion(config.embedding_dim, config.attention_heads)
        fused = config.embedding_dim * 2
        self.binding_head = nn.Sequential(nn.Linear(fused, 256), nn.GELU(), nn.Linear(256, 1))
        self.presentation_head = nn.Linear(fused, 1)
        self.immunogenicity_head = nn.Sequential(nn.Linear(fused, 256), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(256, 1))
        self.condition_embedding = nn.Embedding(25, config.hidden_dim)

    def integrate(self, coordinates: Tensor, atom_features: Tensor, allele_tokens: Tensor, atom_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        condition = self.condition_embedding(allele_tokens).mean(dim=1)
        current = coordinates
        time_step = self.config.flow_time / self.config.ode_steps
        final_features = torch.empty(0, device=coordinates.device)
        for index in range(self.config.ode_steps):
            time = torch.full((coordinates.shape[0],), index * time_step, device=coordinates.device, dtype=coordinates.dtype)
            velocity, final_features = self.structural(current, atom_features, time, condition, atom_mask)
            current = current + time_step * velocity
        return current, final_features

    def forward(self, coordinates: Tensor, atom_features: Tensor, peptide_mask: Tensor, sequence_tokens: Tensor, sequence_mask: Tensor, atom_mask: Tensor | None = None) -> Prediction:
        _, features = self.integrate(coordinates, atom_features, sequence_tokens, atom_mask)
        weights = peptide_mask.to(features.dtype).unsqueeze(-1)
        structure = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        sequence = self.sequence(sequence_tokens, sequence_mask)
        fused = self.fusion(structure, sequence)
        return Prediction(
            log_ic50=self.binding_head(fused).squeeze(-1),
            presentation_probability=torch.sigmoid(self.presentation_head(fused).squeeze(-1)),
            immunogenicity_probability=torch.sigmoid(self.immunogenicity_head(fused).squeeze(-1)),
            structural_embedding=structure,
            sequence_embedding=sequence,
            fused_embedding=fused,
        )
