from dataclasses import dataclass

import torch
from torch import Tensor, nn

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_INDEX = {residue: index + 4 for index, residue in enumerate(AMINO_ACIDS)}
PAD = 0
CLS = 1
SEP = 2
EOS = 3


@dataclass(frozen=True)
class TokenBatch:
    tokens: Tensor
    mask: Tensor


class ProteinTokenizer:
    def __init__(self, maximum_length: int = 48) -> None:
        self.maximum_length = maximum_length

    def encode(self, peptide: str, pseudo_sequence: str) -> list[int]:
        sequence = [CLS]
        sequence.extend(AA_TO_INDEX.get(residue.upper(), AA_TO_INDEX["X"]) for residue in peptide)
        sequence.append(SEP)
        sequence.extend(AA_TO_INDEX.get(residue.upper(), AA_TO_INDEX["X"]) for residue in pseudo_sequence)
        sequence.append(EOS)
        if len(sequence) > self.maximum_length:
            raise ValueError(f"combined sequence exceeds {self.maximum_length} tokens")
        return sequence

    def batch(self, pairs: list[tuple[str, str]], device: torch.device | None = None) -> TokenBatch:
        encoded = [self.encode(peptide, allele) for peptide, allele in pairs]
        longest = max(map(len, encoded))
        tokens = torch.full((len(encoded), longest), PAD, dtype=torch.long, device=device)
        for index, row in enumerate(encoded):
            tokens[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        return TokenBatch(tokens=tokens, mask=tokens.ne(PAD))


class LowRankLinear(nn.Module):
    def __init__(self, source: nn.Linear, rank: int = 4, scale: float = 1.0) -> None:
        super().__init__()
        self.source = source
        self.down = nn.Linear(source.in_features, rank, bias=False)
        self.up = nn.Linear(rank, source.out_features, bias=False)
        self.scale = scale / rank
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)
        for parameter in self.source.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: Tensor) -> Tensor:
        return self.source(inputs) + self.up(self.down(inputs)) * self.scale


class SequenceLayer(nn.Module):
    def __init__(self, width: int, heads: int, rank: int = 4) -> None:
        super().__init__()
        attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.attention = attention
        self.query_adapter = LowRankLinear(nn.Linear(width, width), rank)
        self.key_adapter = LowRankLinear(nn.Linear(width, width), rank)
        self.value_adapter = LowRankLinear(nn.Linear(width, width), rank)
        self.ffn = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:
        query = self.query_adapter(inputs)
        key = self.key_adapter(inputs)
        value = self.value_adapter(inputs)
        attended, _ = self.attention(query, key, value, key_padding_mask=~mask, need_weights=False)
        hidden = self.norm1(inputs + attended)
        return self.norm2(hidden + self.ffn(hidden))


class ProteinSequenceEncoder(nn.Module):
    def __init__(self, vocabulary: int = 25, width: int = 1280, projection: int = 256, layers: int = 4, heads: int = 8, rank: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, width, padding_idx=PAD)
        self.position = nn.Embedding(64, width)
        self.layers = nn.ModuleList([SequenceLayer(width, heads, rank) for _ in range(layers)])
        self.projection = nn.Sequential(nn.Linear(width, 512), nn.GELU(), nn.Linear(512, projection))

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        hidden = self.embedding(tokens) + self.position(positions)
        for layer in self.layers:
            hidden = layer(hidden, mask)
        return self.projection(hidden[:, 0])


def validate_peptide(peptide: str) -> str:
    normalized = peptide.strip().upper()
    if len(normalized) not in {8, 9, 10, 11}:
        raise ValueError("peptide length must be between 8 and 11")
    invalid = set(normalized).difference(AMINO_ACIDS)
    if invalid:
        raise ValueError(f"unsupported residues: {sorted(invalid)}")
    return normalized


def validate_pseudo_sequence(sequence: str) -> str:
    normalized = sequence.strip().upper()
    if len(normalized) != 34:
        raise ValueError("HLA pseudo-sequence must contain 34 residues")
    invalid = set(normalized).difference(AMINO_ACIDS)
    if invalid:
        raise ValueError(f"unsupported residues: {sorted(invalid)}")
    return normalized
