from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from .population import PopulationCoverage


@dataclass(frozen=True)
class Variant:
    chromosome: str
    position: int
    reference: str
    alternate: str
    gene: str
    protein_sequence: str
    mutant_sequence: str
    protein_position: int
    population_frequency: float


@dataclass(frozen=True)
class PeptideCandidate:
    identifier: str
    gene: str
    mutant_peptide: str
    wildtype_peptide: str
    mutation_position: int
    expression_tpm: float
    prevalence: float = 0.0
    stability: float = 0.5


@dataclass(frozen=True)
class AlleleScore:
    allele: str
    mutant_ic50: float
    wildtype_ic50: float
    immunogenicity: float

    @property
    def dai(self) -> float:
        return float(np.log2(max(self.wildtype_ic50, 1e-8) / max(self.mutant_ic50, 1e-8)))


@dataclass(frozen=True)
class RankedTarget:
    candidate: PeptideCandidate
    allele_scores: tuple[AlleleScore, ...]
    global_coverage: float
    composite_score: float


@dataclass(frozen=True)
class WorkflowThresholds:
    germline_frequency: float = 0.0001
    binding_ic50: float = 500.0
    immunogenicity: float = 0.5
    high_confidence_immunogenicity: float = 0.7
    minimum_dai: float = 1.0
    minimum_expression_tpm: float = 1.0
    minimum_prevalence: float = 0.05
    treatment_resistant_stability: float = 0.8


@dataclass(frozen=True)
class CompositeWeights:
    binding: float = 0.25
    immunogenicity: float = 0.25
    coverage: float = 0.20
    recurrence: float = 0.15
    stability: float = 0.15

    def __post_init__(self) -> None:
        if not np.isclose(sum((self.binding, self.immunogenicity, self.coverage, self.recurrence, self.stability)), 1.0):
            raise ValueError("composite weights must sum to one")


def peptide_windows(sequence: str, mutant_sequence: str, mutation_index: int, lengths: Iterable[int] = (8, 9, 10, 11)) -> list[tuple[str, str, int]]:
    if len(sequence) != len(mutant_sequence):
        raise ValueError("matched protein sequences must have equal length")
    if not 0 <= mutation_index < len(sequence):
        raise ValueError("mutation index outside sequence")
    windows: list[tuple[str, str, int]] = []
    for length in lengths:
        first = max(0, mutation_index - length + 1)
        last = min(mutation_index, len(sequence) - length)
        for start in range(first, last + 1):
            windows.append((mutant_sequence[start : start + length], sequence[start : start + length], mutation_index - start))
    return windows


def candidates_from_variant(variant: Variant, expression_tpm: float) -> list[PeptideCandidate]:
    windows = peptide_windows(variant.protein_sequence, variant.mutant_sequence, variant.protein_position)
    prefix = f"{variant.chromosome}:{variant.position}:{variant.reference}>{variant.alternate}"
    return [PeptideCandidate(f"{prefix}:{len(mutant)}:{offset}", variant.gene, mutant, wildtype, offset, expression_tpm) for mutant, wildtype, offset in windows]


class NeoantigenWorkflow:
    def __init__(self, coverage: PopulationCoverage, thresholds: WorkflowThresholds = WorkflowThresholds(), weights: CompositeWeights = CompositeWeights()) -> None:
        self.coverage = coverage
        self.thresholds = thresholds
        self.weights = weights

    def germline_filter(self, variants: Iterable[Variant]) -> list[Variant]:
        return [variant for variant in variants if variant.population_frequency <= self.thresholds.germline_frequency]

    def filter_candidate(self, candidate: PeptideCandidate, scores: Sequence[AlleleScore]) -> bool:
        if candidate.expression_tpm < self.thresholds.minimum_expression_tpm:
            return False
        return any(score.mutant_ic50 < self.thresholds.binding_ic50 and score.immunogenicity > self.thresholds.immunogenicity and score.dai >= self.thresholds.minimum_dai for score in scores)

    def rank(self, candidate_scores: Mapping[PeptideCandidate, Sequence[AlleleScore]]) -> list[RankedTarget]:
        eligible = [(candidate, tuple(scores)) for candidate, scores in candidate_scores.items() if self.filter_candidate(candidate, scores)]
        if not eligible:
            return []
        affinities = np.asarray([min(score.mutant_ic50 for score in scores) for _, scores in eligible])
        raw_binding = -np.log10(affinities.clip(min=1e-8))
        span = raw_binding.max() - raw_binding.min()
        normalized = np.ones_like(raw_binding) if span == 0 else (raw_binding - raw_binding.min()) / span
        ranked: list[RankedTarget] = []
        for index, (candidate, scores) in enumerate(eligible):
            binding_alleles = [score.allele for score in scores if score.mutant_ic50 < self.thresholds.binding_ic50]
            coverage = self.coverage.target_coverage(candidate.identifier, binding_alleles).global_coverage
            immunogenicity = max(score.immunogenicity for score in scores)
            value = self.weights.binding * float(normalized[index]) + self.weights.immunogenicity * immunogenicity + self.weights.coverage * coverage + self.weights.recurrence * candidate.prevalence + self.weights.stability * candidate.stability
            ranked.append(RankedTarget(candidate, scores, coverage, value))
        return sorted(ranked, key=lambda item: item.composite_score, reverse=True)

    def annotate_longitudinal(self, candidates: Iterable[PeptideCandidate], primary_carriers: Mapping[str, int], retained_carriers: Mapping[str, int]) -> list[PeptideCandidate]:
        annotated = []
        for candidate in candidates:
            primary = primary_carriers.get(candidate.identifier, 0)
            stability = retained_carriers.get(candidate.identifier, 0) / primary if primary else 0.5
            annotated.append(replace(candidate, stability=stability))
        return annotated

    def treatment_resistant(self, target: RankedTarget) -> bool:
        candidate = target.candidate
        immunogenicity = max(score.immunogenicity for score in target.allele_scores)
        return candidate.stability >= self.thresholds.treatment_resistant_stability and candidate.prevalence >= self.thresholds.minimum_prevalence and immunogenicity > 0.6


def overlap_coefficient(first: Iterable[PeptideCandidate], second: Iterable[PeptideCandidate]) -> float:
    left = {(candidate.gene, candidate.mutant_peptide) for candidate in first}
    right = {(candidate.gene, candidate.mutant_peptide) for candidate in second}
    denominator = min(len(left), len(right))
    return len(left.intersection(right)) / denominator if denominator else 0.0


def longitudinal_fate(primary: set[str], recurrence: set[str]) -> dict[str, set[str]]:
    return {"retained": primary & recurrence, "lost": primary - recurrence, "gained": recurrence - primary}
