from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Population:
    name: str
    region: str
    allele_frequencies: Mapping[str, float]


@dataclass(frozen=True)
class TargetCoverage:
    target: str
    per_population: np.ndarray
    global_coverage: float


@dataclass(frozen=True)
class PanelSelection:
    targets: tuple[str, ...]
    per_population: np.ndarray
    global_coverage: float


def allele_locus(allele: str) -> str:
    normalized = allele.upper().replace("HLA-", "")
    if not normalized or normalized[0] not in {"A", "B", "C"}:
        raise ValueError(f"invalid class I allele: {allele}")
    return normalized[0]


class PopulationCoverage:
    def __init__(self, populations: Sequence[Population], supertype_correction: float = 0.85) -> None:
        if not populations:
            raise ValueError("at least one population is required")
        if not 0.0 <= supertype_correction <= 1.0:
            raise ValueError("supertype correction must be a probability")
        self.populations = tuple(populations)
        self.supertype_correction = supertype_correction

    def target_coverage(self, target: str, binding_alleles: Iterable[str]) -> TargetCoverage:
        alleles = tuple(dict.fromkeys(binding_alleles))
        values = np.asarray([self._population_probability(population, alleles) for population in self.populations], dtype=np.float64)
        return TargetCoverage(target, values, float(values.mean()))

    def _population_probability(self, population: Population, alleles: Sequence[str]) -> float:
        residual = 1.0
        by_locus: dict[str, list[str]] = {"A": [], "B": [], "C": []}
        for allele in alleles:
            by_locus[allele_locus(allele)].append(allele)
        for locus_alleles in by_locus.values():
            for allele in locus_alleles:
                frequency = float(population.allele_frequencies.get(allele, 0.0))
                if not 0.0 <= frequency <= 1.0:
                    raise ValueError(f"frequency outside [0, 1] for {allele}")
                residual *= (1.0 - frequency) ** 2
        return 1.0 - residual

    def greedy_panel(self, candidates: Mapping[str, Iterable[str]], panel_size: int) -> PanelSelection:
        if panel_size < 1:
            raise ValueError("panel size must be positive")
        coverage = {target: self.target_coverage(target, alleles).per_population for target, alleles in candidates.items()}
        selected: list[str] = []
        residual = np.ones(len(self.populations), dtype=np.float64)
        remaining = set(coverage)
        while remaining and len(selected) < panel_size:
            target = max(remaining, key=lambda item: float(np.sum(residual * coverage[item])))
            selected.append(target)
            residual *= 1.0 - coverage[target]
            remaining.remove(target)
        per_population = 1.0 - residual
        return PanelSelection(tuple(selected), per_population, float(per_population.mean()))

    def regional_coverage(self, panel: PanelSelection) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for population, coverage in zip(self.populations, panel.per_population, strict=True):
            grouped.setdefault(population.region, []).append(float(coverage))
        return {region: float(np.mean(values)) for region, values in grouped.items()}


def panel_coverage(target_coverages: np.ndarray) -> np.ndarray:
    if target_coverages.ndim != 2:
        raise ValueError("target coverage matrix must be two-dimensional")
    return 1.0 - np.prod(1.0 - target_coverages, axis=0)


def submodular_marginal(residual: np.ndarray, candidate: np.ndarray) -> float:
    if residual.shape != candidate.shape:
        raise ValueError("coverage vectors must have matching shapes")
    return float(np.sum(residual * candidate))
