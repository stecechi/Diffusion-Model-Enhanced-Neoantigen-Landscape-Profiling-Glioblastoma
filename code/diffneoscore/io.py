import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .population import Population
from .workflow import Variant


def read_variants(path: Path) -> list[Variant]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        return [Variant(row["chromosome"], int(row["position"]), row["reference"], row["alternate"], row["gene"], row["protein_sequence"], row["mutant_sequence"], int(row["protein_position"]), float(row.get("population_frequency", 0.0))) for row in rows]


def read_expression(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["gene"]: float(row["tpm"]) for row in csv.DictReader(stream, delimiter="\t")}


def read_populations(path: Path) -> list[Population]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return [Population(item["name"], item["region"], item["allele_frequencies"]) for item in payload]


def write_table(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
