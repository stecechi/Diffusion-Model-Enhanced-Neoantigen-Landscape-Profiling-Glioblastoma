# Diffusion Model-Enhanced Neoantigen Landscape Profiling Identifies Immunotherapy-Actionable Targets in Glioblastoma Across Diverse Populations

DiffNeoScore combines an SE(3)-equivariant flow-matching representation of peptide–MHC geometry with a protein sequence representation. Bidirectional cross-attention joins both embeddings for binding affinity, eluted-ligand presentation, and immunogenicity prediction. The downstream workflow filters germline variants, generates 8–11-mer candidates, calculates differential agretopicity, incorporates tumor expression, and selects population-aware target panels.

## Installation

Python 3.11 and CUDA 12.1 are the reference environment.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

The conda environment is created with `conda env create -f environment.yml`. The container image is built with `docker build -t diffneoscore .`.

## Data

Verified access pages, release identifiers, and access terms are collected in `datasets.txt`. Protected TCGA molecular files remain subject to GDC authorization. CGGA files are subject to the provider's sharing policy. The GLASS URL stated in the manuscript did not pass access verification and is therefore omitted from the download list.

Input variants are tab-separated with the fields `chromosome`, `position`, `reference`, `alternate`, `gene`, `protein_sequence`, `mutant_sequence`, `protein_position`, and `population_frequency`. Expression tables contain `gene` and `tpm`. AFND-derived population files are JSON records containing `name`, `region`, and an `allele_frequencies` mapping. Preprocessing uses GRCh38, VEP 110, MuTect2 mutation calls, OptiType 1.3.5 HLA calls at four-digit resolution, an IC50 threshold of 500 nM, a gnomAD v4 population-frequency cutoff of 0.01%, DAI of at least 1, and expression of at least 1 TPM.

## Model and training

The reference model uses eight invariant point-attention layers, hidden width 128, eight attention heads, 256-dimensional branch embeddings, a 512-dimensional fused embedding, ESM-2 650M representations of width 1,280, rank-four adapters in the last four sequence layers, and an immunogenicity dropout rate of 0.2. Structural embeddings are extracted at flow time 0.1 after 20 integration steps.

Training has three phases: 100 epochs of structural pretraining on 4,218 curated PDB complexes, 50 epochs of binding fine-tuning on 187,451 IEDB measurements, and 30 epochs of immunogenicity fine-tuning on 28,734 assay labels. AdamW starts at `3e-4` with weight decay `0.01` and cosine annealing; the final phase uses `1e-5`. Loss weights are `1.0`, `0.5`, `0.3`, and `0.2` for flow, binding, eluted-ligand, and immunogenicity objectives.

```bash
diffneoscore train --config configs/main.json --seed 17
```

The reported setup uses four NVIDIA A100 40 GB GPUs, batch size 32 per GPU, effective batch size 128, and five seeds. The computational-resources section reports about 12 hours for all phases, while the training paragraph and supplementary convergence table report about 14 hours. Both values are retained here because the manuscript is internally inconsistent. Peak inference memory is about 6 GB and a peptide–HLA pair takes about 0.15 seconds on one A100.

## Evaluation

Binding evaluation reports ROC AUC, AUPRC, PPV at sensitivity 0.50, and Spearman correlation. Immunogenicity evaluation reports ROC AUC, PPV and sensitivity at 95% specificity, and MCC. Alleles are grouped as common above 1,000 training peptides, intermediate from 100 through 1,000, and rare below 100. Primary AUC comparisons use two-sided DeLong tests, 10,000 bootstrap samples, and Holm–Bonferroni adjustment. Exploratory comparisons use Benjamini–Hochberg adjustment.

The manuscript's five-seed reference values are binding AUC `0.951 ± 0.004`, PPV `0.58 ± 0.03`, Spearman correlation `0.68 ± 0.02`, immunogenicity AUC `0.912 ± 0.006`, and rare-allele AUC `0.937 ± 0.006`. Dataset-specific binding AUC values are `0.946 ± 0.005` for TCGA-GBM, `0.938 ± 0.005` for CGGA, and `0.941 ± 0.005` for GLASS. These are comparison targets for trained weights and the governed source datasets; they are not asserted by untrained initialization.

## Validation

```bash
pytest -q
ruff check .
mypy --strict code/diffneoscore
```

The validation suite includes equation-level unit checks for Hardy–Weinberg coverage, target-panel integration checks, candidate-generation regression checks, actionability filtering, and statistical operating-point checks.

## License

Source code is distributed under the MIT License. Dataset licenses and controlled-access terms are independent of the source-code license.
