import numpy as np
from diffneoscore.metrics import allele_group, benjamini_hochberg, holm_bonferroni, operating_point_at_specificity, wilson_interval


def test_statistical_metrics() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    scores = np.asarray([0.05, 0.10, 0.20, 0.70, 0.80, 0.90])
    result = operating_point_at_specificity(labels, scores)
    assert result.auc == 1.0
    assert result.specificity >= 0.95
    assert allele_group(1001) == "common"
    assert allele_group(100) == "intermediate"
    assert allele_group(99) == "rare"
    low, high = wilson_interval(58, 100)
    assert low < 0.58 < high
    values = np.asarray([0.001, 0.01, 0.04])
    assert np.all((holm_bonferroni(values) >= values) & (holm_bonferroni(values) <= 1))
    assert np.all((benjamini_hochberg(values) >= values) & (benjamini_hochberg(values) <= 1))
