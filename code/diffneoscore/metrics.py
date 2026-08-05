from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn import metrics


@dataclass(frozen=True)
class BinaryMetrics:
    auc: float
    auprc: float
    ppv: float
    sensitivity: float
    specificity: float
    mcc: float


def operating_point_at_sensitivity(labels: np.ndarray, scores: np.ndarray, target: float = 0.5) -> BinaryMetrics:
    false_positive, true_positive, thresholds = metrics.roc_curve(labels, scores)
    index = int(np.argmin(np.abs(true_positive - target)))
    predictions = scores >= thresholds[index]
    return binary_metrics(labels, scores, predictions)


def operating_point_at_specificity(labels: np.ndarray, scores: np.ndarray, target: float = 0.95) -> BinaryMetrics:
    false_positive, _, thresholds = metrics.roc_curve(labels, scores)
    specificity = 1.0 - false_positive
    feasible = np.where(specificity >= target)[0]
    index = int(feasible[-1]) if len(feasible) else 0
    predictions = scores >= thresholds[index]
    return binary_metrics(labels, scores, predictions)


def binary_metrics(labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray) -> BinaryMetrics:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.int64)
    tn, fp, fn, tp = metrics.confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    ppv = tp / (tp + fp) if tp + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return BinaryMetrics(float(metrics.roc_auc_score(labels, scores)), float(metrics.average_precision_score(labels, scores)), ppv, sensitivity, specificity, float(metrics.matthews_corrcoef(labels, predictions)))


def spearman_ic50(predicted: np.ndarray, measured: np.ndarray) -> float:
    coefficient = stats.spearmanr(predicted, measured).statistic
    return float(coefficient)


def allele_group(training_count: int) -> str:
    if training_count > 1000:
        return "common"
    if training_count >= 100:
        return "intermediate"
    return "rare"


def auc_gap(common_auc: float, rare_auc: float) -> float:
    return common_auc - rare_auc


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    z = stats.norm.ppf(0.5 + confidence / 2)
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = z / denominator * np.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
    return float(center - radius), float(center + radius)


def holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)[::-1]
    adjusted = np.empty_like(values)
    running = 1.0
    count = len(values)
    for reverse_rank, index in enumerate(order):
        rank = count - reverse_rank
        running = min(running, values[index] * count / rank)
        adjusted[index] = running
    return adjusted


def bootstrap_interval(values: np.ndarray, statistic: str = "mean", resamples: int = 10000, confidence: float = 0.95, seed: int = 17) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(resamples, len(array)), replace=True)
    estimates = samples.mean(axis=1) if statistic == "mean" else np.median(samples, axis=1)
    alpha = (1.0 - confidence) / 2
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1 - alpha))


def cohens_d(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    pooled = np.sqrt(((len(left) - 1) * left.var(ddof=1) + (len(right) - 1) * right.var(ddof=1)) / (len(left) + len(right) - 2))
    return float((left.mean() - right.mean()) / pooled)
