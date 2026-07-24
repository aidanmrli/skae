"""Frozen linear-probe implementation for the early-fate diagnostic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


N_CLASSES = 4


@dataclass(frozen=True)
class ProbeResult:
    alpha: float
    cv_balanced_accuracy: float
    cv_scores: tuple[float, ...]
    predictions: tuple[np.ndarray, ...]
    metrics: tuple[dict[str, float], ...]


def class_counts(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or not np.isin(values, np.arange(N_CLASSES)).all():
        raise ValueError("Labels must be a one-dimensional exact four-class encoding")
    return np.bincount(values, minlength=N_CLASSES)


def require_class_counts(labels: np.ndarray, *, minimum: int) -> np.ndarray:
    counts = class_counts(labels)
    if np.any(counts < int(minimum)):
        raise ValueError(f"Four-class count gate failed: {counts.tolist()} < {minimum}")
    return counts


def stratified_folds(labels: np.ndarray, *, n_splits: int, seed: int) -> list[np.ndarray]:
    values = np.asarray(labels, dtype=np.int64)
    require_class_counts(values, minimum=n_splits)
    rng = np.random.default_rng(int(seed))
    fold_parts: list[list[np.ndarray]] = [[] for _ in range(int(n_splits))]
    for class_index in range(N_CLASSES):
        indices = np.flatnonzero(values == class_index)
        indices = indices[rng.permutation(indices.size)]
        for fold_index, part in enumerate(np.array_split(indices, int(n_splits))):
            fold_parts[fold_index].append(part)
    folds = [np.sort(np.concatenate(parts)) for parts in fold_parts]
    all_indices = np.concatenate(folds)
    if not np.array_equal(np.sort(all_indices), np.arange(values.size)):
        raise AssertionError("Fold partition is not exhaustive and disjoint")
    for validation in folds:
        training = np.setdiff1d(np.arange(values.size), validation, assume_unique=True)
        require_class_counts(values[validation], minimum=1)
        require_class_counts(values[training], minimum=1)
    return folds


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    truth = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(predictions, dtype=np.int64)
    if truth.shape != pred.shape:
        raise ValueError("Truth/prediction shape mismatch")
    require_class_counts(truth, minimum=1)
    if not np.isin(pred, np.arange(N_CLASSES)).all():
        raise ValueError("Predictions left the frozen four-class set")
    recalls = []
    f1s = []
    for class_index in range(N_CLASSES):
        true_positive = int(np.sum((truth == class_index) & (pred == class_index)))
        false_negative = int(np.sum((truth == class_index) & (pred != class_index)))
        false_positive = int(np.sum((truth != class_index) & (pred == class_index)))
        recalls.append(true_positive / (true_positive + false_negative))
        denominator = 2 * true_positive + false_positive + false_negative
        f1s.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "accuracy": float(np.mean(truth == pred)),
    }


def _standardize(training: np.ndarray, evaluations: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    train = np.asarray(training, dtype=np.float64)
    if train.ndim != 2 or not np.isfinite(train).all():
        raise ValueError("Training features must be a finite matrix")
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized_train = (train - mean) / scale
    standardized_eval = []
    for values in evaluations:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != train.shape[1] or not np.isfinite(matrix).all():
            raise ValueError("Evaluation feature contract failed")
        standardized_eval.append((matrix - mean) / scale)
    return standardized_train, standardized_eval


def _target_matrix(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(labels, dtype=np.int64)
    targets = -np.ones((values.size, N_CLASSES), dtype=np.float64)
    targets[np.arange(values.size), values] = 1.0
    intercept = targets.mean(axis=0)
    return targets - intercept, intercept


def _dual_scores(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: list[np.ndarray],
    *,
    alpha: float,
) -> list[np.ndarray]:
    targets, intercept = _target_matrix(train_labels)
    gram = train_features @ train_features.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    projection = eigenvectors.T @ targets
    dual = eigenvectors @ (projection / (eigenvalues[:, None] + float(alpha)))
    return [values @ train_features.T @ dual + intercept for values in eval_features]


def _dual_decomposition(
    train_features: np.ndarray, train_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    targets, intercept = _target_matrix(train_labels)
    gram = train_features @ train_features.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    projection = eigenvectors.T @ targets
    return eigenvalues, eigenvectors, projection, intercept


def _scores_from_decomposition(
    cross_gram: np.ndarray,
    decomposition: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    alpha: float,
) -> np.ndarray:
    eigenvalues, eigenvectors, projection, intercept = decomposition
    dual = eigenvectors @ (projection / (eigenvalues[:, None] + float(alpha)))
    return cross_gram @ dual + intercept


def fit_probe(
    training_features: np.ndarray,
    training_labels: np.ndarray,
    test_features: list[np.ndarray],
    test_labels: list[np.ndarray],
    *,
    alphas: list[float],
    n_splits: int,
    split_seed: int,
    minimum_test_count: int,
) -> ProbeResult:
    features = np.asarray(training_features, dtype=np.float64)
    labels = np.asarray(training_labels, dtype=np.int64)
    if features.shape[0] != labels.size:
        raise ValueError("Training row mismatch")
    require_class_counts(labels, minimum=n_splits)
    if len(test_features) != len(test_labels) or not test_features:
        raise ValueError("Test feature/label roster mismatch")
    for values in test_labels:
        require_class_counts(values, minimum=minimum_test_count)
    folds = stratified_folds(labels, n_splits=n_splits, seed=split_seed)
    fold_scores_by_alpha: list[list[float]] = [[] for _ in alphas]
    for validation in folds:
        training = np.setdiff1d(np.arange(labels.size), validation, assume_unique=True)
        fold_train, [fold_validation] = _standardize(
            features[training], [features[validation]]
        )
        decomposition = _dual_decomposition(fold_train, labels[training])
        cross_gram = fold_validation @ fold_train.T
        for alpha_index, alpha in enumerate(alphas):
            scores = _scores_from_decomposition(
                cross_gram,
                decomposition,
                alpha=float(alpha),
            )
            predictions = scores.argmax(axis=1)
            fold_scores_by_alpha[alpha_index].append(
                classification_metrics(labels[validation], predictions)["balanced_accuracy"]
            )
    cv_by_alpha = [float(np.mean(values)) for values in fold_scores_by_alpha]
    maximum = max(cv_by_alpha)
    eligible = [
        index for index, value in enumerate(cv_by_alpha) if maximum - value <= 1e-12
    ]
    chosen_index = max(eligible, key=lambda index: float(alphas[index]))
    train, standardized_tests = _standardize(features, test_features)
    test_scores = _dual_scores(
        train,
        labels,
        standardized_tests,
        alpha=float(alphas[chosen_index]),
    )
    predictions = tuple(scores.argmax(axis=1).astype(np.int64) for scores in test_scores)
    metrics = tuple(
        classification_metrics(truth, pred)
        for truth, pred in zip(test_labels, predictions)
    )
    return ProbeResult(
        alpha=float(alphas[chosen_index]),
        cv_balanced_accuracy=float(cv_by_alpha[chosen_index]),
        cv_scores=tuple(cv_by_alpha),
        predictions=predictions,
        metrics=metrics,
    )
