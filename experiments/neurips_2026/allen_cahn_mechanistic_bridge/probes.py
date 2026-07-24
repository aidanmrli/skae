"""Identical nested-CV linear ridge probes for all frozen feature sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class Standardization:
    mean: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class FittedRidge:
    classes: np.ndarray
    scaled_train: np.ndarray
    transform: Standardization
    coefficients: np.ndarray
    target_mean: np.ndarray
    selected_alpha: float


def standardize_fit(values: np.ndarray) -> tuple[np.ndarray, Standardization]:
    array = np.asarray(values, dtype=np.float64)
    mean = array.mean(axis=0)
    scale = array.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (array - mean) / scale, Standardization(mean, scale)


def standardize_apply(values: np.ndarray, transform: Standardization) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - transform.mean) / transform.scale


def _one_hot(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    mapping = {int(value): index for index, value in enumerate(classes)}
    result = np.zeros((labels.size, classes.size), dtype=np.float64)
    result[np.arange(labels.size), [mapping[int(value)] for value in labels]] = 1.0
    return result


def _dual_predictions(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float,
    classes: np.ndarray,
) -> np.ndarray:
    scaled_train, transform = standardize_fit(train_x)
    scaled_test = standardize_apply(test_x, transform)
    targets = _one_hot(train_y, classes)
    target_mean = targets.mean(axis=0, keepdims=True)
    centered = targets - target_mean
    gram = scaled_train @ scaled_train.T
    coefficients = np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0]), centered
    )
    scores = scaled_test @ scaled_train.T @ coefficients + target_mean
    return classes[np.argmax(scores, axis=1)]


def _fit_dual(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    alpha: float,
    classes: np.ndarray,
) -> FittedRidge:
    scaled, transform = standardize_fit(train_x)
    targets = _one_hot(train_y, classes)
    target_mean = targets.mean(axis=0, keepdims=True)
    centered = targets - target_mean
    gram = scaled @ scaled.T
    coefficients = np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0]), centered
    )
    return FittedRidge(
        classes=classes,
        scaled_train=scaled,
        transform=transform,
        coefficients=coefficients,
        target_mean=target_mean,
        selected_alpha=float(alpha),
    )


def predict_fitted(model: FittedRidge, features: np.ndarray) -> np.ndarray:
    scaled = standardize_apply(features, model.transform)
    scores = scaled @ model.scaled_train.T @ model.coefficients + model.target_mean
    return model.classes[np.argmax(scores, axis=1)]


def classification_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(truth, predicted)),
    }


def _select_alpha(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    alphas: Iterable[float],
    folds: int,
    seed: int,
    classes: np.ndarray,
) -> tuple[float, dict[str, float]]:
    candidates = sorted(float(value) for value in alphas)
    totals = {alpha: [] for alpha in candidates}
    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    for train_index, score_index in splitter.split(features, labels):
        train_x, transform = standardize_fit(features[train_index])
        score_x = standardize_apply(features[score_index], transform)
        targets = _one_hot(labels[train_index], classes)
        target_mean = targets.mean(axis=0, keepdims=True)
        centered = targets - target_mean
        gram = train_x @ train_x.T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        projected = eigenvectors.T @ centered
        cross = score_x @ train_x.T @ eigenvectors
        for alpha in candidates:
            scores = cross @ (projected / (eigenvalues[:, None] + alpha)) + target_mean
            prediction = classes[np.argmax(scores, axis=1)]
            totals[alpha].append(
                float(balanced_accuracy_score(labels[score_index], prediction))
            )
    means = {str(alpha): float(np.mean(values)) for alpha, values in totals.items()}
    best_score = max(means.values())
    selected = max(float(alpha) for alpha in candidates if means[str(alpha)] >= best_score - 1e-12)
    return selected, means


def fit_nested_ridge(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    alphas: Iterable[float],
    outer_folds: int,
    inner_folds: int,
    final_folds: int,
    seed: int,
) -> tuple[FittedRidge, dict[str, Any]]:
    train_x = np.asarray(train_features, dtype=np.float64)
    train_y = np.asarray(train_labels, dtype=np.int64)
    if train_x.ndim != 2 or train_x.shape[0] != train_y.size:
        raise ValueError("Probe features and labels are unpaired")
    if not np.isfinite(train_x).all():
        raise FloatingPointError("Probe features contain nonfinite values")
    classes = np.unique(train_y)
    if classes.size != 4 or not np.array_equal(classes, np.arange(4)):
        raise ValueError("Frozen Allen--Cahn probe requires all four train fates")

    outer = StratifiedKFold(
        n_splits=int(outer_folds), shuffle=True, random_state=int(seed)
    )
    oof = np.empty_like(train_y)
    outer_alphas: list[float] = []
    for fold, (fit_index, score_index) in enumerate(outer.split(train_x, train_y)):
        selected, _ = _select_alpha(
            train_x[fit_index],
            train_y[fit_index],
            alphas=alphas,
            folds=inner_folds,
            seed=int(seed) + 100 + fold,
            classes=classes,
        )
        outer_alphas.append(selected)
        oof[score_index] = _dual_predictions(
            train_x[fit_index],
            train_y[fit_index],
            train_x[score_index],
            alpha=selected,
            classes=classes,
        )
    selected, final_scores = _select_alpha(
        train_x,
        train_y,
        alphas=alphas,
        folds=final_folds,
        seed=int(seed) + 1000,
        classes=classes,
    )
    fitted = _fit_dual(train_x, train_y, alpha=selected, classes=classes)
    audit = {
        "selected_alpha": selected,
        "outer_selected_alphas": outer_alphas,
        "final_cv_balanced_accuracy_by_alpha": final_scores,
        "outer_oof": classification_metrics(train_y, oof),
        "train_rows": int(train_x.shape[0]),
        "feature_dim": int(train_x.shape[1]),
    }
    return fitted, audit


def score_fitted(
    model: FittedRidge, test_features: np.ndarray, test_labels: np.ndarray
) -> dict[str, Any]:
    test_x = np.asarray(test_features, dtype=np.float64)
    test_y = np.asarray(test_labels, dtype=np.int64)
    if test_x.ndim != 2 or test_x.shape[0] != test_y.size:
        raise ValueError("Test features and labels are unpaired")
    if test_x.shape[1] != model.transform.mean.size or not np.isfinite(test_x).all():
        raise ValueError("Test feature dimension/finite contract failed")
    prediction = predict_fitted(model, test_x)
    return {
        "test": classification_metrics(test_y, prediction),
        "test_predictions": prediction.tolist(),
        "test_rows": int(test_x.shape[0]),
    }


def nested_ridge_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    alphas: Iterable[float],
    outer_folds: int,
    inner_folds: int,
    final_folds: int,
    seed: int,
) -> dict[str, Any]:
    fitted, audit = fit_nested_ridge(
        train_features,
        train_labels,
        alphas=alphas,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        final_folds=final_folds,
        seed=seed,
    )
    return {**audit, **score_fitted(fitted, test_features, test_labels)}
