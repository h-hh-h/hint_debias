import numpy as np
from sklearn.metrics import roc_auc_score


def safe_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def safe_acc(y_true, y_score, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0:
        return float("nan")
    y_pred = (y_score >= threshold).astype(int)
    return float(np.mean(y_pred == y_true))


def safe_f1(y_true, y_score, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0:
        return float("nan")
    y_pred = (y_score >= threshold).astype(int)
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    den = (2.0 * tp) + fp + fn
    if den <= 0.0:
        return float("nan")
    return float((2.0 * tp) / den)


def safe_rmse(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_score) ** 2)))
