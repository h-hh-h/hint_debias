from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier


def fit_propensity(X: np.ndarray, A: np.ndarray, model: str = "lr"):
    model = model.lower()
    if model == "lr":
        m = LogisticRegression(max_iter=200, solver="liblinear")
    elif model == "gbdt":
        m = GradientBoostingClassifier()
    elif model == "mlp":
        m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200)
    else:
        raise ValueError(f"Unknown propensity model: {model}")
    m.fit(X, A)
    return m


def fit_outcome(X: np.ndarray, y: np.ndarray, model: str = "gbdt"):
    model = model.lower()
    if model == "gbdt":
        m = GradientBoostingClassifier()
    elif model == "lr":
        m = LogisticRegression(max_iter=300, solver="liblinear")
    elif model == "mlp":
        m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200)
    else:
        raise ValueError(f"Unknown outcome model: {model}")
    m.fit(X, y)
    return m


def crossfit_dr_pseudolabel(
    X: pd.DataFrame,
    A: np.ndarray,
    y: np.ndarray,
    kfold: int = 5,
    eps: float = 0.05,
    clip01: bool = True,
    seed: int = 42,
    outcome_model: str = "gbdt",
    propensity_model: str = "lr",
    return_raw_e: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      y0_tilde: DR pseudo-label for Y(0)
      e_hat: propensity P(A=1|X)
      m0_hat: outcome model for A=0
      m1_hat: outcome model for A=1 (not needed for y0_tilde but returned for completeness)
    """
    X_use = X.drop(columns=["A_hint"], errors="ignore").values.astype(float)
    n = len(y)

    e_hat_raw = np.zeros(n, dtype=float)
    m0_hat = np.zeros(n, dtype=float)
    m1_hat = np.zeros(n, dtype=float)

    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
    for tr, te in kf.split(np.arange(n)):
        X_tr, X_te = X_use[tr], X_use[te]
        A_tr, y_tr = A[tr], y[tr]

        # propensity
        g = fit_propensity(X_tr, A_tr, model=propensity_model)
        e_hat_raw[te] = g.predict_proba(X_te)[:, 1]

        # outcome m0/m1 (train on strata; fallback if too small)
        mask0 = A_tr == 0
        mask1 = A_tr == 1

        if mask0.sum() >= 200:
            f0 = fit_outcome(X_tr[mask0], y_tr[mask0], model=outcome_model)
            m0_hat[te] = f0.predict_proba(X_te)[:, 1]
        else:
            # fallback: fit on all
            f0 = fit_outcome(X_tr, y_tr, model=outcome_model)
            m0_hat[te] = f0.predict_proba(X_te)[:, 1]

        if mask1.sum() >= 200:
            f1 = fit_outcome(X_tr[mask1], y_tr[mask1], model=outcome_model)
            m1_hat[te] = f1.predict_proba(X_te)[:, 1]
        else:
            f1 = fit_outcome(X_tr, y_tr, model=outcome_model)
            m1_hat[te] = f1.predict_proba(X_te)[:, 1]

    # stabilize
    e_hat = np.clip(e_hat_raw, eps, 1 - eps)

    # DR(AIPW) for Y(0):
    # y0 = m0 + (1-A)*(y-m0)/(1-e)
    y0_tilde = m0_hat + ((1 - A) * (y - m0_hat)) / (1 - e_hat)

    if clip01:
        y0_tilde = np.clip(y0_tilde, 0.0, 1.0)

    if return_raw_e:
        return y0_tilde.astype(float), e_hat, m0_hat, m1_hat, e_hat_raw.astype(float)
    return y0_tilde.astype(float), e_hat, m0_hat, m1_hat


def dr_pseudolabel_no_crossfit(
    X: pd.DataFrame,
    A: np.ndarray,
    y: np.ndarray,
    eps: float = 0.05,
    clip01: bool = True,
    outcome_model: str = "gbdt",
    propensity_model: str = "lr",
    return_raw_e: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Non-cross-fitted DR pseudo-labels:
    fit propensity/outcome on all train data, then predict on same data.
    """
    X_use = X.drop(columns=["A_hint"], errors="ignore").values.astype(float)

    g = fit_propensity(X_use, A, model=propensity_model)
    e_hat_raw = g.predict_proba(X_use)[:, 1]
    e_hat = np.clip(e_hat_raw, eps, 1 - eps)

    mask0 = A == 0
    mask1 = A == 1

    if mask0.sum() >= 200:
        f0 = fit_outcome(X_use[mask0], y[mask0], model=outcome_model)
        m0_hat = f0.predict_proba(X_use)[:, 1]
    else:
        f0 = fit_outcome(X_use, y, model=outcome_model)
        m0_hat = f0.predict_proba(X_use)[:, 1]

    if mask1.sum() >= 200:
        f1 = fit_outcome(X_use[mask1], y[mask1], model=outcome_model)
        m1_hat = f1.predict_proba(X_use)[:, 1]
    else:
        f1 = fit_outcome(X_use, y, model=outcome_model)
        m1_hat = f1.predict_proba(X_use)[:, 1]

    y0_tilde = m0_hat + ((1 - A) * (y - m0_hat)) / (1 - e_hat)
    if clip01:
        y0_tilde = np.clip(y0_tilde, 0.0, 1.0)

    if return_raw_e:
        return y0_tilde.astype(float), e_hat.astype(float), m0_hat.astype(float), m1_hat.astype(float), e_hat_raw.astype(float)
    return y0_tilde.astype(float), e_hat.astype(float), m0_hat.astype(float), m1_hat.astype(float)
