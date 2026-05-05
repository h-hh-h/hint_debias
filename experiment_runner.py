import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .cache import cache_key, cache_paths, load_cache, save_cache
from .cd_models import finetune_cd, predict_cd, train_and_eval_cd
from .columns import build_treatment_A, infer_cols
from .config import DEFAULT_DATASETS
from .data_io import ensure_downloaded, pick_main_csv
from .diagnostics import run_step1_diagnostics
from .dr import crossfit_dr_pseudolabel, dr_pseudolabel_no_crossfit, fit_propensity
from .features import make_features, temporal_split_by_student
from .logging_utils import get_logger, setup_logging
from .metrics import safe_acc, safe_auc, safe_f1, safe_rmse
from .ncdm import set_student_state_in_model, student_state_from_model
from .qmatrix import build_q_matrix

logger = get_logger()


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def residualize_state(state: np.ndarray, bias: np.ndarray) -> Tuple[Optional[np.ndarray], Tuple[float, float]]:
    """
    Remove linear component explainable by bias variable.
    Return (state_resid, corr_before_after_on_mean).
    """
    if state is None or state.ndim != 2:
        return None, (float("nan"), float("nan"))
    bias = np.asarray(bias).astype(float).reshape(-1, 1)
    resid = np.zeros_like(state, dtype=float)
    for k in range(state.shape[1]):
        rg = Ridge(alpha=1.0)
        rg.fit(bias, state[:, k])
        pred = rg.predict(bias)
        resid[:, k] = state[:, k] - pred
    before = _pearson_corr(state.mean(axis=1), bias.reshape(-1))
    after = _pearson_corr(resid.mean(axis=1), bias.reshape(-1))
    return resid, (before, after)


def _state_summaries(state: np.ndarray) -> Dict[str, np.ndarray]:
    mean = state.mean(axis=1)
    l2 = np.linalg.norm(state, axis=1)

    s = state - state.mean(axis=0, keepdims=True)
    try:
        u, _, _ = np.linalg.svd(s, full_matrices=False)
        pc1 = u[:, 0]
    except Exception:
        pc1 = np.zeros(state.shape[0], dtype=float)
    return {"mean": mean, "l2": l2, "pc1": pc1}


def _probe_state_predict_A(
    state: Optional[np.ndarray],
    train_u: np.ndarray,
    test_u: np.ndarray,
    A_train: np.ndarray,
    A_test: np.ndarray,
) -> Dict[str, float]:
    if state is None or state.ndim != 2:
        return {"auc": float("nan"), "acc": float("nan"), "logloss": float("nan")}
    n_user = state.shape[0]
    if train_u.size == 0 or test_u.size == 0:
        return {"auc": float("nan"), "acc": float("nan"), "logloss": float("nan")}
    if train_u.max() >= n_user or test_u.max() >= n_user:
        return {"auc": float("nan"), "acc": float("nan"), "logloss": float("nan")}
    if np.unique(A_train).size < 2 or np.unique(A_test).size < 2:
        return {"auc": float("nan"), "acc": float("nan"), "logloss": float("nan")}

    X_tr = state[train_u]
    X_te = state[test_u]
    clf = LogisticRegression(max_iter=200, solver="liblinear")
    clf.fit(X_tr, A_train.astype(int))
    prob = clf.predict_proba(X_te)[:, 1]
    return {
        "auc": float(roc_auc_score(A_test, prob)),
        "acc": float(accuracy_score(A_test, prob >= 0.5)),
        "logloss": float(log_loss(A_test, prob, labels=[0, 1])),
    }


def _probe_state_predict_z(
    state: Optional[np.ndarray],
    train_u: np.ndarray,
    test_u: np.ndarray,
    z_user: np.ndarray,
) -> Dict[str, float]:
    if state is None or state.ndim != 2:
        return {"r2": float("nan")}
    n_user = state.shape[0]
    if train_u.size == 0 or test_u.size == 0:
        return {"r2": float("nan")}
    if train_u.max() >= n_user or test_u.max() >= n_user:
        return {"r2": float("nan")}
    z = np.asarray(z_user).astype(float).reshape(-1)
    if z.size < n_user:
        return {"r2": float("nan")}

    X_tr = state[train_u]
    X_te = state[test_u]
    y_tr = z[train_u]
    y_te = z[test_u]
    if np.std(y_tr) <= 0 or np.std(y_te) <= 0:
        return {"r2": float("nan")}

    rg = LinearRegression()
    rg.fit(X_tr, y_tr)
    return {"r2": float(rg.score(X_te, y_te))}


def _state_bias_stats(state: Optional[np.ndarray], bias: np.ndarray) -> Dict[str, float]:
    if state is None or state.ndim != 2:
        return {
            "corr_dim_mean_abs": float("nan"),
            "corr_dim_max_abs": float("nan"),
            "cov_dim_mean_abs": float("nan"),
            "cov_dim_max_abs": float("nan"),
        }
    b = np.asarray(bias).astype(float).reshape(-1)
    if b.size != state.shape[0] or np.std(b) == 0:
        return {
            "corr_dim_mean_abs": float("nan"),
            "corr_dim_max_abs": float("nan"),
            "cov_dim_mean_abs": float("nan"),
            "cov_dim_max_abs": float("nan"),
        }
    b0 = b - np.mean(b)
    corr_abs = []
    cov_abs = []
    for j in range(state.shape[1]):
        s = state[:, j].astype(float)
        c = _pearson_corr(s, b)
        if np.isfinite(c):
            corr_abs.append(abs(float(c)))
        s0 = s - np.mean(s)
        cov = float(np.mean(s0 * b0))
        if np.isfinite(cov):
            cov_abs.append(abs(cov))
    if len(corr_abs) == 0 or len(cov_abs) == 0:
        return {
            "corr_dim_mean_abs": float("nan"),
            "corr_dim_max_abs": float("nan"),
            "cov_dim_mean_abs": float("nan"),
            "cov_dim_max_abs": float("nan"),
        }
    return {
        "corr_dim_mean_abs": float(np.mean(corr_abs)),
        "corr_dim_max_abs": float(np.max(corr_abs)),
        "cov_dim_mean_abs": float(np.mean(cov_abs)),
        "cov_dim_max_abs": float(np.max(cov_abs)),
    }


def _reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if mask.any():
            rows.append(
                {
                    "bin_left": lo,
                    "bin_right": hi,
                    "count": int(mask.sum()),
                    "prob_mean": float(np.mean(y_prob[mask])),
                    "true_mean": float(np.mean(y_true[mask])),
                }
            )
        else:
            rows.append(
                {
                    "bin_left": lo,
                    "bin_right": hi,
                    "count": 0,
                    "prob_mean": float("nan"),
                    "true_mean": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _plot_reliability(df: pd.DataFrame, path: str, title: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        logger.warning("[CAL] matplotlib not available; skip reliability plot.")
        return
    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.plot(df["prob_mean"], df["true_mean"], marker="o")
    plt.xlabel("predicted probability")
    plt.ylabel("observed accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _eval_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0 or y_prob.size == 0:
        return {"auc": float("nan"), "logloss": float("nan"), "acc": float("nan"), "f1": float("nan"), "rmse": float("nan"), "brier": float("nan"), "ece": float("nan")}
    if y_true.shape[0] != y_prob.shape[0]:
        return {"auc": float("nan"), "logloss": float("nan"), "acc": float("nan"), "f1": float("nan"), "rmse": float("nan"), "brier": float("nan"), "ece": float("nan")}
    if not np.isfinite(y_prob).all():
        return {"auc": float("nan"), "logloss": float("nan"), "acc": float("nan"), "f1": float("nan"), "rmse": float("nan"), "brier": float("nan"), "ece": float("nan")}
    ll = float("nan")
    if y_true.size > 10 and len(np.unique(y_true)) >= 2:
        ll = log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))
    brier = float("nan")
    if y_true.size > 0:
        brier = float(np.mean((y_prob - y_true) ** 2))
    rel = _reliability_curve(y_true, np.clip(y_prob, 0.0, 1.0), bins=10)
    ece = float("nan")
    valid = rel["count"] > 0
    if valid.any():
        n = float(rel.loc[valid, "count"].sum())
        if n > 0:
            ece = float(
                np.sum(
                    (rel.loc[valid, "count"] / n)
                    * np.abs(rel.loc[valid, "prob_mean"] - rel.loc[valid, "true_mean"])
                )
            )
    return {
        "auc": safe_auc(y_true, y_prob),
        "logloss": ll,
        "acc": safe_acc(y_true, y_prob),
        "f1": safe_f1(y_true, y_prob),
        "rmse": safe_rmse(y_true, y_prob),
        "brier": brier,
        "ece": ece,
    }


def _ks_distance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    xs = np.sort(x)
    ys = np.sort(y)
    grid = np.sort(np.concatenate([xs, ys]))
    cdf_x = np.searchsorted(xs, grid, side="right") / float(xs.size)
    cdf_y = np.searchsorted(ys, grid, side="right") / float(ys.size)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _wasserstein_1d_approx(x: np.ndarray, y: np.ndarray, n_quantiles: int = 1001) -> float:
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    qs = np.linspace(0.0, 1.0, int(n_quantiles))
    qx = np.quantile(x, qs)
    qy = np.quantile(y, qs)
    return float(np.mean(np.abs(qx - qy)))


def _build_trimmed_y0(
    A: np.ndarray,
    y: np.ndarray,
    m0_hat: np.ndarray,
    e_hat_raw: np.ndarray,
    eps: float,
    clip01: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A).astype(int)
    y = np.asarray(y).astype(float)
    m0 = np.asarray(m0_hat).astype(float)
    e_raw = np.asarray(e_hat_raw).astype(float)

    e_clip = np.clip(e_raw, eps, 1 - eps)
    keep = (e_raw >= eps) & (e_raw <= 1 - eps)

    y0_raw = m0 + ((1 - A) * keep.astype(float) * (y - m0)) / (1 - e_clip)
    y0 = np.clip(y0_raw, 0.0, 1.0) if clip01 else y0_raw
    return y0.astype(float), y0_raw.astype(float)


def _dr_stability_row(
    dataset: str,
    setting: str,
    eps: float,
    A_train: np.ndarray,
    e_hat_train: np.ndarray,
    e_hat_raw_train: np.ndarray,
    e_hat_test: np.ndarray,
    y0_tilde: np.ndarray,
    y0_raw: np.ndarray,
    m0_hat: np.ndarray,
    tail_tau: float = 0.5,
) -> Dict[str, float]:
    A = np.asarray(A_train).astype(int)
    mask0 = A == 0
    e_clip = np.clip(np.asarray(e_hat_train).astype(float), eps, 1 - eps)
    e_raw = np.asarray(e_hat_raw_train).astype(float)

    row = {
        "dataset": dataset,
        "setting": setting,
        "eps": float(eps),
        "n_train": int(A.size),
        "n_a0": int(mask0.sum()),
        "tail_tau": float(tail_tau),
        "w_p90": float("nan"),
        "w_p95": float("nan"),
        "w_p99": float("nan"),
        "w_max": float("nan"),
        "clip_rate_upper_a0": float("nan"),
        "clip_rate_lower_a0": float("nan"),
        "ess": float("nan"),
        "n_ess": float("nan"),
        "tail_rate_oob": float("nan"),
        "tail_rate_dev": float("nan"),
        "ks_shift": _ks_distance(e_raw, np.asarray(e_hat_test).astype(float)),
        "wasserstein_shift": _wasserstein_1d_approx(e_raw, np.asarray(e_hat_test).astype(float)),
    }
    if mask0.sum() == 0:
        return row

    w0 = 1.0 / (1.0 - e_clip[mask0])
    row["w_p90"] = float(np.percentile(w0, 90))
    row["w_p95"] = float(np.percentile(w0, 95))
    row["w_p99"] = float(np.percentile(w0, 99))
    row["w_max"] = float(np.max(w0))
    row["clip_rate_upper_a0"] = float(np.mean(e_raw[mask0] > (1 - eps)))
    row["clip_rate_lower_a0"] = float(np.mean(e_raw[mask0] < eps))

    w_sum = float(np.sum(w0))
    w_sq_sum = float(np.sum(w0**2))
    if w_sq_sum > 0:
        ess = (w_sum**2) / w_sq_sum
        row["ess"] = float(ess)
        row["n_ess"] = float(ess / max(int(mask0.sum()), 1))

    y0 = np.asarray(y0_tilde).astype(float)
    y0_r = np.asarray(y0_raw).astype(float)
    m0 = np.asarray(m0_hat).astype(float)
    row["tail_rate_oob"] = float(np.mean((y0_r < 0.0) | (y0_r > 1.0)))
    row["tail_rate_dev"] = float(np.mean(np.abs(y0 - m0) > tail_tau))
    return row


def _parse_cd_value_map(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if text is None:
        return out
    s = str(text).strip()
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        k = str(k).strip().lower()
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out


def _parse_ds_cd_value_map(text: str) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    if text is None:
        return out
    s = str(text).strip()
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        if "|" not in k:
            continue
        ds, model = k.split("|", 1)
        ds_key = str(ds).strip().lower()
        model_key = str(model).strip().lower()
        if not ds_key or not model_key:
            continue
        try:
            out[(ds_key, model_key)] = float(v)
        except Exception:
            continue
    return out


def _canon_dataset_key(dataset: str) -> str:
    return str(dataset).strip().lower()


def _parse_dataset_value_map(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if text is None:
        return out
    s = str(text).strip()
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        ds_key = _canon_dataset_key(k)
        try:
            out[ds_key] = float(v)
        except Exception:
            continue
    return out


def _resolve_dataset_user_keep_ratio(args: argparse.Namespace, dataset: str) -> float:
    ratio = float(getattr(args, "dataset_user_keep_ratio", 1.0))
    ds_map = _parse_dataset_value_map(getattr(args, "dataset_user_keep_map", ""))
    ds_key = _canon_dataset_key(dataset)
    if ds_key in ds_map:
        ratio = float(ds_map[ds_key])
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"dataset_user_keep_ratio for {dataset} must be in [0,1], got {ratio}")
    return float(ratio)


def _deterministic_user_keep_mask(user_series: pd.Series, keep_ratio: float, seed: int) -> np.ndarray:
    r = float(keep_ratio)
    if r >= 1.0:
        return np.ones(len(user_series), dtype=bool)
    if r <= 0.0:
        return np.zeros(len(user_series), dtype=bool)

    # Deterministic hash-based sampling by user id: preserve per-user sequences.
    u = user_series.astype(str)
    h = pd.util.hash_pandas_object(u, index=False).to_numpy(dtype=np.uint64)
    seed_mix = (int(seed) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    mixed = h ^ np.uint64(seed_mix)
    scores = ((mixed >> np.uint64(11)).astype(np.float64)) / float(1 << 53)
    return (scores < r)


def _resolve_or_dr_res_hparams(args: argparse.Namespace, dataset: str, cd_model: str) -> Tuple[float, float]:
    model_key = str(cd_model).strip().lower()
    ds_key = _canon_dataset_key(dataset)
    alpha = float(getattr(args, "or_dr_res_alpha", 1.0))
    eps = float(getattr(args, "or_dr_res_eps", 0.05))

    # dataset+cd_model tuned defaults from current sensitivity evidence.
    matched_ds_auto = False
    if bool(getattr(args, "or_dr_res_auto_by_dataset_cd", False)):
        rec_ds = {
            ("assistment-2009-2010-skill", "irt"): (0.5, 0.01),
            ("assistment-2009-2010-skill", "kancd"): (0.0, 0.01),
            ("assistment-2009-2010-skill", "ncdm"): (2.0, 0.10),
            ("assistment-2017", "irt"): (0.0, 0.01),
            ("assistment-2017", "kancd"): (0.5, 0.10),
            ("assistment-2017", "ncdm"): (2.0, 0.05),
        }
        if (ds_key, model_key) in rec_ds:
            alpha, eps = rec_ds[(ds_key, model_key)]
            matched_ds_auto = True

    if (not matched_ds_auto) and bool(getattr(args, "or_dr_res_auto_by_cd", False)):
        rec = {
            "irt": (0.5, 0.01),
            "kancd": (0.5, 0.01),
            "ncdm": (2.0, 0.05),
        }
        if model_key in rec:
            alpha, eps = rec[model_key]

    alpha_map = _parse_cd_value_map(getattr(args, "or_dr_res_alpha_map", ""))
    eps_map = _parse_cd_value_map(getattr(args, "or_dr_res_eps_map", ""))
    if model_key in alpha_map:
        alpha = float(alpha_map[model_key])
    if model_key in eps_map:
        eps = float(eps_map[model_key])

    alpha_map_ds = _parse_ds_cd_value_map(getattr(args, "or_dr_res_alpha_map_dataset_cd", ""))
    eps_map_ds = _parse_ds_cd_value_map(getattr(args, "or_dr_res_eps_map_dataset_cd", ""))
    if (ds_key, model_key) in alpha_map_ds:
        alpha = float(alpha_map_ds[(ds_key, model_key)])
    if (ds_key, model_key) in eps_map_ds:
        eps = float(eps_map_ds[(ds_key, model_key)])
    return float(alpha), float(eps)


def _build_dr_residual_weights(
    A_train: np.ndarray,
    e_hat_train: np.ndarray,
    eps: float,
    mode: str = "ipw",
    overlap_gamma: float = 0.0,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.asarray(A_train).astype(int)
    e = np.asarray(e_hat_train).astype(float)
    lo = max(float(eps), 1e-6)
    hi = 1.0 - lo
    if hi <= lo:
        lo = 1e-6
        hi = 1.0 - lo
    e_clip = np.clip(e, lo, hi)

    mask0 = (A == 0)
    w = np.zeros_like(e_clip, dtype=float)
    if not np.any(mask0):
        return w.astype(np.float32), mask0.astype(np.float32), e_clip.astype(float)

    den = np.maximum(1.0 - e_clip[mask0], 1e-6)
    mode_key = str(mode).strip().lower()
    if mode_key == "ipw":
        w0 = 1.0 / den
    elif mode_key == "stabilized":
        p0 = float(np.mean(mask0.astype(float)))
        w0 = p0 / den
    elif mode_key == "overlap":
        w0 = e_clip[mask0] / den
    else:
        raise ValueError(f"Unknown or_dr_res_weight_mode: {mode}")

    if float(overlap_gamma) > 0.0:
        overlap = np.clip(4.0 * e_clip[mask0] * (1.0 - e_clip[mask0]), 0.0, 1.0)
        w0 = w0 * np.power(overlap, float(overlap_gamma))

    if bool(normalize):
        m = float(np.mean(w0)) if w0.size > 0 else 0.0
        if np.isfinite(m) and m > 0:
            w0 = w0 / m

    w[mask0] = w0
    return w.astype(np.float32), mask0.astype(np.float32), e_clip.astype(float)


def _load_or_build_cache(ds: str, args: argparse.Namespace):
    folder = ensure_downloaded(ds, args.data_root)
    csv_path = pick_main_csv(folder)
    key = cache_key(ds, csv_path, args)
    paths = cache_paths(args.data_root, key)

    loaded = None
    if args.use_cache and (not args.rebuild_cache):
        loaded = load_cache(paths)
    if loaded is not None:
        logger.info("[CACHE] hit: %s", paths["base"])
        return loaded, csv_path

    logger.info("[CACHE] miss -> preprocess: %s", paths["base"])
    df = pd.read_csv(
        csv_path,
        encoding="latin1",
        on_bad_lines="skip",
        sep=None,
        engine="python",
    )
    cols = infer_cols(df)
    keep_ratio = _resolve_dataset_user_keep_ratio(args, ds)
    sample_seed = int(getattr(args, "dataset_user_sample_seed", getattr(args, "seed", 42)))
    if keep_ratio < 1.0:
        rows_before = int(len(df))
        users_before = int(df[cols.user].astype(str).nunique())
        keep_mask = _deterministic_user_keep_mask(df[cols.user], keep_ratio=keep_ratio, seed=sample_seed)
        df = df.loc[keep_mask].reset_index(drop=True)
        rows_after = int(len(df))
        users_after = int(df[cols.user].astype(str).nunique()) if rows_after > 0 else 0
        logger.info(
            "[SAMPLE] dataset=%s keep_user_ratio=%.4f seed=%s | rows %s -> %s | users %s -> %s",
            ds,
            keep_ratio,
            sample_seed,
            rows_before,
            rows_after,
            users_before,
            users_after,
        )
        if rows_after <= 0:
            raise ValueError(f"Sampling removed all rows for dataset={ds}. Increase keep ratio.")
    else:
        logger.info("[SAMPLE] dataset=%s keep_user_ratio=1.0 (disabled)", ds)
    A = build_treatment_A(df, cols)

    X, user2idx, item2idx, skill2idx, d_sorted = make_features(df, cols, A)
    y = d_sorted["_y"].values.astype(int)
    A_s = d_sorted["_A"].values.astype(int)

    train_mask, test_mask = temporal_split_by_student(d_sorted, test_ratio=0.2)
    X_train = X.loc[train_mask].reset_index(drop=True)
    y_train_int = y[train_mask]
    A_train = A_s[train_mask]

    y0_tilde_train, e_hat_train, m0_hat_train, _ = crossfit_dr_pseudolabel(
        X_train,
        A_train,
        y_train_int,
        kfold=5,
        eps=0.05,
        clip01=True,
        seed=args.seed,
        propensity_model=args.propensity_model,
    )
    y_train = y_train_int.astype(np.float32)

    Q = build_q_matrix(d_sorted, item2idx, skill2idx)
    d_train = d_sorted.loc[train_mask].reset_index(drop=True)
    d_test = d_sorted.loc[test_mask].reset_index(drop=True)

    train_u = d_train["_u"].map(user2idx).values.astype(np.int64)
    train_i = d_train["_i"].map(item2idx).values.astype(np.int64)
    test_u = d_test["_u"].map(user2idx).values.astype(np.int64)
    test_i = d_test["_i"].map(item2idx).values.astype(np.int64)

    arrays = {
        "y": y.astype(np.int8),
        "A": A_s.astype(np.int8),
        "train_mask": train_mask.astype(np.uint8),
        "test_mask": test_mask.astype(np.uint8),
        "y_train": y_train.astype(np.float32),
        "y0_tilde_train": y0_tilde_train.astype(np.float32),
        "e_hat_train": e_hat_train.astype(np.float32),
        "m0_hat_train": m0_hat_train.astype(np.float32),
        "train_u": train_u.astype(np.int32),
        "train_i": train_i.astype(np.int32),
        "test_u": test_u.astype(np.int32),
        "test_i": test_i.astype(np.int32),
    }
    maps = {"user2idx": user2idx, "item2idx": item2idx, "skill2idx": skill2idx}
    meta = {
        "dataset": ds,
        "csv": csv_path,
        "rows": int(len(d_sorted)),
        "users": int(len(user2idx)),
        "items": int(len(item2idx)),
        "skills": int(len(skill2idx)),
        "dataset_user_keep_ratio": float(keep_ratio),
        "dataset_user_sample_seed": int(sample_seed),
        "seed": int(args.seed),
        "propensity_model": str(args.propensity_model),
        "kfold": 5,
        "eps": 0.05,
        "split": 0.2,
    }
    save_cache(paths, d_sorted, X, arrays, maps, Q, meta)
    return (d_sorted, X, arrays, maps, Q, meta), csv_path


def run_experiments(args: argparse.Namespace) -> None:
    if args.run_dir is None:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = os.path.join("./outputs", f"exp_{run_stamp}")
    os.makedirs(args.run_dir, exist_ok=True)
    if args.log_path is None:
        args.log_path = os.path.join(args.run_dir, "run.log")
    setup_logging(args.log_path)

    params = {
        "datasets": list(args.datasets),
        "cd_models": list(args.cd_models),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "device": args.device,
        "rep_debias_finetune_epochs": int(args.rep_debias_finetune_epochs),
        "eps_list": list(args.eps_list),
        "outcome_models": list(args.outcome_models),
        "propensity_bins": int(args.propensity_bins),
        "propensity_model": str(args.propensity_model),
        "dr_tail_tau": float(args.dr_tail_tau),
        "dataset_user_keep_ratio": float(getattr(args, "dataset_user_keep_ratio", 1.0)),
        "dataset_user_keep_map": str(getattr(args, "dataset_user_keep_map", "")),
        "dataset_user_sample_seed": int(getattr(args, "dataset_user_sample_seed", 42)),
        "enable_or_dr_res": bool(getattr(args, "enable_or_dr_res", False)),
        "or_dr_res_alpha": float(getattr(args, "or_dr_res_alpha", 1.0)),
        "or_dr_res_eps": float(getattr(args, "or_dr_res_eps", 0.05)),
        "or_dr_res_auto_by_dataset_cd": bool(getattr(args, "or_dr_res_auto_by_dataset_cd", False)),
        "or_dr_res_auto_by_cd": bool(getattr(args, "or_dr_res_auto_by_cd", False)),
        "or_dr_res_alpha_map": str(getattr(args, "or_dr_res_alpha_map", "")),
        "or_dr_res_eps_map": str(getattr(args, "or_dr_res_eps_map", "")),
        "or_dr_res_alpha_map_dataset_cd": str(getattr(args, "or_dr_res_alpha_map_dataset_cd", "")),
        "or_dr_res_eps_map_dataset_cd": str(getattr(args, "or_dr_res_eps_map_dataset_cd", "")),
        "or_dr_res_weight_mode": str(getattr(args, "or_dr_res_weight_mode", "ipw")),
        "or_dr_res_overlap_gamma": float(getattr(args, "or_dr_res_overlap_gamma", 0.0)),
        "or_dr_res_weight_normalize": bool(getattr(args, "or_dr_res_weight_normalize", True)),
        "or_dr_res_dr_warmup_epochs": int(getattr(args, "or_dr_res_dr_warmup_epochs", 0)),
        "enable_or_dr_res_pre_rep_mlp": bool(getattr(args, "enable_or_dr_res_pre_rep_mlp", False)),
        "or_dr_res_pre_rep_hidden": int(getattr(args, "or_dr_res_pre_rep_hidden", 32)),
        "or_dr_res_pre_rep_mag_eta": float(getattr(args, "or_dr_res_pre_rep_mag_eta", 0.0)),
        "or_dr_res_pre_rep_mode": str(getattr(args, "or_dr_res_pre_rep_mode", "subtract")),
        "or_dr_res_pre_rep_warmup_epochs": int(getattr(args, "or_dr_res_pre_rep_warmup_epochs", 0)),
        "skip_outcome_sensitivity": bool(getattr(args, "skip_outcome_sensitivity", False)),
        "skip_epsilon_sensitivity": bool(getattr(args, "skip_epsilon_sensitivity", False)),
        "only_stability_diag": bool(getattr(args, "only_stability_diag", False)),
        "only_leakage": bool(getattr(args, "only_leakage", False)),
    }
    with open(os.path.join(args.run_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    main_rows: List[Dict] = []
    ablation_rows: List[Dict] = []
    eps_rows: List[Dict] = []
    outcome_rows: List[Dict] = []
    leakage_rows: List[Dict] = []
    bin_rows: List[Dict] = []
    dr_stability_rows: List[Dict] = []
    ps_only_rows: List[Dict] = []

    first_dataset = args.datasets[0] if args.datasets else None
    only_stability_diag = bool(getattr(args, "only_stability_diag", False))

    for ds in args.datasets:
        logger.info("%s", "\n" + "=" * 80)
        logger.info("[DATASET] %s", ds)

        ds_key = ds.replace("/", "_")
        ds_dir = os.path.join(args.run_dir, ds_key)
        os.makedirs(ds_dir, exist_ok=True)

        loaded, csv_path = _load_or_build_cache(ds, args)
        d_sorted, X, arrays, maps, Q, _meta = loaded

        user2idx = maps["user2idx"]
        item2idx = maps["item2idx"]
        skill2idx = maps["skill2idx"]
        n_user, n_item, n_skill = len(user2idx), len(item2idx), len(skill2idx)

        y = arrays["y"].astype(int)
        A_s = arrays["A"].astype(int)
        train_mask = arrays["train_mask"].astype(bool)
        test_mask = arrays["test_mask"].astype(bool)

        y_train = arrays["y_train"].astype(np.float32)
        y_train_int = y_train.astype(int)
        y0_tilde_train = arrays["y0_tilde_train"].astype(np.float32)
        e_hat_train = arrays["e_hat_train"].astype(np.float32)
        m0_hat_train = arrays.get("m0_hat_train")
        if m0_hat_train is not None:
            m0_hat_train = m0_hat_train.astype(np.float32)
        need_m0 = m0_hat_train is None

        train_u = arrays["train_u"].astype(np.int64)
        train_i = arrays["train_i"].astype(np.int64)
        test_u = arrays["test_u"].astype(np.int64)
        test_i = arrays["test_i"].astype(np.int64)

        d_train = d_sorted.loc[train_mask].reset_index(drop=True)
        d_test = d_sorted.loc[test_mask].reset_index(drop=True)
        train_u = d_train["_u"].map(user2idx).values.astype(np.int64)
        train_i = d_train["_i"].map(item2idx).values.astype(np.int64)
        test_u = d_test["_u"].map(user2idx).values.astype(np.int64)
        test_i = d_test["_i"].map(item2idx).values.astype(np.int64)

        train_k = Q[train_i].astype(np.float32)
        test_k = Q[test_i].astype(np.float32)

        y_test = y[test_mask]
        A_test = A_s[test_mask]
        A_train = A_s[train_mask]
        X_train = X.loc[train_mask].reset_index(drop=True)
        X_test = X.loc[test_mask].reset_index(drop=True)

        if need_m0:
            _y0_tmp, _e_tmp, m0_hat_train, _ = crossfit_dr_pseudolabel(
                X_train,
                A_train,
                y_train_int,
                kfold=5,
                eps=0.05,
                clip01=True,
                seed=args.seed,
                outcome_model=args.outcome_models[0],
                propensity_model=args.propensity_model,
            )

        step1_metrics: Dict[str, float] = {}
        if not getattr(args, "only_leakage", False):
            # Step1 diagnostics (propensity/weights/balance)
            step1_metrics = run_step1_diagnostics(
                X_train=X_train,
                A_train=A_train,
                e_hat_train=e_hat_train,
                y_train=y_train_int,
                m0_hat_train=m0_hat_train,
                X_test=X_test,
                A_test=A_test,
                run_dir=ds_dir,
                eps=0.05,
                propensity_model=args.propensity_model,
            )

        # Student-level propensity proxy (for residualization)
        e_mean = pd.Series(e_hat_train).groupby(train_u).mean()
        e_vec = np.array([e_mean.get(i, e_mean.mean()) for i in range(n_user)], dtype=float)

        # IPW weights
        e_clip = np.clip(e_hat_train, 0.05, 1 - 0.05)
        w_ipw = np.where(A_train == 1, 1.0 / e_clip, 1.0 / (1.0 - e_clip))

        nohint = A_test == 0
        y_nohint = y_test[nohint]

        # Propensity on test for bin robustness
        X_train_use = X_train.drop(columns=["A_hint"], errors="ignore")
        X_test_use = X_test.drop(columns=["A_hint"], errors="ignore")
        X_train_probe = X_train_use.values.astype(float)
        X_test_probe = X_test_use.values.astype(float)
        prop_model = fit_propensity(X_train_probe, A_train, model=args.propensity_model)
        e_hat_test_raw = prop_model.predict_proba(X_test_probe)[:, 1]
        bin_edges = np.linspace(0.0, 1.0, int(args.propensity_bins) + 1)
        bin_masks: List[Tuple[int, float, float, np.ndarray]] = []
        for b in range(len(bin_edges) - 1):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            if b == len(bin_edges) - 2:
                mask = (e_hat_test_raw >= lo) & (e_hat_test_raw <= hi)
            else:
                mask = (e_hat_test_raw >= lo) & (e_hat_test_raw < hi)
            bin_masks.append((b, float(lo), float(hi), mask))

        # Main methods
        for cd_model in args.cd_models:
            if only_stability_diag:
                break
            alpha_res, eps_res = _resolve_or_dr_res_hparams(args, ds, cd_model)
            dr_weight_mode = str(getattr(args, "or_dr_res_weight_mode", "ipw"))
            dr_overlap_gamma = float(getattr(args, "or_dr_res_overlap_gamma", 0.0))
            dr_weight_normalize = bool(getattr(args, "or_dr_res_weight_normalize", True))
            dr_warmup_epochs = int(getattr(args, "or_dr_res_dr_warmup_epochs", 0))

            if bool(getattr(args, "enable_or_dr_res", False)):
                logger.info(
                    "[OR+DRres] %s: alpha=%.4f eps=%.4f mode=%s overlap_gamma=%.3f norm=%s dr_warmup=%d",
                    str(cd_model),
                    float(alpha_res),
                    float(eps_res),
                    dr_weight_mode,
                    float(dr_overlap_gamma),
                    str(dr_weight_normalize),
                    int(dr_warmup_epochs),
                )

            # Raw
            raw_model, raw_proba = train_and_eval_cd(
                cd_model,
                train_u,
                train_i,
                train_k,
                y_train.astype(np.float32),
                test_u,
                test_i,
                test_k,
                y_test.astype(np.float32),
                n_user,
                n_item,
                n_skill,
                epochs=args.epochs,
                device=args.device,
                progress_desc=f"{cd_model}-raw",
            )

            # A0-only (skip in leakage-only mode)
            a0_proba = np.full_like(y_test, np.nan, dtype=float)
            if not getattr(args, "only_leakage", False):
                mask0 = A_train == 0
                if mask0.sum() > 0:
                    _a0_model, a0_proba = train_and_eval_cd(
                        cd_model,
                        train_u[mask0],
                        train_i[mask0],
                        train_k[mask0],
                        y_train[mask0].astype(np.float32),
                        test_u,
                        test_i,
                        test_k,
                        y_test.astype(np.float32),
                        n_user,
                        n_item,
                        n_skill,
                        epochs=args.epochs,
                        device=args.device,
                        progress_desc=f"{cd_model}-a0",
                    )

            # Outcome-only (m0)
            out_state = None
            if m0_hat_train is not None:
                out_model, out_proba = train_and_eval_cd(
                    cd_model,
                    train_u,
                    train_i,
                    train_k,
                    m0_hat_train.astype(np.float32),
                    test_u,
                    test_i,
                    test_k,
                    y_test.astype(np.float32),
                    n_user,
                    n_item,
                    n_skill,
                    epochs=args.epochs,
                    device=args.device,
                    progress_desc=f"{cd_model}-outcome",
                )
                out_state = student_state_from_model(out_model, n_user, n_skill)
            else:
                out_proba = np.full_like(y_test, np.nan, dtype=float)

            # Outcome + DR residual(A=0): BCE(m0) + alpha * I(A=0)*w*(Y-yhat)^2
            or_dr_res_state = None
            or_dr_res_proba = np.full_like(y_test, np.nan, dtype=float)
            or_dr_res_rep_state = None
            or_dr_res_rep_proba = np.full_like(y_test, np.nan, dtype=float)
            if bool(getattr(args, "enable_or_dr_res", False)) and (m0_hat_train is not None):
                w_a0, mask_a0, _e_res = _build_dr_residual_weights(
                    A_train=A_train,
                    e_hat_train=e_hat_train,
                    eps=float(eps_res),
                    mode=dr_weight_mode,
                    overlap_gamma=dr_overlap_gamma,
                    normalize=dr_weight_normalize,
                )
                or_dr_res_model, or_dr_res_proba = train_and_eval_cd(
                    cd_model,
                    train_u,
                    train_i,
                    train_k,
                    m0_hat_train.astype(np.float32),
                    test_u,
                    test_i,
                    test_k,
                    y_test.astype(np.float32),
                    n_user,
                    n_item,
                    n_skill,
                    epochs=args.epochs,
                    device=args.device,
                    progress_desc=f"{cd_model}-out_drres",
                    dr_residual_coef=alpha_res,
                    dr_residual_target=y_train.astype(np.float32),
                    dr_residual_weight=w_a0.astype(np.float32),
                    dr_residual_mask=mask_a0.astype(np.float32),
                    dr_residual_warmup_epochs=dr_warmup_epochs,
                )
                or_dr_res_state = student_state_from_model(or_dr_res_model, n_user, n_skill)

                # Outcome + DR_res + pre-forward rep-debias MLP: s_u^deb = s_u - g(z_u)
                if bool(getattr(args, "enable_or_dr_res_pre_rep_mlp", False)):
                    or_dr_res_rep_model, or_dr_res_rep_proba = train_and_eval_cd(
                        cd_model,
                        train_u,
                        train_i,
                        train_k,
                        m0_hat_train.astype(np.float32),
                        test_u,
                        test_i,
                        test_k,
                        y_test.astype(np.float32),
                        n_user,
                        n_item,
                        n_skill,
                        epochs=args.epochs,
                        device=args.device,
                        progress_desc=f"{cd_model}-out_drres_repmlp",
                        dr_residual_coef=alpha_res,
                        dr_residual_target=y_train.astype(np.float32),
                        dr_residual_weight=w_a0.astype(np.float32),
                        dr_residual_mask=mask_a0.astype(np.float32),
                        dr_residual_warmup_epochs=dr_warmup_epochs,
                        pre_rep_debias=True,
                        pre_rep_propensity=e_vec.astype(np.float32),
                        pre_rep_hidden=int(getattr(args, "or_dr_res_pre_rep_hidden", 32)),
                        pre_rep_mag_coef=float(getattr(args, "or_dr_res_pre_rep_mag_eta", 0.0)),
                        pre_rep_mode=str(getattr(args, "or_dr_res_pre_rep_mode", "subtract")),
                        pre_rep_warmup_epochs=int(getattr(args, "or_dr_res_pre_rep_warmup_epochs", 0)),
                    )
                    or_dr_res_rep_state = student_state_from_model(or_dr_res_rep_model, n_user, n_skill)

            # IPW-only (skip in leakage-only mode)
            ipw_proba = np.full_like(y_test, np.nan, dtype=float)
            if not getattr(args, "only_leakage", False):
                _ipw_model, ipw_proba = train_and_eval_cd(
                    cd_model,
                    train_u,
                    train_i,
                    train_k,
                    y_train.astype(np.float32),
                    test_u,
                    test_i,
                    test_k,
                    y_test.astype(np.float32),
                    n_user,
                    n_item,
                    n_skill,
                    epochs=args.epochs,
                    device=args.device,
                    progress_desc=f"{cd_model}-ipw",
                    sample_weight=w_ipw.astype(np.float32),
                )

            # Data (DR pseudo-label)
            data_model, data_proba = train_and_eval_cd(
                cd_model,
                train_u,
                train_i,
                train_k,
                y0_tilde_train.astype(np.float32),
                test_u,
                test_i,
                test_k,
                y_test.astype(np.float32),
                n_user,
                n_item,
                n_skill,
                epochs=args.epochs,
                device=args.device,
                progress_desc=f"{cd_model}-data",
            )

            # Rep-only: residualize raw state
            rep_proba = np.full_like(y_test, np.nan, dtype=float)
            rep_ok = False
            rep_state = None
            raw_state = student_state_from_model(raw_model, n_user, n_skill)
            if raw_state is not None:
                rep_state, _ = residualize_state(raw_state, e_vec)
                if rep_state is not None and set_student_state_in_model(raw_model, rep_state, n_user):
                    rep_proba = predict_cd(
                        cd_model,
                        raw_model,
                        test_u,
                        test_i,
                        test_k,
                        device=args.device,
                        progress_desc=f"{cd_model}-rep",
                    )
                    rep_ok = True

            # Full: residualize data model state
            full_proba = np.full_like(y_test, np.nan, dtype=float)
            full_ok = False
            full_state = None
            data_state = student_state_from_model(data_model, n_user, n_skill)
            if data_state is not None:
                full_state, _ = residualize_state(data_state, e_vec)
                if full_state is not None and set_student_state_in_model(data_model, full_state, n_user):
                    full_proba = predict_cd(
                        cd_model,
                        data_model,
                        test_u,
                        test_i,
                        test_k,
                        device=args.device,
                        progress_desc=f"{cd_model}-full",
                    )
                    full_ok = True

            # Full + FT (skip in leakage-only mode)
            full_ft_proba = np.full_like(y_test, np.nan, dtype=float)
            if full_ok and args.rep_debias_finetune_epochs > 0 and (not getattr(args, "only_leakage", False)):
                finetune_cd(
                    cd_model,
                    data_model,
                    train_u,
                    train_i,
                    train_k,
                    y0_tilde_train.astype(np.float32),
                    n_user=n_user,
                    epochs=int(args.rep_debias_finetune_epochs),
                    device=args.device,
                    freeze_student=True,
                )
                full_ft_proba = predict_cd(
                    cd_model,
                    data_model,
                    test_u,
                    test_i,
                    test_k,
                    device=args.device,
                    progress_desc=f"{cd_model}-fullft",
                )
            elif full_ok:
                full_ft_proba = full_proba.copy()

            # leakage / correlation evidence
            states_for_leakage = {
                "Raw": raw_state,
                "Outcome-only": out_state,
                "Outcome+DRres(A0)": or_dr_res_state,
                "Outcome+DRres(A0)+RepMLP": or_dr_res_rep_state,
                "Data": data_state,
                "Rep": rep_state if rep_ok else None,
                "Full": full_state if full_ok else None,
            }
            for tag, state in states_for_leakage.items():
                if state is None or state.ndim != 2:
                    leakage_rows.append(
                        {
                            "dataset": ds,
                            "cd_model": cd_model,
                            "method": tag,
                            "corr_mean": float("nan"),
                            "corr_l2": float("nan"),
                            "corr_pc1": float("nan"),
                            "corr_dim_mean_abs": float("nan"),
                            "corr_dim_max_abs": float("nan"),
                            "cov_dim_mean_abs": float("nan"),
                            "cov_dim_max_abs": float("nan"),
                            "probe_auc": float("nan"),
                            "probe_acc": float("nan"),
                            "probe_logloss": float("nan"),
                            "probe_z_r2": float("nan"),
                        }
                    )
                    continue
                prop = e_vec
                if prop.shape[0] != state.shape[0]:
                    prop = prop[: state.shape[0]]
                sums = _state_summaries(state)
                corr_mean = _pearson_corr(sums["mean"], prop)
                corr_l2 = _pearson_corr(sums["l2"], prop)
                corr_pc1 = _pearson_corr(sums["pc1"], prop)
                probe = _probe_state_predict_A(state, train_u, test_u, A_train, A_test)
                probe_z = _probe_state_predict_z(state, train_u, test_u, prop)
                bias_stats = _state_bias_stats(state, prop)
                leakage_rows.append(
                    {
                        "dataset": ds,
                        "cd_model": cd_model,
                        "method": tag,
                        "corr_mean": corr_mean,
                        "corr_l2": corr_l2,
                        "corr_pc1": corr_pc1,
                        "corr_dim_mean_abs": bias_stats["corr_dim_mean_abs"],
                        "corr_dim_max_abs": bias_stats["corr_dim_max_abs"],
                        "cov_dim_mean_abs": bias_stats["cov_dim_mean_abs"],
                        "cov_dim_max_abs": bias_stats["cov_dim_max_abs"],
                        "probe_auc": probe["auc"],
                        "probe_acc": probe["acc"],
                        "probe_logloss": probe["logloss"],
                        "probe_z_r2": probe_z["r2"],
                    }
                )

            if getattr(args, "only_leakage", False):
                continue

            # collect metrics
            method_probas = {
                "Raw": raw_proba,
                "A0-only": a0_proba,
                "Outcome-only": out_proba,
                "IPW-only": ipw_proba,
                "Data": data_proba,
                "Full": full_proba,
            }
            if bool(getattr(args, "enable_or_dr_res", False)):
                method_probas["Outcome+DRres(A0)"] = or_dr_res_proba
                if bool(getattr(args, "enable_or_dr_res_pre_rep_mlp", False)):
                    method_probas["Outcome+DRres(A0)+RepMLP"] = or_dr_res_rep_proba
            for method, proba in method_probas.items():
                metrics = _eval_metrics(y_nohint, proba[nohint])
                is_or_dr = str(method).startswith("Outcome+DRres")
                main_rows.append(
                    {
                        "dataset": ds,
                        "cd_model": cd_model,
                        "method": method,
                        "or_dr_res_alpha": float(alpha_res) if is_or_dr else float("nan"),
                        "or_dr_res_eps": float(eps_res) if is_or_dr else float("nan"),
                        "or_dr_res_weight_mode": dr_weight_mode if is_or_dr else "",
                        "or_dr_res_overlap_gamma": float(dr_overlap_gamma) if is_or_dr else float("nan"),
                        "or_dr_res_dr_warmup_epochs": int(dr_warmup_epochs) if is_or_dr else 0,
                        "or_dr_res_pre_rep_mode": str(getattr(args, "or_dr_res_pre_rep_mode", "subtract")) if method == "Outcome+DRres(A0)+RepMLP" else "",
                        "or_dr_res_pre_rep_warmup_epochs": int(getattr(args, "or_dr_res_pre_rep_warmup_epochs", 0)) if method == "Outcome+DRres(A0)+RepMLP" else 0,
                        **metrics,
                        "n_test": int(len(y_test)),
                        "n_nohint": int(nohint.sum()),
                    }
                )

            # propensity-bin robustness
            for method, proba in method_probas.items():
                for b, lo, hi, mask in bin_masks:
                    m = mask & nohint
                    metrics = _eval_metrics(y_test[m], proba[m])
                    bin_rows.append(
                        {
                            "dataset": ds,
                            "cd_model": cd_model,
                            "method": method,
                            "bin": int(b),
                            "bin_left": float(lo),
                            "bin_right": float(hi),
                            "count": int(m.sum()),
                            **metrics,
                        }
                    )

            # ablation rows
            ablation_rows.append(
                {
                    "dataset": ds,
                    "cd_model": cd_model,
                    "version": "Raw",
                    **_eval_metrics(y_nohint, raw_proba[nohint]),
                }
            )
            ablation_rows.append(
                {
                    "dataset": ds,
                    "cd_model": cd_model,
                    "version": "Data",
                    **_eval_metrics(y_nohint, data_proba[nohint]),
                }
            )
            if bool(getattr(args, "enable_or_dr_res", False)):
                ablation_rows.append(
                    {
                        "dataset": ds,
                        "cd_model": cd_model,
                        "version": "Outcome+DRres(A0)",
                        **_eval_metrics(y_nohint, or_dr_res_proba[nohint]),
                    }
                )
                if bool(getattr(args, "enable_or_dr_res_pre_rep_mlp", False)):
                    ablation_rows.append(
                        {
                            "dataset": ds,
                            "cd_model": cd_model,
                            "version": "Outcome+DRres(A0)+RepMLP",
                            **_eval_metrics(y_nohint, or_dr_res_rep_proba[nohint]),
                        }
                    )
            ablation_rows.append(
                {
                    "dataset": ds,
                    "cd_model": cd_model,
                    "version": "Rep",
                    **_eval_metrics(y_nohint, rep_proba[nohint]),
                }
            )
            ablation_rows.append(
                {
                    "dataset": ds,
                    "cd_model": cd_model,
                    "version": "Full",
                    **_eval_metrics(y_nohint, full_proba[nohint]),
                }
            )
            ablation_rows.append(
                {
                    "dataset": ds,
                    "cd_model": cd_model,
                    "version": "H2D-CD (Full) + FT",
                    **_eval_metrics(y_nohint, full_ft_proba[nohint]),
                }
            )

            # reliability plots (at least one dataset)
            if ds == first_dataset and cd_model.lower() == "ncdm":
                cal_dir = os.path.join(args.run_dir, "calibration")
                os.makedirs(cal_dir, exist_ok=True)
                for tag, proba in [
                    ("raw", raw_proba),
                    ("data", data_proba),
                    ("full", full_proba),
                ]:
                    rel = _reliability_curve(y_nohint, proba[nohint], bins=10)
                    rel.to_csv(os.path.join(cal_dir, f"reliability_{tag}_{ds_key}.csv"), index=False)
                    _plot_reliability(
                        rel,
                        os.path.join(cal_dir, f"reliability_{tag}_{ds_key}.png"),
                        f"Reliability {tag} ({ds})",
                    )

        # Epsilon sensitivity (at least 1 dataset × all CD models)
        # Outcome-model sensitivity + epsilon sensitivity (first dataset only)
        if ((ds == first_dataset) or only_stability_diag) and (not getattr(args, "only_leakage", False)):
            sens_dir = os.path.join(args.run_dir, "outcome_sensitivity")
            os.makedirs(sens_dir, exist_ok=True)
            if (not only_stability_diag) and (not bool(getattr(args, "skip_outcome_sensitivity", False))):
                for om in args.outcome_models:
                    y0_om, _e_om, m0_om, _ = crossfit_dr_pseudolabel(
                        X_train,
                        A_train,
                        y_train_int,
                        kfold=5,
                        eps=0.05,
                        clip01=True,
                        seed=args.seed,
                        outcome_model=str(om),
                        propensity_model=args.propensity_model,
                    )
                    for cd_model in args.cd_models:
                        _, proba_data = train_and_eval_cd(
                            cd_model,
                            train_u,
                            train_i,
                            train_k,
                            y0_om.astype(np.float32),
                            test_u,
                            test_i,
                            test_k,
                            y_test.astype(np.float32),
                            n_user,
                            n_item,
                            n_skill,
                            epochs=args.epochs,
                            device=args.device,
                            progress_desc=f"{cd_model}-data-{om}",
                        )
                        outcome_rows.append(
                            {
                                "dataset": ds,
                                "cd_model": cd_model,
                                "outcome_model": str(om),
                                "method": "Data",
                                **_eval_metrics(y_nohint, proba_data[nohint]),
                            }
                        )

                        if m0_om is not None:
                            _, proba_out = train_and_eval_cd(
                                cd_model,
                                train_u,
                                train_i,
                                train_k,
                                m0_om.astype(np.float32),
                                test_u,
                                test_i,
                                test_k,
                                y_test.astype(np.float32),
                                n_user,
                                n_item,
                                n_skill,
                                epochs=args.epochs,
                                device=args.device,
                                progress_desc=f"{cd_model}-outcome-{om}",
                            )
                            outcome_rows.append(
                                {
                                    "dataset": ds,
                                    "cd_model": cd_model,
                                    "outcome_model": str(om),
                                    "method": "Outcome-only",
                                    **_eval_metrics(y_nohint, proba_out[nohint]),
                                }
                            )

            eps_dir = os.path.join(args.run_dir, "epsilon")
            os.makedirs(eps_dir, exist_ok=True)
            if bool(getattr(args, "skip_epsilon_sensitivity", False)):
                continue
            for eps in args.eps_list:
                eps = float(eps)
                y0_eps, e_eps, m0_eps, _, e_raw_eps = crossfit_dr_pseudolabel(
                    X_train,
                    A_train,
                    y_train_int,
                    kfold=5,
                    eps=eps,
                    clip01=True,
                    seed=args.seed,
                    propensity_model=args.propensity_model,
                    return_raw_e=True,
                )

                # Cross-fit diagnostics
                e_test_clip_eps = np.clip(e_hat_test_raw, eps, 1 - eps)
                y0_raw_eps = m0_eps + ((1 - A_train) * (y_train_int - m0_eps)) / (1 - e_eps)
                cf_row = _dr_stability_row(
                    dataset=ds,
                    setting="crossfit",
                    eps=eps,
                    A_train=A_train,
                    e_hat_train=e_eps,
                    e_hat_raw_train=e_raw_eps,
                    e_hat_test=e_test_clip_eps,
                    y0_tilde=y0_eps,
                    y0_raw=y0_raw_eps,
                    m0_hat=m0_eps,
                    tail_tau=float(args.dr_tail_tau),
                )
                cf_row["propensity_model"] = str(args.propensity_model)
                dr_stability_rows.append(cf_row)
                ps_only_rows.append(
                    {
                        "dataset": ds,
                        "propensity_model": str(args.propensity_model),
                        "eps": eps,
                        "propensity_train_auc": step1_metrics.get("propensity_train_auc", float("nan")),
                        "propensity_test_auc": step1_metrics.get("propensity_test_auc", float("nan")),
                        "propensity_train_logloss": step1_metrics.get("propensity_train_logloss", float("nan")),
                        "propensity_test_logloss": step1_metrics.get("propensity_test_logloss", float("nan")),
                        "propensity_train_brier": step1_metrics.get("propensity_train_brier", float("nan")),
                        "propensity_test_brier": step1_metrics.get("propensity_test_brier", float("nan")),
                        "propensity_train_pr_auc": step1_metrics.get("propensity_train_pr_auc", float("nan")),
                        "propensity_test_pr_auc": step1_metrics.get("propensity_test_pr_auc", float("nan")),
                        "propensity_train_ece": step1_metrics.get("propensity_train_ece", float("nan")),
                        "propensity_test_ece": step1_metrics.get("propensity_test_ece", float("nan")),
                        "overlap_lt_eps": float(np.mean(e_raw_eps < eps)),
                        "overlap_gt_1m_eps": float(np.mean(e_raw_eps > (1 - eps))),
                        "w_p95": cf_row["w_p95"],
                        "w_p99": cf_row["w_p99"],
                        "w_max": cf_row["w_max"],
                        "clip_rate_upper_a0": cf_row["clip_rate_upper_a0"],
                        "clip_rate_lower_a0": cf_row["clip_rate_lower_a0"],
                        "ess": cf_row["ess"],
                        "n_ess": cf_row["n_ess"],
                        "ks_shift": cf_row["ks_shift"],
                        "wasserstein_shift": cf_row["wasserstein_shift"],
                        "smd_mean_unweighted": step1_metrics.get("smd_mean_unweighted", float("nan")),
                        "smd_mean_weighted": step1_metrics.get("smd_mean_weighted", float("nan")),
                        "smd_max_unweighted": step1_metrics.get("smd_max_unweighted", float("nan")),
                        "smd_max_weighted": step1_metrics.get("smd_max_weighted", float("nan")),
                    }
                )

                # Cross-fit + trimming diagnostics
                y0_trim, y0_trim_raw = _build_trimmed_y0(
                    A=A_train,
                    y=y_train_int,
                    m0_hat=m0_eps,
                    e_hat_raw=e_raw_eps,
                    eps=eps,
                    clip01=True,
                )
                row_trim = _dr_stability_row(
                    dataset=ds,
                    setting="crossfit_trim",
                    eps=eps,
                    A_train=A_train,
                    e_hat_train=np.clip(e_raw_eps, eps, 1 - eps),
                    e_hat_raw_train=e_raw_eps,
                    e_hat_test=e_test_clip_eps,
                    y0_tilde=y0_trim,
                    y0_raw=y0_trim_raw,
                    m0_hat=m0_eps,
                    tail_tau=float(args.dr_tail_tau),
                )
                row_trim["propensity_model"] = str(args.propensity_model)
                dr_stability_rows.append(row_trim)

                # No-cross-fit diagnostics
                y0_ncf, e_ncf, m0_ncf, _, e_raw_ncf = dr_pseudolabel_no_crossfit(
                    X_train,
                    A_train,
                    y_train_int,
                    eps=eps,
                    clip01=True,
                    outcome_model=args.outcome_models[0],
                    propensity_model=args.propensity_model,
                    return_raw_e=True,
                )
                y0_raw_ncf = m0_ncf + ((1 - A_train) * (y_train_int - m0_ncf)) / (1 - e_ncf)
                row_ncf = _dr_stability_row(
                    dataset=ds,
                    setting="no_crossfit",
                    eps=eps,
                    A_train=A_train,
                    e_hat_train=e_ncf,
                    e_hat_raw_train=e_raw_ncf,
                    e_hat_test=e_test_clip_eps,
                    y0_tilde=y0_ncf,
                    y0_raw=y0_raw_ncf,
                    m0_hat=m0_ncf,
                    tail_tau=float(args.dr_tail_tau),
                )
                row_ncf["propensity_model"] = str(args.propensity_model)
                dr_stability_rows.append(row_ncf)

                if not only_stability_diag:
                    for cd_model in args.cd_models:
                        _, proba = train_and_eval_cd(
                            cd_model,
                            train_u,
                            train_i,
                            train_k,
                            y0_eps.astype(np.float32),
                            test_u,
                            test_i,
                            test_k,
                            y_test.astype(np.float32),
                            n_user,
                            n_item,
                            n_skill,
                            epochs=args.epochs,
                            device=args.device,
                            progress_desc=f"{cd_model}-eps{eps}",
                        )
                        metrics = _eval_metrics(y_nohint, proba[nohint])
                        eps_rows.append(
                            {
                                "dataset": ds,
                                "cd_model": cd_model,
                                "eps": eps,
                                **metrics,
                            }
                        )

    # save tables
    if not getattr(args, "only_leakage", False):
        main_df = pd.DataFrame(main_rows)
        main_df.to_csv(os.path.join(args.run_dir, "main_results.csv"), index=False)
        main_df.to_json(os.path.join(args.run_dir, "main_results.json"), orient="records", indent=2)

        ablation_df = pd.DataFrame(ablation_rows)
        ablation_df.to_csv(os.path.join(args.run_dir, "ablation_results.csv"), index=False)
        ablation_df.to_json(os.path.join(args.run_dir, "ablation_results.json"), orient="records", indent=2)

        if eps_rows:
            eps_df = pd.DataFrame(eps_rows)
            eps_df.to_csv(os.path.join(args.run_dir, "epsilon_results.csv"), index=False)
            eps_df.to_json(os.path.join(args.run_dir, "epsilon_results.json"), orient="records", indent=2)

        if outcome_rows:
            out_df = pd.DataFrame(outcome_rows)
            out_df.to_csv(os.path.join(args.run_dir, "outcome_sensitivity_results.csv"), index=False)
            out_df.to_json(os.path.join(args.run_dir, "outcome_sensitivity_results.json"), orient="records", indent=2)

        if dr_stability_rows:
            dr_df = pd.DataFrame(dr_stability_rows)
            dr_df.to_csv(os.path.join(args.run_dir, "dr_stability_results.csv"), index=False)
            dr_df.to_json(os.path.join(args.run_dir, "dr_stability_results.json"), orient="records", indent=2)
            tab_cols = ["dataset", "propensity_model", "eps", "w_p95", "w_p99", "w_max", "n_ess", "tail_rate_dev", "ks_shift"]
            tab_df = dr_df[dr_df["setting"] == "crossfit"][tab_cols].copy()
            tab_df.to_csv(os.path.join(args.run_dir, "dr_stability_table.csv"), index=False)

        if ps_only_rows:
            ps_df = pd.DataFrame(ps_only_rows)
            ps_df.to_csv(os.path.join(args.run_dir, "ps_only_eval.csv"), index=False)
            ps_df.to_json(os.path.join(args.run_dir, "ps_only_eval.json"), orient="records", indent=2)

    if leakage_rows:
        leak_df = pd.DataFrame(leakage_rows)
        leak_df.to_csv(os.path.join(args.run_dir, "leakage_results.csv"), index=False)
        leak_df.to_json(os.path.join(args.run_dir, "leakage_results.json"), orient="records", indent=2)

    if (not getattr(args, "only_leakage", False)) and bin_rows:
        bin_df = pd.DataFrame(bin_rows)
        bin_df.to_csv(os.path.join(args.run_dir, "propensity_bin_results.csv"), index=False)
        bin_df.to_json(os.path.join(args.run_dir, "propensity_bin_results.json"), orient="records", indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./edudata_cache")
    ap.add_argument("--datasets", type=str, nargs="*", default=DEFAULT_DATASETS)
    ap.add_argument("--cd_models", type=str, nargs="*", default=["ncdm", "kancd", "irt"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_cache", action="store_true", default=True)
    ap.add_argument("--rebuild_cache", action="store_true", default=False)
    ap.add_argument("--run_dir", type=str, default=None)
    ap.add_argument("--log_path", type=str, default=None)
    ap.add_argument("--rep_debias_finetune_epochs", type=int, default=1)
    ap.add_argument("--eps_list", type=float, nargs="*", default=[0.01, 0.02, 0.05, 0.1])
    ap.add_argument("--outcome_models", type=str, nargs="*", default=["gbdt", "lr", "mlp"])
    ap.add_argument("--propensity_bins", type=int, default=5)
    ap.add_argument("--propensity_model", type=str, default="lr", choices=["lr", "gbdt", "mlp"])
    ap.add_argument("--dr_tail_tau", type=float, default=0.5)
    ap.add_argument("--dataset_user_keep_ratio", type=float, default=1.0)
    ap.add_argument(
        "--dataset_user_keep_map",
        type=str,
        default="",
        help='Per-dataset keep ratio, e.g. "EdNet-KT3:0.5,assistment-2017:1.0"',
    )
    ap.add_argument("--dataset_user_sample_seed", type=int, default=42)
    ap.add_argument("--enable_or_dr_res", action="store_true", default=False)
    ap.add_argument("--or_dr_res_alpha", type=float, default=1.0)
    ap.add_argument("--or_dr_res_eps", type=float, default=0.05)
    ap.add_argument("--or_dr_res_auto_by_dataset_cd", action="store_true", default=False)
    ap.add_argument("--or_dr_res_auto_by_cd", action="store_true", default=False)
    ap.add_argument("--or_dr_res_alpha_map", type=str, default="")
    ap.add_argument("--or_dr_res_eps_map", type=str, default="")
    ap.add_argument("--or_dr_res_alpha_map_dataset_cd", type=str, default="")
    ap.add_argument("--or_dr_res_eps_map_dataset_cd", type=str, default="")
    ap.add_argument("--or_dr_res_weight_mode", type=str, default="ipw", choices=["ipw", "stabilized", "overlap"])
    ap.add_argument("--or_dr_res_overlap_gamma", type=float, default=0.0)
    ap.add_argument("--or_dr_res_weight_normalize", action="store_true", default=True)
    ap.add_argument("--no_or_dr_res_weight_normalize", dest="or_dr_res_weight_normalize", action="store_false")
    ap.add_argument("--or_dr_res_dr_warmup_epochs", type=int, default=0)
    ap.add_argument("--enable_or_dr_res_pre_rep_mlp", action="store_true", default=False)
    ap.add_argument("--or_dr_res_pre_rep_hidden", type=int, default=32)
    ap.add_argument("--or_dr_res_pre_rep_mag_eta", type=float, default=0.0)
    ap.add_argument("--or_dr_res_pre_rep_mode", type=str, default="subtract", choices=["subtract", "gated"])
    ap.add_argument("--or_dr_res_pre_rep_warmup_epochs", type=int, default=0)
    ap.add_argument("--skip_outcome_sensitivity", action="store_true", default=False)
    ap.add_argument("--skip_epsilon_sensitivity", action="store_true", default=False)
    ap.add_argument("--only_stability_diag", action="store_true", default=False)
    ap.add_argument("--only_leakage", action="store_true", default=False)
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    if not args.datasets:
        raise ValueError("No datasets provided.")
    if len(args.datasets) < 2:
        logger.warning("[WARN] datasets < 2; some tables may be incomplete for the spec.")
    run_experiments(args)


if __name__ == "__main__":
    main()
