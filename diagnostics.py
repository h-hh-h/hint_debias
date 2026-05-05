import os
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from .dr import fit_propensity
from .logging_utils import get_logger

logger = get_logger()


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def _normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-float(z) / math.sqrt(2.0))


def _auc_hanley_se(auc: float, n_pos: int, n_neg: int) -> float:
    if n_pos <= 0 or n_neg <= 0 or not np.isfinite(auc):
        return float("nan")
    q1 = auc / (2.0 - auc) if auc < 2.0 else float("nan")
    q2 = (2.0 * auc * auc) / (1.0 + auc) if auc > -1.0 else float("nan")
    if (not np.isfinite(q1)) or (not np.isfinite(q2)):
        return float("nan")
    var = (
        (auc * (1.0 - auc))
        + (n_pos - 1.0) * (q1 - auc * auc)
        + (n_neg - 1.0) * (q2 - auc * auc)
    ) / (n_pos * n_neg)
    if (not np.isfinite(var)) or var < 0.0:
        return float("nan")
    return float(np.sqrt(max(var, 0.0)))


def _auc_vs_random_stats(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """
    One-sided test for H0: AUC=0.5 vs H1: AUC>0.5.
    z/p-value uses Mann-Whitney U normal approximation with tie correction.
    95% CI uses Hanley-McNeil SE approximation.
    """
    out: Dict[str, Any] = {
        "auc_delta_vs_random": float("nan"),
        "auc_zscore_vs_random": float("nan"),
        "auc_pvalue_vs_random_one_sided": float("nan"),
        "auc_pvalue_vs_random_two_sided": float("nan"),
        "auc_ci95_low": float("nan"),
        "auc_ci95_high": float("nan"),
        "auc_sig_gt_random_0p05": False,
        "auc_test_n_pos": 0,
        "auc_test_n_neg": 0,
        "auc_test_n_total": 0,
    }

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)
    if y_true.size == 0 or y_score.size == 0 or y_true.size != y_score.size:
        return out
    if len(np.unique(y_true)) < 2:
        return out

    auc = _safe_auc(y_true, y_score)
    if not np.isfinite(auc):
        return out

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    n_total = int(y_true.size)
    if n_pos <= 0 or n_neg <= 0:
        return out

    ranks = pd.Series(y_score).rank(method="average").to_numpy(dtype=float)
    u_pos = float(np.sum(ranks[y_true == 1]) - n_pos * (n_pos + 1) / 2.0)
    mean_u = 0.5 * n_pos * n_neg

    tie_counts = np.unique(y_score, return_counts=True)[1].astype(float)
    tie_term = float(np.sum(tie_counts**3 - tie_counts))
    tie_corr = tie_term / (n_total * (n_total - 1.0)) if n_total > 1 else 0.0
    var_u = (n_pos * n_neg / 12.0) * ((n_total + 1.0) - tie_corr)

    if np.isfinite(var_u) and var_u > 0:
        z = float((u_pos - mean_u) / np.sqrt(var_u))
        cdf_z = _normal_cdf(z)
        p_one = float(np.clip(1.0 - cdf_z, 0.0, 1.0))
        p_two = float(np.clip(2.0 * min(cdf_z, 1.0 - cdf_z), 0.0, 1.0))
    else:
        z = float("nan")
        p_one = float("nan")
        p_two = float("nan")

    se_auc = _auc_hanley_se(auc, n_pos=n_pos, n_neg=n_neg)
    if np.isfinite(se_auc):
        ci_low = float(np.clip(auc - 1.96 * se_auc, 0.0, 1.0))
        ci_high = float(np.clip(auc + 1.96 * se_auc, 0.0, 1.0))
    else:
        ci_low, ci_high = float("nan"), float("nan")

    out.update(
        {
            "auc_delta_vs_random": float(auc - 0.5),
            "auc_zscore_vs_random": z,
            "auc_pvalue_vs_random_one_sided": p_one,
            "auc_pvalue_vs_random_two_sided": p_two,
            "auc_ci95_low": ci_low,
            "auc_ci95_high": ci_high,
            "auc_sig_gt_random_0p05": bool(np.isfinite(p_one) and (p_one < alpha) and (auc > 0.5)),
            "auc_test_n_pos": int(n_pos),
            "auc_test_n_neg": int(n_neg),
            "auc_test_n_total": int(n_total),
        }
    )
    return out


def _safe_logloss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    y_score = np.clip(np.asarray(y_score).astype(float), 1e-6, 1 - 1e-6)
    return log_loss(y_true, y_score)


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return average_precision_score(y_true, y_score)


def _brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_true - y_score) ** 2))


def _save_histogram(values: np.ndarray, path: str, bins: int = 20) -> None:
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    df = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts})
    df.to_csv(path, index=False)
    return df


def _calibration_bins(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_idx = np.digitize(y_pred, edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = bin_idx == b
        cnt = int(mask.sum())
        pred_mean = float(np.mean(y_pred[mask])) if cnt > 0 else float("nan")
        true_mean = float(np.mean(y_true[mask])) if cnt > 0 else float("nan")
        rows.append(
            {
                "bin_left": float(edges[b]),
                "bin_right": float(edges[b + 1]),
                "count": cnt,
                "pred_mean": pred_mean,
                "true_mean": true_mean,
            }
        )
    return pd.DataFrame(rows)


def _ece_from_calibration_bins(df: pd.DataFrame) -> float:
    valid = df["count"] > 0
    if not valid.any():
        return float("nan")
    n = float(df.loc[valid, "count"].sum())
    if n <= 0:
        return float("nan")
    return float(
        np.sum(
            (df.loc[valid, "count"] / n)
            * np.abs(df.loc[valid, "pred_mean"] - df.loc[valid, "true_mean"])
        )
    )


def _plot_reliability(df: pd.DataFrame, path: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        logger.warning("[STEP1] matplotlib not available; skip reliability plot.")
        return
    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.plot(df["pred_mean"], df["true_mean"], marker="o")
    plt.xlabel("predicted probability")
    plt.ylabel("observed rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _weighted_mean_var(x: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    w = np.asarray(w).astype(float)
    x = np.asarray(x).astype(float)
    w_sum = np.sum(w)
    if w_sum <= 0:
        return float("nan"), float("nan")
    mean = np.sum(w * x) / w_sum
    var = np.sum(w * (x - mean) ** 2) / w_sum
    return float(mean), float(var)


def _smd_table(X: pd.DataFrame, A: np.ndarray, weights: np.ndarray = None) -> pd.DataFrame:
    feats = X.columns.tolist()
    out = []
    A = np.asarray(A).astype(int)
    if weights is None:
        weights = np.ones_like(A, dtype=float)
    for col in feats:
        x = X[col].values.astype(float)
        mask1 = A == 1
        mask0 = A == 0
        w1 = weights[mask1]
        w0 = weights[mask0]
        m1, v1 = _weighted_mean_var(x[mask1], w1)
        m0, v0 = _weighted_mean_var(x[mask0], w0)
        sd_pooled = np.sqrt((v1 + v0) / 2.0) if np.isfinite(v1) and np.isfinite(v0) else float("nan")
        if sd_pooled == 0 or not np.isfinite(sd_pooled):
            smd = 0.0
        else:
            smd = abs(m1 - m0) / sd_pooled
        out.append({"feature": col, "mean_t": m1, "mean_c": m0, "smd": smd})
    return pd.DataFrame(out)


def _maybe_love_plot(df: pd.DataFrame, path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        logger.warning("[STEP1] matplotlib not available; skip love plot.")
        return

    df = df.sort_values("smd_unweighted", ascending=True)
    y = np.arange(len(df))
    plt.figure(figsize=(6, max(3, 0.25 * len(df))))
    plt.scatter(df["smd_unweighted"], y, label="unweighted", s=16)
    plt.scatter(df["smd_weighted"], y, label="ipw", s=16)
    plt.yticks(y, df["feature"].tolist(), fontsize=8)
    plt.axvline(0.1, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("SMD")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_step1_diagnostics(
    X_train: pd.DataFrame,
    A_train: np.ndarray,
    e_hat_train: np.ndarray,
    y_train: np.ndarray,
    m0_hat_train: Optional[np.ndarray],
    X_test: pd.DataFrame,
    A_test: np.ndarray,
    run_dir: str,
    eps: float = 0.05,
    propensity_model: str = "lr",
) -> Dict[str, float]:
    step_dir = os.path.join(run_dir, "step1")
    os.makedirs(step_dir, exist_ok=True)

    X_train_use = X_train.drop(columns=["A_hint"], errors="ignore")
    X_test_use = X_test.drop(columns=["A_hint"], errors="ignore")

    # Propensity eval (train uses OOF e_hat; test uses model fitted on train)
    prop_model = fit_propensity(X_train_use.values.astype(float), A_train, model=propensity_model)
    e_hat_test = prop_model.predict_proba(X_test_use.values.astype(float))[:, 1]

    metrics = {
        "propensity_model": str(propensity_model),
        "propensity_random_auc_baseline": 0.5,
        "propensity_train_auc": _safe_auc(A_train, e_hat_train),
        "propensity_train_logloss": _safe_logloss(A_train, e_hat_train),
        "propensity_train_brier": _brier_score(A_train, e_hat_train),
        "propensity_train_pr_auc": _safe_pr_auc(A_train, e_hat_train),
        "propensity_test_auc": _safe_auc(A_test, e_hat_test),
        "propensity_test_logloss": _safe_logloss(A_test, e_hat_test),
        "propensity_test_brier": _brier_score(A_test, e_hat_test),
        "propensity_test_pr_auc": _safe_pr_auc(A_test, e_hat_test),
    }
    train_auc_stats = _auc_vs_random_stats(A_train, e_hat_train, alpha=0.05)
    test_auc_stats = _auc_vs_random_stats(A_test, e_hat_test, alpha=0.05)
    metrics.update({f"propensity_train_{k}": v for k, v in train_auc_stats.items()})
    metrics.update({f"propensity_test_{k}": v for k, v in test_auc_stats.items()})

    cal_train = _calibration_bins(A_train, e_hat_train, bins=10)
    cal_test = _calibration_bins(A_test, e_hat_test, bins=10)
    metrics["propensity_train_ece"] = _ece_from_calibration_bins(cal_train)
    metrics["propensity_test_ece"] = _ece_from_calibration_bins(cal_test)

    pd.DataFrame([metrics]).to_json(os.path.join(step_dir, "propensity_metrics.json"), orient="records", indent=2)
    cal_train.to_csv(os.path.join(step_dir, "propensity_calibration_train.csv"), index=False)
    cal_test.to_csv(os.path.join(step_dir, "propensity_calibration_test.csv"), index=False)
    hist_train = _save_histogram(e_hat_train, os.path.join(step_dir, "propensity_hist_train.csv"))
    hist_test = _save_histogram(e_hat_test, os.path.join(step_dir, "propensity_hist_test.csv"))

    # Overlap / trimming stats
    e_hat_train = np.asarray(e_hat_train).astype(float)
    overlap_stats = {
        "eps": float(eps),
        "train_min": float(np.min(e_hat_train)),
        "train_max": float(np.max(e_hat_train)),
        "train_mean": float(np.mean(e_hat_train)),
        "train_pct_lt_eps": float(np.mean(e_hat_train < eps)),
        "train_pct_gt_1m_eps": float(np.mean(e_hat_train > (1 - eps))),
    }
    pd.DataFrame([overlap_stats]).to_json(os.path.join(step_dir, "overlap_stats.json"), orient="records", indent=2)

    # IPW weights stats
    e_clip = np.clip(e_hat_train, eps, 1 - eps)
    w_ipw = np.where(A_train == 1, 1.0 / e_clip, 1.0 / (1.0 - e_clip))
    w_stats = {
        "w_mean": float(np.mean(w_ipw)),
        "w_std": float(np.std(w_ipw)),
        "w_max": float(np.max(w_ipw)),
        "w_p95": float(np.percentile(w_ipw, 95)),
        "w_p99": float(np.percentile(w_ipw, 99)),
    }
    pd.DataFrame([w_stats]).to_json(os.path.join(step_dir, "weight_stats.json"), orient="records", indent=2)
    # weight histogram (clip extreme tails for stable bins)
    w_clip = np.clip(w_ipw, 0.0, np.percentile(w_ipw, 99))
    w_edges = np.linspace(float(np.min(w_clip)), float(np.max(w_clip) + 1e-8), 20 + 1)
    w_counts, _ = np.histogram(w_clip, bins=w_edges)
    w_hist = pd.DataFrame({"bin_left": w_edges[:-1], "bin_right": w_edges[1:], "count": w_counts})
    w_hist.to_csv(os.path.join(step_dir, "weight_hist.csv"), index=False)

    # Covariate balance (SMD)
    smd_unw = _smd_table(X_train_use, A_train, weights=None).rename(columns={"smd": "smd_unweighted"})
    smd_w = _smd_table(X_train_use, A_train, weights=w_ipw).rename(columns={"smd": "smd_weighted"})
    smd = smd_unw[["feature", "smd_unweighted"]].merge(
        smd_w[["feature", "smd_weighted"]], on="feature", how="left"
    )
    smd.to_csv(os.path.join(step_dir, "balance_smd.csv"), index=False)
    metrics["smd_mean_unweighted"] = float(smd["smd_unweighted"].mean())
    metrics["smd_mean_weighted"] = float(smd["smd_weighted"].mean())
    metrics["smd_max_unweighted"] = float(smd["smd_unweighted"].max())
    metrics["smd_max_weighted"] = float(smd["smd_weighted"].max())

    _maybe_love_plot(smd, os.path.join(step_dir, "love_plot.png"))

    # optional plots
    try:
        import matplotlib.pyplot as plt

        # propensity overlay plot
        plt.figure(figsize=(5, 4))
        plt.plot(hist_train["bin_left"], hist_train["count"], label="train", marker="o")
        plt.plot(hist_test["bin_left"], hist_test["count"], label="test", marker="o")
        plt.xlabel("propensity bin left")
        plt.ylabel("count")
        plt.title("Propensity Histogram (Train vs Test)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(step_dir, "propensity_hist_train_test.png"), dpi=150)
        plt.close()

        _plot_reliability(cal_train, os.path.join(step_dir, "propensity_reliability_train.png"), "Propensity Reliability (Train)")
        _plot_reliability(cal_test, os.path.join(step_dir, "propensity_reliability_test.png"), "Propensity Reliability (Test)")

        # weight histogram
        plt.figure(figsize=(5, 4))
        plt.bar(w_hist["bin_left"], w_hist["count"], width=(w_hist["bin_right"] - w_hist["bin_left"]), align="edge")
        plt.xlabel("IPW weight (clipped at p99)")
        plt.ylabel("count")
        plt.title("IPW Weight Histogram")
        plt.tight_layout()
        plt.savefig(os.path.join(step_dir, "weight_hist.png"), dpi=150)
        plt.close()
    except Exception:
        logger.warning("[STEP1] matplotlib not available; skip hist plots.")

    # Outcome model diagnostics on A=0 subset (m0_hat)
    if m0_hat_train is not None:
        mask0 = np.asarray(A_train).astype(int) == 0
        y0 = np.asarray(y_train).astype(int)[mask0]
        m0 = np.asarray(m0_hat_train).astype(float)[mask0]
        out_metrics = {
            "m0_nohint_auc": _safe_auc(y0, m0),
            "m0_nohint_logloss": _safe_logloss(y0, m0),
            "n_nohint": int(mask0.sum()),
        }
        pd.DataFrame([out_metrics]).to_json(
            os.path.join(step_dir, "outcome_m0_metrics.json"), orient="records", indent=2
        )
        calib = _calibration_bins(y0, m0, bins=10)
        calib.to_csv(os.path.join(step_dir, "outcome_m0_calibration.csv"), index=False)
        logger.info(
            "[STEP1] m0(A=0) AUC/LogLoss: %.4f / %.4f",
            out_metrics["m0_nohint_auc"],
            out_metrics["m0_nohint_logloss"],
        )
    else:
        logger.info("[STEP1] m0_hat_train not available; skip outcome diagnostics.")

    logger.info("[STEP1] propensity AUC train/test: %.4f / %.4f", metrics["propensity_train_auc"], metrics["propensity_test_auc"])
    logger.info(
        "[STEP1] propensity AUC-vs-random(train): delta=%.4f p(one)=%.3g sig@0.05=%s",
        metrics["propensity_train_auc_delta_vs_random"],
        metrics["propensity_train_auc_pvalue_vs_random_one_sided"],
        metrics["propensity_train_auc_sig_gt_random_0p05"],
    )
    logger.info(
        "[STEP1] propensity AUC-vs-random(test):  delta=%.4f p(one)=%.3g sig@0.05=%s",
        metrics["propensity_test_auc_delta_vs_random"],
        metrics["propensity_test_auc_pvalue_vs_random_one_sided"],
        metrics["propensity_test_auc_sig_gt_random_0p05"],
    )
    logger.info("[STEP1] propensity LogLoss train/test: %.4f / %.4f", metrics["propensity_train_logloss"], metrics["propensity_test_logloss"])
    logger.info("[STEP1] propensity Brier train/test: %.4f / %.4f", metrics["propensity_train_brier"], metrics["propensity_test_brier"])
    logger.info("[STEP1] propensity ECE train/test: %.4f / %.4f", metrics["propensity_train_ece"], metrics["propensity_test_ece"])
    logger.info("[STEP1] overlap: lt_eps=%.4f gt_1m_eps=%.4f", overlap_stats["train_pct_lt_eps"], overlap_stats["train_pct_gt_1m_eps"])
    logger.info("[STEP1] weight max/p99: %.4f / %.4f", w_stats["w_max"], w_stats["w_p99"])

    metrics.update(overlap_stats)
    metrics.update(w_stats)
    return metrics
