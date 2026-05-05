import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .logging_utils import get_logger

logger = get_logger()


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for step2 plots.") from exc
    return plt


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _state_summaries(state: np.ndarray) -> Dict[str, np.ndarray]:
    mean = state.mean(axis=1)
    l2 = np.linalg.norm(state, axis=1)

    # first principal component score
    s = state - state.mean(axis=0, keepdims=True)
    try:
        u, _, _ = np.linalg.svd(s, full_matrices=False)
        pc1 = u[:, 0]
    except Exception:
        pc1 = np.zeros(state.shape[0], dtype=float)

    return {"mean": mean, "l2": l2, "pc1": pc1}


def _bin_curve(x: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (x >= lo) & (x <= hi)
        else:
            mask = (x >= lo) & (x < hi)
        if mask.any():
            rows.append(
                {
                    "bin_left": lo,
                    "bin_right": hi,
                    "count": int(mask.sum()),
                    "x_mean": float(np.mean(x[mask])),
                    "y_mean": float(np.mean(y[mask])),
                }
            )
        else:
            rows.append(
                {
                    "bin_left": lo,
                    "bin_right": hi,
                    "count": 0,
                    "x_mean": float("nan"),
                    "y_mean": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _plot_scatter_with_bins(
    x: np.ndarray,
    y: np.ndarray,
    bins_df: pd.DataFrame,
    path: str,
    title: str,
    xlabel: str,
    ylabel: str,
):
    plt = _require_matplotlib()
    plt.figure(figsize=(5, 4))
    plt.scatter(x, y, s=6, alpha=0.15)
    plt.plot(bins_df["x_mean"], bins_df["y_mean"], color="red", linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_state_analysis(
    state: np.ndarray,
    propensity: np.ndarray,
    out_dir: str,
    tag: str,
    ds_key: str,
):
    os.makedirs(out_dir, exist_ok=True)
    summaries = _state_summaries(state)

    corr = {k: _pearson_corr(v, propensity) for k, v in summaries.items()}
    corr_path = os.path.join(out_dir, f"state_corr_{tag}_{ds_key}.json")
    with open(corr_path, "w", encoding="utf-8") as f:
        json.dump(corr, f, ensure_ascii=False, indent=2)

    rows = {"student_idx": np.arange(state.shape[0]), "propensity": propensity}
    for k, v in summaries.items():
        rows[f"state_{k}"] = v
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(out_dir, f"state_summary_{tag}_{ds_key}.csv"), index=False)

    for k, v in summaries.items():
        bins_df = _bin_curve(propensity, v, bins=10)
        bins_df.to_csv(os.path.join(out_dir, f"state_bins_{tag}_{k}_{ds_key}.csv"), index=False)
        _plot_scatter_with_bins(
            propensity,
            v,
            bins_df,
            os.path.join(out_dir, f"state_scatter_{tag}_{k}_{ds_key}.png"),
            title=f"state-{k} vs propensity ({tag})",
            xlabel="propensity",
            ylabel=f"state-{k}",
        )


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


def _ece_from_reliability(df: pd.DataFrame) -> float:
    valid = df["count"] > 0
    if not valid.any():
        return float("nan")
    n = df.loc[valid, "count"].sum()
    return float(
        np.sum(df.loc[valid, "count"] / n * np.abs(df.loc[valid, "prob_mean"] - df.loc[valid, "true_mean"]))
    )


def _brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    return float(np.mean((y_prob - y_true) ** 2))


def _plot_reliability(df: pd.DataFrame, path: str, title: str):
    plt = _require_matplotlib()
    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.plot(df["prob_mean"], df["true_mean"], marker="o")
    plt.xlabel("predicted probability")
    plt.ylabel("observed accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _probe_state_predict_A(
    state: np.ndarray,
    train_u: np.ndarray,
    test_u: np.ndarray,
    A_train: np.ndarray,
    A_test: np.ndarray,
) -> Dict[str, float]:
    """
    Fit a simple logistic probe: state -> A.
    Returns test metrics (auc/acc/logloss).
    """
    if state is None or state.ndim != 2:
        return {"auc": float("nan"), "acc": float("nan"), "logloss": float("nan")}

    n_user = state.shape[0]
    if train_u.size == 0 or test_u.size == 0:
        return {"auc": float("nan"), "acc": float("nan"), "logloss": float("nan")}
    if train_u.max() >= n_user or test_u.max() >= n_user:
        return {"auc": float("nan"), "acc": float("nan"), "logloss": float("nan")}

    X_tr = state[train_u]
    X_te = state[test_u]

    if np.unique(A_train).size < 2 or np.unique(A_test).size < 2:
        return {"auc": float("nan"), "acc": float("nan"), "logloss": float("nan")}

    clf = LogisticRegression(max_iter=200, solver="liblinear")
    clf.fit(X_tr, A_train.astype(int))
    prob = clf.predict_proba(X_te)[:, 1]

    auc = float(roc_auc_score(A_test, prob))
    acc = float(accuracy_score(A_test, prob >= 0.5))
    ll = float(log_loss(A_test, prob, labels=[0, 1]))
    return {"auc": auc, "acc": acc, "logloss": ll}


def run_step2_per_dataset(
    ds: str,
    run_dir: str,
    base_state: Optional[np.ndarray],
    deb_state: Optional[np.ndarray],
    e_hat_train: np.ndarray,
    A_train: np.ndarray,
    train_u: np.ndarray,
    test_u: np.ndarray,
    base_proba: np.ndarray,
    deb_proba: np.ndarray,
    y_test: np.ndarray,
    A_test: np.ndarray,
):
    ds_key = ds.replace("/", "_")
    step_dir = os.path.join(run_dir, "step2")
    os.makedirs(step_dir, exist_ok=True)

    # student-level propensity proxy
    e_series = pd.Series(e_hat_train).groupby(train_u).mean()
    n_user = int(train_u.max()) + 1
    if base_state is not None:
        n_user = max(n_user, int(base_state.shape[0]))
    if deb_state is not None:
        n_user = max(n_user, int(deb_state.shape[0]))
    propensity = np.array([e_series.get(i, e_series.mean()) for i in range(n_user)], dtype=float)
    np.save(os.path.join(step_dir, f"propensity_{ds_key}.npy"), propensity)

    # save states
    if base_state is not None:
        np.save(os.path.join(step_dir, f"student_state_baseline_{ds_key}.npy"), base_state)
        _save_state_analysis(base_state, propensity, step_dir, "baseline", ds_key)
    else:
        logger.warning("[STEP2] baseline state missing; skip state analysis.")

    if deb_state is not None:
        np.save(os.path.join(step_dir, f"student_state_debiased_{ds_key}.npy"), deb_state)
        _save_state_analysis(deb_state, propensity, step_dir, "debiased", ds_key)
    else:
        logger.warning("[STEP2] debiased state missing; skip state analysis.")

    # probe: state -> A (leakage of hint tendency)
    for tag, state in [("baseline", base_state), ("debiased", deb_state)]:
        metrics = _probe_state_predict_A(state, train_u, test_u, A_train, A_test)
        with open(os.path.join(step_dir, f"probe_stateA_{tag}_{ds_key}.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    # calibration on A=0 test subset
    nohint = A_test == 0
    y_nh = y_test[nohint]
    base_nh = base_proba[nohint]
    deb_nh = deb_proba[nohint]

    for tag, prob in [("baseline", base_nh), ("debiased", deb_nh)]:
        rel = _reliability_curve(y_nh, prob, bins=10)
        rel.to_csv(os.path.join(step_dir, f"reliability_{tag}_{ds_key}.csv"), index=False)
        _plot_reliability(rel, os.path.join(step_dir, f"reliability_{tag}_{ds_key}.png"), f"Reliability ({tag})")
        metrics = {
            "ece": _ece_from_reliability(rel),
            "brier": _brier_score(y_nh, prob),
        }
        with open(os.path.join(step_dir, f"calibration_{tag}_{ds_key}.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = pd.Series(a).rank(method="average").values
    b = pd.Series(b).rank(method="average").values
    return float(np.corrcoef(a, b)[0, 1])


def _procrustes_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - a.mean(axis=0, keepdims=True)
    b0 = b - b.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(b0.T @ a0, full_matrices=False)
    r = u @ vt
    b_aligned = b0 @ r
    num = np.sum(a0 * b_aligned, axis=1)
    denom = np.linalg.norm(a0, axis=1) * np.linalg.norm(b_aligned, axis=1)
    return float(np.mean(num / (denom + 1e-8)))


def aggregate_step2_stability(outputs_root: str, dataset: str):
    ds_key = dataset.replace("/", "_")
    runs = []
    for name in os.listdir(outputs_root):
        if not name.startswith("run_"):
            continue
        run_dir = os.path.join(outputs_root, name)
        params_path = os.path.join(run_dir, "params.json")
        if not os.path.exists(params_path):
            continue
        try:
            with open(params_path, "r", encoding="utf-8") as f:
                params = json.load(f)
        except Exception:
            continue
        if dataset not in params.get("datasets", []):
            continue
        seed = params.get("seed")
        step2_dir = os.path.join(run_dir, "step2")
        runs.append((seed, step2_dir))

    if len(runs) < 2:
        logger.warning("[STEP2] Not enough runs for stability analysis (%s).", dataset)
        return

    def collect_states(tag: str) -> List[Tuple[int, np.ndarray]]:
        out = []
        for seed, step2_dir in runs:
            path = os.path.join(step2_dir, f"student_state_{tag}_{ds_key}.npy")
            if os.path.exists(path):
                out.append((seed, np.load(path)))
        return out

    for tag in ["baseline", "debiased"]:
        states = collect_states(tag)
        if len(states) < 2:
            continue

        seeds = [s for s, _ in states]
        n = len(seeds)
        spearman_mat = np.full((n, n), np.nan)
        proc_mat = np.full((n, n), np.nan)

        for i in range(n):
            for j in range(n):
                if i == j:
                    spearman_mat[i, j] = 1.0
                    proc_mat[i, j] = 1.0
                    continue
                a = states[i][1]
                b = states[j][1]
                if a.shape != b.shape:
                    continue
                spearman_mat[i, j] = _spearman(a.mean(axis=1), b.mean(axis=1))
                proc_mat[i, j] = _procrustes_similarity(a, b)

        out_dir = os.path.join(outputs_root, "step2_aggregate", ds_key)
        os.makedirs(out_dir, exist_ok=True)

        pd.DataFrame(spearman_mat, index=seeds, columns=seeds).to_csv(
            os.path.join(out_dir, f"stability_spearman_{tag}.csv")
        )
        pd.DataFrame(proc_mat, index=seeds, columns=seeds).to_csv(
            os.path.join(out_dir, f"stability_procrustes_{tag}.csv")
        )

        plt = _require_matplotlib()
        for mat, name in [(spearman_mat, "spearman"), (proc_mat, "procrustes")]:
            plt.figure(figsize=(5, 4))
            plt.imshow(mat, vmin=0, vmax=1, cmap="viridis")
            plt.colorbar()
            plt.xticks(np.arange(n), seeds, rotation=45)
            plt.yticks(np.arange(n), seeds)
            plt.title(f"{dataset} {tag} {name}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"stability_{name}_{tag}.png"), dpi=150)
            plt.close()

    logger.info("[STEP2] stability aggregated for %s", dataset)
