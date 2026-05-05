import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .cd_models import train_and_eval_cd
from .dr import crossfit_dr_pseudolabel
from .logging_utils import get_logger
from .metrics import safe_acc, safe_auc, safe_rmse

logger = get_logger()


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for step4 plots.") from exc
    return plt


def _select_features(X: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    cols = ["prior_cnt", "prior_acc20", "prior_hint20", "opp", "item_acc"]
    if feature_set == "base":
        keep = cols
    elif feature_set == "history_only":
        keep = ["prior_cnt", "prior_acc20", "prior_hint20"]
    elif feature_set == "history_item":
        keep = ["prior_cnt", "prior_acc20", "prior_hint20", "item_acc"]
    elif feature_set == "history_opp":
        keep = ["prior_cnt", "prior_acc20", "prior_hint20", "opp"]
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    return X[keep].copy()


def _weighted_mean_var(x: np.ndarray, w: np.ndarray):
    w = np.asarray(w).astype(float)
    x = np.asarray(x).astype(float)
    w_sum = np.sum(w)
    if w_sum <= 0:
        return float("nan"), float("nan")
    mean = np.sum(w * x) / w_sum
    var = np.sum(w * (x - mean) ** 2) / w_sum
    return float(mean), float(var)


def _smd_table(X: pd.DataFrame, A: np.ndarray, weights: Optional[np.ndarray] = None) -> pd.DataFrame:
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


def _plot_love(smd: pd.DataFrame, path: str):
    plt = _require_matplotlib()
    smd = smd.sort_values("smd_unweighted", ascending=True)
    y = np.arange(len(smd))
    plt.figure(figsize=(6, max(3, 0.25 * len(smd))))
    plt.scatter(smd["smd_unweighted"], y, label="unweighted", s=16)
    plt.scatter(smd["smd_weighted"], y, label="ipw", s=16)
    plt.yticks(y, smd["feature"].tolist(), fontsize=8)
    plt.axvline(0.1, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("SMD")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _config_id(cfg: Dict) -> str:
    parts = [
        f"group={cfg['group']}",
        f"eps={cfg['eps']}",
        f"trim={cfg['trim']}",
        f"outcome={cfg['outcome_model']}",
        f"feat={cfg['feature_set']}",
        f"cd={cfg['cd_model']}",
    ]
    return "__".join(parts)


def build_step4_configs() -> List[Dict]:
    base = {
        "eps": 0.05,
        "trim": None,
        "outcome_model": "gbdt",
        "feature_set": "base",
        "cd_model": "ncdm",
    }
    configs: List[Dict] = []

    configs.append({**base, "group": "baseline"})

    for eps in [0.01, 0.05, 0.1]:
        configs.append({**base, "group": "eps", "eps": eps})

    for trim in [0.9, 0.95]:
        configs.append({**base, "group": "trim", "trim": trim})

    for om in ["gbdt", "lr", "mlp"]:
        configs.append({**base, "group": "outcome", "outcome_model": om})

    for fs in ["base", "history_only", "history_item"]:
        configs.append({**base, "group": "feature", "feature_set": fs})

    for cd in ["ncdm", "kancd", "irt", "dkt"]:
        configs.append({**base, "group": "cd", "cd_model": cd})

    # deduplicate
    seen = set()
    uniq = []
    for c in configs:
        key = (c["eps"], c["trim"], c["outcome_model"], c["feature_set"], c["cd_model"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq
    # return [{
    #     "group": "cd",
    #     "eps": 0.05,
    #     "trim": None,
    #     "outcome_model": "gbdt",
    #     "feature_set": "base",
    #     "cd_model": "kancd",
    # }]



def run_step4_sensitivity(
    ds: str,
    run_dir: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    A_train: np.ndarray,
    A_test: np.ndarray,
    train_u: np.ndarray,
    train_i: np.ndarray,
    train_k: np.ndarray,
    test_u: np.ndarray,
    test_i: np.ndarray,
    test_k: np.ndarray,
    n_user: int,
    n_item: int,
    n_skill: int,
    epochs: int,
    device: str,
    seed: int,
):
    step_dir = os.path.join(run_dir, "step4")
    os.makedirs(step_dir, exist_ok=True)

    results = []
    configs = build_step4_configs()
    for cfg in tqdm(configs, desc="Step4 configs", leave=False, dynamic_ncols=True):
        cfg_id = _config_id(cfg)
        out_dir = os.path.join(step_dir, cfg_id)
        os.makedirs(out_dir, exist_ok=True)

        try:
            Xtr = _select_features(X_train, cfg["feature_set"])
            Xte = _select_features(X_test, cfg["feature_set"])

            y0_tilde_train, e_hat_train, _, _ = crossfit_dr_pseudolabel(
                Xtr,
                A_train,
                y_train,
                kfold=5,
                eps=cfg["eps"],
                clip01=True,
                seed=seed,
                outcome_model=cfg["outcome_model"],
            )

            mask = np.ones_like(y_train, dtype=bool)
            if cfg["trim"] is not None:
                t = float(cfg["trim"])
                lo, hi = 1 - t, t
                mask = (e_hat_train >= lo) & (e_hat_train <= hi)

            Xtr_use = Xtr.loc[mask].reset_index(drop=True)
            y_tr = y_train[mask].astype(np.float32)
            y0_tr = y0_tilde_train[mask].astype(np.float32)
            A_tr = A_train[mask]
            u_tr = train_u[mask]
            i_tr = train_i[mask]
            k_tr = train_k[mask]

            if len(y_tr) < 100:
                logger.warning("[STEP4] too few samples after trim, skip: %s", cfg_id)
                continue

            # weight/overlap stats
            e_clip = np.clip(e_hat_train, cfg["eps"], 1 - cfg["eps"])
            w_ipw = np.where(A_train == 1, 1.0 / e_clip, 1.0 / (1.0 - e_clip))
            overlap_stats = {
                "eps": float(cfg["eps"]),
                "trim": cfg["trim"],
                "train_pct_lt_eps": float(np.mean(e_hat_train < cfg["eps"])),
                "train_pct_gt_1m_eps": float(np.mean(e_hat_train > (1 - cfg["eps"]))),
            }
            weight_stats = {
                "w_mean": float(np.mean(w_ipw)),
                "w_std": float(np.std(w_ipw)),
                "w_max": float(np.max(w_ipw)),
                "w_p95": float(np.percentile(w_ipw, 95)),
                "w_p99": float(np.percentile(w_ipw, 99)),
            }
            with open(os.path.join(out_dir, "overlap_stats.json"), "w", encoding="utf-8") as f:
                json.dump(overlap_stats, f, ensure_ascii=False, indent=2)
            with open(os.path.join(out_dir, "weight_stats.json"), "w", encoding="utf-8") as f:
                json.dump(weight_stats, f, ensure_ascii=False, indent=2)

            smd_unw = _smd_table(Xtr, A_train, weights=None).rename(columns={"smd": "smd_unweighted"})
            smd_w = _smd_table(Xtr, A_train, weights=w_ipw).rename(columns={"smd": "smd_weighted"})
            smd = smd_unw[["feature", "smd_unweighted"]].merge(
                smd_w[["feature", "smd_weighted"]], on="feature", how="left"
            )
            smd.to_csv(os.path.join(out_dir, "balance_smd.csv"), index=False)
            _plot_love(smd, os.path.join(out_dir, "love_plot.png"))

            # train CD model
            base_model, base_proba = train_and_eval_cd(
                cfg["cd_model"],
                u_tr,
                i_tr,
                k_tr,
                y_tr,
                test_u,
                test_i,
                test_k,
                y_test,
                n_user,
                n_item,
                n_skill,
                epochs=epochs,
                device=device,
                progress_desc=f"{cfg['cd_model']}-base",
            )
            deb_model, deb_proba = train_and_eval_cd(
                cfg["cd_model"],
                u_tr,
                i_tr,
                k_tr,
                y0_tr,
                test_u,
                test_i,
                test_k,
                y_test,
                n_user,
                n_item,
                n_skill,
                epochs=epochs,
                device=device,
                progress_desc=f"{cfg['cd_model']}-deb",
            )

            nohint = A_test == 0
            auc_base = safe_auc(y_test[nohint], base_proba[nohint])
            auc_deb = safe_auc(y_test[nohint], deb_proba[nohint])
            acc_base = safe_acc(y_test[nohint], base_proba[nohint])
            acc_deb = safe_acc(y_test[nohint], deb_proba[nohint])
            rmse_base = safe_rmse(y_test[nohint], base_proba[nohint])
            rmse_deb = safe_rmse(y_test[nohint], deb_proba[nohint])

            from sklearn.metrics import log_loss

            ll_base = (
                log_loss(y_test[nohint], np.clip(base_proba[nohint], 1e-6, 1 - 1e-6))
                if nohint.sum() > 10
                else float("nan")
            )
            ll_deb = (
                log_loss(y_test[nohint], np.clip(deb_proba[nohint], 1e-6, 1 - 1e-6))
                if nohint.sum() > 10
                else float("nan")
            )

            metrics = {
                "dataset": ds,
                "group": cfg["group"],
                "eps": cfg["eps"],
                "trim": cfg["trim"],
                "outcome_model": cfg["outcome_model"],
                "feature_set": cfg["feature_set"],
                "cd_model": cfg["cd_model"],
                "auc_base": auc_base,
                "auc_deb": auc_deb,
                "logloss_base": ll_base,
                "logloss_deb": ll_deb,
                "acc_base": acc_base,
                "acc_deb": acc_deb,
                "rmse_base": rmse_base,
                "rmse_deb": rmse_deb,
            }

            with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            results.append(metrics)
        except Exception as exc:
            logger.exception("[STEP4] failed config %s", cfg_id)
            with open(os.path.join(out_dir, "error.txt"), "w", encoding="utf-8") as f:
                f.write(repr(exc))
            continue

    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(step_dir, "step4_summary.csv"), index=False)
        df.to_json(os.path.join(step_dir, "step4_summary.json"), orient="records", indent=2)
        logger.info("[STEP4] saved summary to %s", step_dir)
