import argparse
import json
import os
import sys
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss
from tqdm import tqdm

from .cache import cache_key, cache_paths, load_cache, save_cache
from .columns import build_treatment_A, infer_cols
from .config import CACHE_VERSION, DEFAULT_DATASETS
from .data_io import ensure_downloaded, pick_main_csv
from .diagnostics import run_step1_diagnostics
from .dr import crossfit_dr_pseudolabel
from .features import make_features, temporal_split_by_student
from .logging_utils import get_logger, setup_logging
from .metrics import safe_acc, safe_auc, safe_rmse
from .ncdm import (
    finetune_ncdm,
    predict_ncdm,
    set_student_state_in_model,
    student_state_from_model,
    train_and_eval_ncdm,
)
from .qmatrix import build_q_matrix
from .step2 import aggregate_step2_stability, run_step2_per_dataset
from .step4 import run_step4_sensitivity

logger = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./edudata_cache")
    ap.add_argument("--datasets", type=str, nargs="*", default=DEFAULT_DATASETS)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_cache", action="store_true", default=True)
    ap.add_argument("--rebuild_cache", action="store_true", default=False)
    ap.add_argument("--run_step4", action="store_true", default=False)
    ap.add_argument(
        "--rep_debias_finetune_epochs",
        type=int,
        default=0,
        help="If >0, fine-tune debiased model after residualizing student embeddings.",
    )
    ap.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Output directory for this run (logs + summaries).",
    )
    ap.add_argument("--log_path", type=str, default=None)
    args = ap.parse_args()

    if args.run_dir is None:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = os.path.join("./outputs", f"run_{run_stamp}")
    os.makedirs(args.run_dir, exist_ok=True)
    if args.log_path is None:
        args.log_path = os.path.join(args.run_dir, "run.log")
    setup_logging(args.log_path)
    np.random.seed(args.seed)
    params = {
        "datasets": list(args.datasets),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "device": args.device,
        # "data_root": args.data_root,
        # "use_cache": bool(args.use_cache),
        # "rebuild_cache": bool(args.rebuild_cache),
        # "run_dir": args.run_dir,
        # "log_path": args.log_path,
    }
    with open(os.path.join(args.run_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    for ds in tqdm(args.datasets, desc="Datasets", leave=False, dynamic_ncols=True):
        logger.info("%s", "\n" + "=" * 80)
        logger.info("[DATASET] %s", ds)

        folder = ensure_downloaded(ds, args.data_root)
        csv_path = pick_main_csv(folder)
        logger.info("[LOAD] %s", csv_path)

        # ============== CACHE ==============
        key = cache_key(ds, csv_path, args)
        paths = cache_paths(args.data_root, key)

        loaded = None
        if args.use_cache and (not args.rebuild_cache):
            loaded = load_cache(paths)

        if loaded is not None:
            logger.info("[CACHE] hit: %s", paths["base"])
            d_sorted, X, arrays, maps, Q, meta = loaded

            user2idx = maps["user2idx"]
            item2idx = maps["item2idx"]
            skill2idx = maps["skill2idx"]

            y = arrays["y"].astype(int)
            A_s = arrays["A"].astype(int)
            train_mask = arrays["train_mask"].astype(bool)
            test_mask = arrays["test_mask"].astype(bool)

            # training labels
            y_train = arrays["y_train"].astype(np.float32)
            y_train_int = y_train.astype(int)
            y0_tilde_train = arrays["y0_tilde_train"].astype(np.float32)
            e_hat_train = arrays["e_hat_train"].astype(np.float32)
            m0_hat_train = arrays.get("m0_hat_train")
            if m0_hat_train is not None:
                m0_hat_train = m0_hat_train.astype(np.float32)

            # cached index arrays
            train_u = arrays["train_u"].astype(np.int64)
            train_i = arrays["train_i"].astype(np.int64)
            test_u = arrays["test_u"].astype(np.int64)
            test_i = arrays["test_i"].astype(np.int64)

            # test y/A from masks
            y_test = y[test_mask]
            A_test = A_s[test_mask]
            A_train = A_s[train_mask]
            X_train = X.loc[train_mask].reset_index(drop=True)
            X_test = X.loc[test_mask].reset_index(drop=True)

            # rebuild indices (keep behavior consistent with original)
            d_train = d_sorted.loc[train_mask].reset_index(drop=True)
            d_test = d_sorted.loc[test_mask].reset_index(drop=True)

            train_u = d_train["_u"].map(user2idx).values.astype(np.int64)
            train_i = d_train["_i"].map(item2idx).values.astype(np.int64)
            train_k = Q[train_i].astype(np.float32)  # (N_train, n_skill)

            test_u = d_test["_u"].map(user2idx).values.astype(np.int64)
            test_i = d_test["_i"].map(item2idx).values.astype(np.int64)
            test_k = Q[test_i].astype(np.float32)  # (N_test, n_skill)

        else:
            logger.info("[CACHE] miss -> preprocess: %s", paths["base"])

            # read CSV
            df = pd.read_csv(csv_path, low_memory=False, encoding="latin1", on_bad_lines="skip")
            cols = infer_cols(df)
            A = build_treatment_A(df, cols)

            # features and sorted table
            X, user2idx, item2idx, skill2idx, d_sorted = make_features(df, cols, A)
            y = d_sorted["_y"].values.astype(int)
            A_s = d_sorted["_A"].values.astype(int)

            # time split
            train_mask, test_mask = temporal_split_by_student(d_sorted, test_ratio=0.2)

            # train/test slices
            X_train = X.loc[train_mask].reset_index(drop=True)
            y_train_int = y[train_mask]
            A_train = A_s[train_mask]

            X_test = X.loc[test_mask].reset_index(drop=True)
            y_test = y[test_mask]
            A_test = A_s[test_mask]

            # DR pseudo-labels (cross-fitting on train)
            y0_tilde_train, e_hat_train, m0_hat_train, _ = crossfit_dr_pseudolabel(
                X_train, A_train, y_train_int, kfold=5, eps=0.05, clip01=True, seed=args.seed
            )
            y_train = y_train_int.astype(np.float32)

            # Q-matrix
            Q = build_q_matrix(d_sorted, item2idx, skill2idx)
            d_train = d_sorted.loc[train_mask].reset_index(drop=True)
            d_test = d_sorted.loc[test_mask].reset_index(drop=True)

            train_u = d_train["_u"].map(user2idx).values.astype(np.int64)
            train_i = d_train["_i"].map(item2idx).values.astype(np.int64)
            train_k = Q[train_i].astype(np.float32)  # (N_train, n_skill)

            test_u = d_test["_u"].map(user2idx).values.astype(np.int64)
            test_i = d_test["_i"].map(item2idx).values.astype(np.int64)
            test_k = Q[test_i].astype(np.float32)  # (N_test, n_skill)

            # save cache
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
                "seed": int(args.seed),
                "kfold": 5,
                "eps": 0.05,
                "split": 0.2,
                "cache_version": CACHE_VERSION,
            }
            save_cache(paths, d_sorted, X, arrays, maps, Q, meta)
            logger.info("[CACHE] saved: %s", paths["base"])

        # ============== step1 diagnostics ==============
        run_step1_diagnostics(
            X_train=X_train,
            A_train=A_train,
            e_hat_train=e_hat_train,
            y_train=y_train_int,
            m0_hat_train=m0_hat_train,
            X_test=X_test,
            A_test=A_test,
            run_dir=args.run_dir,
            eps=0.05,
        )

        # ============== training/eval ==============
        n_user, n_item, n_skill = len(user2idx), len(item2idx), len(skill2idx)
        logger.info(
            "[SIZES] users=%s items=%s skills=%s train=%s test=%s",
            n_user,
            n_item,
            n_skill,
            len(train_u),
            len(test_u),
        )
        logger.info("[HINT] train hint-rate=%.4f test hint-rate=%.4f", A_train.mean(), A_test.mean())

        nohint = A_test == 0
        if nohint.sum() < 1000:
            logger.warning("[WARN] Test no-hint samples small: %s => metrics may be noisy.", int(nohint.sum()))

        def check_indices(name, u, i, n_user, n_item):
            u_min, u_max = int(u.min()), int(u.max())
            i_min, i_max = int(i.min()), int(i.max())
            logger.info(
                "[CHECK:%s] u in [%s,%s] (n_user=%s) ; i in [%s,%s] (n_item=%s)",
                name,
                u_min,
                u_max,
                n_user,
                i_min,
                i_max,
                n_item,
            )
            assert u_min >= 0 and u_max < n_user, f"user index out of range: [{u_min},{u_max}] vs {n_user}"
            assert i_min >= 0 and i_max < n_item, f"item index out of range: [{i_min},{i_max}] vs {n_item}"

        check_indices("train", train_u, train_i, n_user, n_item)
        check_indices("test", test_u, test_i, n_user, n_item)
        logger.info(
            "[DEBUG] train_k: %s %s min/max: %s %s",
            train_k.dtype,
            train_k.shape,
            train_k.min(),
            train_k.max(),
        )
        assert train_k.ndim == 2 and train_k.shape[1] == n_skill, f"train_k shape wrong: {train_k.shape}, n_skill={n_skill}"
        assert np.isfinite(train_k).all()
        assert train_k.min() >= 0 and train_k.max() <= 1

        # 1) Baseline: train on raw correct
        logger.info("\n[TRAIN] baseline NCDM (raw correct)")
        base_model, base_proba = train_and_eval_ncdm(
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
            progress_desc="baseline",
        )

        # 2) Debiased: train on DR pseudo-label for Y(0)
        logger.info("\n[TRAIN] debiased NCDM (DR pseudo-label for Y(0))")
        deb_model, deb_proba = train_and_eval_ncdm(
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
            progress_desc="debiased",
        )

        # Metrics on no-hint test subset (aligned with target)
        auc_base = safe_auc(y_test[nohint], base_proba[nohint])
        auc_deb = safe_auc(y_test[nohint], deb_proba[nohint])
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
        acc_base = safe_acc(y_test[nohint], base_proba[nohint])
        acc_deb = safe_acc(y_test[nohint], deb_proba[nohint])
        rmse_base = safe_rmse(y_test[nohint], base_proba[nohint])
        rmse_deb = safe_rmse(y_test[nohint], deb_proba[nohint])

        logger.info("\n[RESULT] (test where A=0)")
        logger.info("  baseline  AUC=%.4f  LogLoss=%.4f  Acc=%.4f  RMSE=%.4f", auc_base, ll_base, acc_base, rmse_base)
        logger.info("  debiased  AUC=%.4f  LogLoss=%.4f  Acc=%.4f  RMSE=%.4f", auc_deb, ll_deb, acc_deb, rmse_deb)

        # -------------------------
        # Representation-layer post-debias (residualization)
        # -------------------------
        base_state = student_state_from_model(base_model, n_user, n_skill)
        deb_state = student_state_from_model(deb_model, n_user, n_skill)

        # Student-level hint propensity proxy: average e_hat on train per student
        train_u_idx = train_u
        e_mean = pd.Series(e_hat_train).groupby(train_u_idx).mean()
        hint_rate = pd.Series(A_train).groupby(train_u_idx).mean()
        # align to full user index
        e_vec = np.array([e_mean.get(i, e_mean.mean()) for i in range(n_user)], dtype=float)
        h_vec = np.array([hint_rate.get(i, hint_rate.mean()) for i in range(n_user)], dtype=float)

        def residualize(state: np.ndarray, bias: np.ndarray) -> Tuple[np.ndarray, float]:
            """
            Remove linear component explainable by bias variable.
            Return (state_resid, avg_abs_corr_before_after_proxy)
            """
            if state is None or state.ndim != 2:
                return None, float("nan")
            # Fit ridge per-dimension: state[:,k] ~ bias
            bias_ = bias.reshape(-1, 1)
            resid = np.zeros_like(state, dtype=float)
            for k in range(state.shape[1]):
                rg = Ridge(alpha=1.0)
                rg.fit(bias_, state[:, k])
                pred = rg.predict(bias_)
                resid[:, k] = state[:, k] - pred
            # report correlation on a simple proxy (mean mastery)
            before = np.corrcoef(state.mean(axis=1), bias)[0, 1]
            after = np.corrcoef(resid.mean(axis=1), bias)[0, 1]
            return resid, (float(before), float(after))

        deb_resid_metrics = None
        if deb_state is not None:
            deb_resid, corr_pair = residualize(deb_state, e_vec)
            logger.info("\n[REP-DEBIAS] residualize debiased student state by mean propensity e")
            logger.info("  corr(mean_state, e): before=%.4f after=%.4f", corr_pair[0], corr_pair[1])

            if deb_resid is not None and set_student_state_in_model(deb_model, deb_resid, n_user):
                if args.rep_debias_finetune_epochs and args.rep_debias_finetune_epochs > 0:
                    finetune_ncdm(
                        deb_model,
                        train_u,
                        train_i,
                        train_k,
                        y0_tilde_train.astype(np.float32),
                        n_user=n_user,
                        epochs=int(args.rep_debias_finetune_epochs),
                        device=args.device,
                    )

                deb_resid_proba = predict_ncdm(
                    deb_model,
                    test_u,
                    test_i,
                    test_k,
                    device=args.device,
                    progress_desc="debiased-resid",
                )
                auc_deb_resid = safe_auc(y_test[nohint], deb_resid_proba[nohint])
                acc_deb_resid = safe_acc(y_test[nohint], deb_resid_proba[nohint])
                rmse_deb_resid = safe_rmse(y_test[nohint], deb_resid_proba[nohint])
                ll_deb_resid = (
                    log_loss(y_test[nohint], np.clip(deb_resid_proba[nohint], 1e-6, 1 - 1e-6))
                    if nohint.sum() > 10
                    else float("nan")
                )
                deb_resid_metrics = {
                    "auc": auc_deb_resid,
                    "logloss": ll_deb_resid,
                    "acc": acc_deb_resid,
                    "rmse": rmse_deb_resid,
                }
                logger.info("\n[RESULT] debiased+resid (test where A=0)")
                logger.info(
                    "  AUC=%.4f  LogLoss=%.4f  Acc=%.4f  RMSE=%.4f",
                    auc_deb_resid,
                    ll_deb_resid,
                    acc_deb_resid,
                    rmse_deb_resid,
                )
            else:
                logger.warning("\n[REP-DEBIAS] Could not apply residualized state to model; skip residualized prediction.")
        else:
            logger.warning("\n[REP-DEBIAS] Could not extract student embeddings from EduCDM model; skip.")

        # ============== step2 metrics/plots ==============
        run_step2_per_dataset(
            ds=ds,
            run_dir=args.run_dir,
            base_state=base_state,
            deb_state=deb_state,
            e_hat_train=e_hat_train,
            A_train=A_train,
            train_u=train_u,
            test_u=test_u,
            base_proba=base_proba,
            deb_proba=deb_proba,
            y_test=y_test,
            A_test=A_test,
        )
        aggregate_step2_stability("./outputs", ds)

        if args.run_step4:
            run_step4_sensitivity(
                ds=ds,
                run_dir=args.run_dir,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train_int,
                y_test=y_test,
                A_train=A_train,
                A_test=A_test,
                train_u=train_u,
                train_i=train_i,
                train_k=train_k,
                test_u=test_u,
                test_i=test_i,
                test_k=test_k,
                n_user=n_user,
                n_item=n_item,
                n_skill=n_skill,
                epochs=args.epochs,
                device=args.device,
                seed=args.seed,
            )

        # Save quick artifacts
        out = {
            "dataset": ds,
            "csv": csv_path,
            "n_user": n_user,
            "n_item": n_item,
            "n_skill": n_skill,
            "test_nohint_auc_baseline": auc_base,
            "test_nohint_auc_debiased": auc_deb,
            "test_nohint_logloss_baseline": ll_base,
            "test_nohint_logloss_debiased": ll_deb,
            "test_nohint_acc_baseline": acc_base,
            "test_nohint_acc_debiased": acc_deb,
            "test_nohint_rmse_baseline": rmse_base,
            "test_nohint_rmse_debiased": rmse_deb,
            "train_hint_rate": float(A_train.mean()),
            "test_hint_rate": float(A_test.mean()),
            "rep_debias_finetune_epochs": int(args.rep_debias_finetune_epochs),
        }
        if deb_resid_metrics is not None:
            out.update(
                {
                    "test_nohint_auc_debiased_resid": deb_resid_metrics["auc"],
                    "test_nohint_logloss_debiased_resid": deb_resid_metrics["logloss"],
                    "test_nohint_acc_debiased_resid": deb_resid_metrics["acc"],
                    "test_nohint_rmse_debiased_resid": deb_resid_metrics["rmse"],
                }
            )
        with open(
            os.path.join(args.run_dir, f"{ds.replace('/', '_')}_summary.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        logger.info("[SAVE] %s", os.path.join(args.run_dir, f"{ds.replace('/', '_')}_summary.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        setup_logging("./outputs/run.log")
        logger.exception("[FATAL] %r", e)
        sys.exit(1)
