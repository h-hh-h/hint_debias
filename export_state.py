import argparse
import json
import os
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from .columns import build_treatment_A, infer_cols
from .data_io import ensure_downloaded, pick_main_csv
from .dr import crossfit_dr_pseudolabel, fit_propensity, fit_outcome
from .features import make_features, temporal_split_by_student
from .logging_utils import get_logger, setup_logging
from .ncdm import _get_torch_net, _find_student_embedding, train_and_eval_ncdm
from .qmatrix import build_q_matrix

logger = get_logger()


def _pick_csv(args: argparse.Namespace) -> str:
    if args.csv:
        return args.csv
    if not args.dataset:
        raise SystemExit("Need --csv or --dataset.")
    folder = ensure_downloaded(args.dataset, args.data_root)
    return pick_main_csv(folder)


def _normalize_id(col: pd.Series, keep_raw: bool) -> Tuple[pd.Series, Optional[Dict[str, int]]]:
    if keep_raw:
        return col, None
    v = pd.to_numeric(col, errors="coerce")
    if v.notna().all():
        return v.astype(int), None
    codes, uniques = pd.factorize(col.astype(str), sort=True)
    mapping = {str(u): int(i) for i, u in enumerate(uniques)}
    return pd.Series(codes, index=col.index).astype(int), mapping


def _to_timestamp_seconds(raw: pd.Series) -> Tuple[pd.Series, str]:
    ts_num = pd.to_numeric(raw, errors="coerce")
    if ts_num.notna().all():
        return ts_num.astype(np.int64), "numeric"
    ts_dt = pd.to_datetime(raw, errors="coerce")
    if ts_dt.notna().any():
        ts_s = (ts_dt.astype("int64") // 1_000_000_000).astype(np.int64)
        return ts_s, "datetime"
    return pd.Series(np.arange(len(raw), dtype=np.int64), index=raw.index), "row_order"

def _apply_item_acc_prefix(d_sorted: pd.DataFrame, X: pd.DataFrame, train_mask: np.ndarray) -> None:
    item_acc = d_sorted.loc[train_mask].groupby("_i")["_y"].mean()
    X["item_acc"] = d_sorted["_i"].map(item_acc).fillna(0.5).values.astype(float)


def _dr_from_prefix(
    X: pd.DataFrame,
    A: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    outcome_model: str,
    eps: float,
) -> np.ndarray:
    X_use = X.drop(columns=["A_hint"], errors="ignore").values.astype(float)
    X_tr = X_use[train_mask]
    A_tr = A[train_mask]
    y_tr = y[train_mask]

    if len(y_tr) < 200:
        logger.warning("[WARN] prefix too small for DR; fallback to raw y.")
        return y.astype(float)

    g = fit_propensity(X_tr, A_tr)
    f0 = fit_outcome(X_tr[A_tr == 0], y_tr[A_tr == 0], model=outcome_model) if (A_tr == 0).sum() >= 50 else fit_outcome(X_tr, y_tr, model=outcome_model)

    e_hat = g.predict_proba(X_use)[:, 1]
    m0_hat = f0.predict_proba(X_use)[:, 1]
    e_hat = np.clip(e_hat, eps, 1 - eps)

    y0_tilde = m0_hat + ((1 - A) * (y - m0_hat)) / (1 - e_hat)
    y0_tilde = np.clip(y0_tilde, 0.0, 1.0)
    return y0_tilde.astype(float)


def export_state(args: argparse.Namespace) -> None:
    csv_path = _pick_csv(args)
    logger.info("[LOAD] %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False, encoding="latin1", on_bad_lines="skip")
    df["_row_id"] = np.arange(len(df), dtype=np.int64)

    cols = infer_cols(df)
    A = build_treatment_A(df, cols)
    X, user2idx, item2idx, skill2idx, d_sorted = make_features(df, cols, A)

    y = d_sorted["_y"].values.astype(int)
    A_sorted = d_sorted["_A"].values.astype(int)

    if args.strict_no_leak:
        train_mask, _test_mask = temporal_split_by_student(d_sorted, test_ratio=args.time_split)
        _apply_item_acc_prefix(d_sorted, X, train_mask)
    else:
        train_mask = np.ones(len(d_sorted), dtype=bool)

    if args.target == "y0_tilde":
        if args.strict_no_leak:
            y_target = _dr_from_prefix(X, A_sorted, y, train_mask, args.outcome_model, args.eps)
        else:
            y_target, _, _, _ = crossfit_dr_pseudolabel(
                X,
                A_sorted,
                y,
                kfold=args.kfold,
                eps=args.eps,
                clip01=True,
                seed=args.seed,
                outcome_model=args.outcome_model,
            )
    else:
        y_target = y.astype(float)

    Q = build_q_matrix(d_sorted, item2idx, skill2idx)

    u_idx = d_sorted["_u"].map(user2idx).values.astype(np.int64)
    i_idx = d_sorted["_i"].map(item2idx).values.astype(np.int64)
    k_mat = Q[i_idx].astype(np.float32)

    # ids for key
    user_raw = d_sorted[cols.user]
    user_num, user_map = _normalize_id(user_raw, args.keep_raw_ids)
    if args.keep_raw_ids:
        logger.warning("[WARN] keep_raw_ids=True may break ivrec lookup (expects int user_id).")

    ts_series, ts_mode = _to_timestamp_seconds(d_sorted[cols.order])
    if ts_mode != "numeric":
        logger.warning("[WARN] timestamp not numeric; converted via %s.", ts_mode)
    ts_arr = ts_series.values.astype(np.int64)

    # Train NCDM on target labels (debiased by default)
    n_user = len(user2idx)
    n_item = len(item2idx)
    n_skill = len(skill2idx)
    logger.info("[NCDM] users=%s items=%s skills=%s rows=%s", n_user, n_item, n_skill, len(d_sorted))
    if args.strict_no_leak:
        logger.info("[SPLIT] prefix rows=%s (ratio=%.2f)", int(train_mask.sum()), 1.0 - float(args.time_split))

    model, _ = train_and_eval_ncdm(
        train_u=u_idx[train_mask],
        train_i=i_idx[train_mask],
        train_k=k_mat[train_mask],
        train_y=y_target[train_mask].astype(np.float32),
        test_u=u_idx[train_mask],
        test_i=i_idx[train_mask],
        test_k=k_mat[train_mask],
        test_y=y_target[train_mask].astype(np.float32),
        n_user=n_user,
        n_item=n_item,
        n_skill=n_skill,
        epochs=args.ncdm_epochs,
        lr=args.ncdm_lr,
        device=args.device,
        batch_size=args.ncdm_batch_size,
        progress_desc="export_state",
    )

    # Dynamic state export (per-user, per-timestamp)
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:
        raise RuntimeError("PyTorch is required for state export.") from e

    net = _get_torch_net(model)
    if net is None:
        raise RuntimeError("Could not locate NCDM torch net for state export.")
    emb = _find_student_embedding(net, n_user)
    if emb is None:
        raise RuntimeError("Could not find student embedding for state export.")

    dev = torch.device(args.device)
    net.to(dev)
    net.eval()

    for p in net.parameters():
        p.requires_grad_(False)
    emb.weight.requires_grad_(True)

    opt = torch.optim.SGD([emb.weight], lr=args.state_lr)
    base_weight = emb.weight.detach().clone()

    states: Dict[str, np.ndarray] = {}
    state_records: List[Tuple[str, np.ndarray, float, int]] = []
    bias_arr = X["prior_hint20"].values.astype(float)
    last_u = None
    last_ts = None

    for idx in tqdm(range(len(u_idx)), desc="export_state", leave=False, dynamic_ncols=True):
        u = int(u_idx[idx])
        if last_u is None or u != last_u:
            with torch.no_grad():
                emb.weight[u].copy_(base_weight[u])
            last_u = u
            last_ts = None

        ts = int(ts_arr[idx])
        if last_ts is None or ts != last_ts:
            if args.keep_raw_ids:
                key = f"u{str(user_num.iloc[idx])}_t{ts}"
            else:
                key = f"u{int(user_num.iloc[idx])}_t{ts}"
            if key not in states:
                state_vec = emb.weight[u].detach().cpu().numpy().copy()
                if args.residualize:
                    state_records.append((key, state_vec, float(bias_arr[idx]), int(ts)))
                else:
                    states[key] = state_vec
            last_ts = ts

        uu = torch.as_tensor([u], dtype=torch.long, device=dev)
        ii = torch.as_tensor([int(i_idx[idx])], dtype=torch.long, device=dev)
        kk = torch.as_tensor(k_mat[idx : idx + 1], dtype=torch.float32, device=dev)
        yy = torch.as_tensor([float(y_target[idx])], dtype=torch.float32, device=dev)

        for _ in range(args.state_steps):
            opt.zero_grad(set_to_none=True)
            try:
                yhat = net(uu, ii, kk).view(-1)
            except TypeError:
                yhat = net((uu, ii, kk)).view(-1)
            yhat = torch.clamp(yhat, 1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(yhat, yy)
            loss.backward()
            opt.step()

    if args.residualize:
        if state_records:
            state_records.sort(key=lambda r: r[3])
            dim = int(state_records[0][1].shape[0])
            n = 0
            sum_b = 0.0
            sum_b2 = 0.0
            sum_s = np.zeros(dim, dtype=np.float64)
            sum_bs = np.zeros(dim, dtype=np.float64)
            for key, state_vec, bias_v, _ts in state_records:
                if n >= args.resid_min_n:
                    mean_b = sum_b / n
                    mean_s = sum_s / n
                    var = sum_b2 - n * mean_b * mean_b
                    denom = var + args.resid_lambda
                    cov = sum_bs - n * mean_b * mean_s
                    beta = cov / denom
                    alpha = mean_s - beta * mean_b
                    resid = state_vec - (alpha + beta * bias_v)
                    states[key] = resid.astype(np.float32)
                else:
                    states[key] = state_vec.astype(np.float32)
                sum_b += bias_v
                sum_b2 += bias_v * bias_v
                sum_s += state_vec
                sum_bs += state_vec * bias_v
                n += 1
        else:
            logger.warning("[WARN] No state records collected; skip residualization.")

    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    np.savez_compressed(args.out_npz, **states)
    logger.info("[SAVE] %s (%s states)", args.out_npz, len(states))

    maps = {}
    if user_map is not None:
        maps["user_id_map"] = user_map
    if maps:
        map_path = args.out_map or (args.out_npz + ".map.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(maps, f, ensure_ascii=False, indent=2)
        logger.info("[SAVE] %s", map_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--data_root", type=str, default="./edudata_cache")
    ap.add_argument("--out_npz", type=str, required=True)
    ap.add_argument("--out_map", type=str, default=None)
    ap.add_argument("--keep_raw_ids", action="store_true", default=False)
    ap.add_argument("--target", type=str, default="y0_tilde", choices=["y0_tilde", "y"])
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outcome_model", type=str, default="gbdt")
    ap.add_argument("--time_split", type=float, default=0.2)
    ap.add_argument("--no_strict_no_leak", dest="strict_no_leak", action="store_false", default=True)
    ap.add_argument("--ncdm_epochs", type=int, default=5)
    ap.add_argument("--ncdm_lr", type=float, default=1e-3)
    ap.add_argument("--ncdm_batch_size", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--state_lr", type=float, default=0.05)
    ap.add_argument("--state_steps", type=int, default=1)
    ap.add_argument("--no_residualize", dest="residualize", action="store_false", default=True)
    ap.add_argument("--resid_lambda", type=float, default=1.0)
    ap.add_argument("--resid_min_n", type=int, default=200)
    ap.add_argument("--log_path", type=str, default=None)
    args = ap.parse_args()

    if args.log_path is None:
        args.log_path = os.path.join("./outputs", "export_state.log")
    setup_logging(args.log_path)
    export_state(args)


if __name__ == "__main__":
    main()
