import argparse
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .columns import build_treatment_A, infer_cols
from .data_io import ensure_downloaded, pick_main_csv
from .dr import crossfit_dr_pseudolabel
from .features import make_features
from .logging_utils import get_logger, setup_logging

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


def export_tilde(args: argparse.Namespace) -> None:
    pbar = tqdm(total=5, desc="export_tilde", leave=False, dynamic_ncols=True)
    csv_path = _pick_csv(args)
    logger.info("[LOAD] %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False, encoding="latin1", on_bad_lines="skip")
    df["_row_id"] = np.arange(len(df), dtype=np.int64)
    pbar.update(1)

    cols = infer_cols(df)
    A = build_treatment_A(df, cols)
    X, _, _, _, d_sorted = make_features(df, cols, A)
    y = d_sorted["_y"].values.astype(int)
    A_sorted = d_sorted["_A"].values.astype(int)
    pbar.update(1)

    y0_tilde, _, _, _ = crossfit_dr_pseudolabel(
        X,
        A_sorted,
        y,
        kfold=args.kfold,
        eps=args.eps,
        clip01=True,
        seed=args.seed,
        outcome_model=args.outcome_model,
    )
    d_sorted["tilde_y0"] = y0_tilde
    pbar.update(1)

    if "_row_id" in d_sorted.columns:
        d_out = d_sorted.sort_values("_row_id").reset_index(drop=True)
    else:
        d_out = d_sorted.reset_index(drop=True)

    out = pd.DataFrame()
    out["user_id"] = d_out[cols.user]
    out["skill_id"] = d_out[cols.skill]
    out["timestamp"] = d_out[cols.order]
    out["tilde_y0"] = d_out["tilde_y0"].astype(float)
    if args.include_correct and cols.correct in d_out.columns:
        out["correct"] = pd.to_numeric(d_out[cols.correct], errors="coerce").fillna(0.0).clip(0, 1)

    for c in ["assignment_id", "problem_set", "problem_set_id", "session_id"]:
        if c in d_out.columns:
            out[c] = d_out[c]

    user_map = None
    skill_map = None
    out["user_id"], user_map = _normalize_id(out["user_id"], args.keep_raw_ids)
    out["skill_id"], skill_map = _normalize_id(out["skill_id"], args.keep_raw_ids)
    pbar.update(1)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    logger.info("[SAVE] %s", args.out_csv)
    pbar.update(1)
    pbar.close()

    maps = {}
    if user_map is not None:
        maps["user_id_map"] = user_map
    if skill_map is not None:
        maps["skill_id_map"] = skill_map
    if maps:
        map_path = args.out_map or (args.out_csv + ".map.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(maps, f, ensure_ascii=False, indent=2)
        logger.info("[SAVE] %s", map_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--data_root", type=str, default="./edudata_cache")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_map", type=str, default=None)
    ap.add_argument("--keep_raw_ids", action="store_true", default=False)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outcome_model", type=str, default="gbdt")
    ap.add_argument("--include_correct", action="store_true", default=False)
    ap.add_argument("--log_path", type=str, default=None)
    args = ap.parse_args()

    if args.log_path is None:
        args.log_path = os.path.join("./outputs", "export_tilde.log")
    setup_logging(args.log_path)

    export_tilde(args)


if __name__ == "__main__":
    main()
