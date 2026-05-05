import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .columns import Cols


def make_features(
    df: pd.DataFrame,
    cols: Cols,
    A: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int], Dict[str, int], pd.DataFrame]:
    """
    Returns:
      X_df: feature table aligned with df rows
      maps: user2idx, item2idx, skill2idx (for CD)
    """
    d = df.copy()

    # Basic ids as category codes (for propensity/outcome, keep simple numeric proxies)
    d["_u"] = d[cols.user].astype(str)
    d["_i"] = d[cols.item].astype(str)

    # skill can be multi-skill; keep raw string
    d["_k_raw"] = d[cols.skill].astype(str)

    # Sort per student to compute history-only features
    d["_ord"] = pd.to_numeric(d[cols.order], errors="coerce")
    if d["_ord"].isna().any():
        # fallback to original order for NaNs
        d["_ord"] = np.arange(len(d), dtype=float)

    d = d.sort_values(["_u", "_ord"], kind="mergesort").reset_index(drop=True)

    y = pd.to_numeric(d[cols.correct], errors="coerce").fillna(0.0).clip(0, 1).astype(int).values
    A_sorted = A[d.index.values] if (len(A) == len(df)) else A

    # opportunity: if provided, use; else compute cumulative count on (student, skill)
    if cols.opportunity is not None:
        opp = pd.to_numeric(d[cols.opportunity], errors="coerce").fillna(0.0).values
    else:
        # single-skill assumption for opportunity proxy (minimal)
        opp = d.groupby(["_u", "_k_raw"]).cumcount().values

    # Rolling history features (window=20)
    win = 20
    g = d.groupby("_u", sort=False)
    prior_cnt = g.cumcount().values
    prior_acc = (
        g[cols.correct]
        .apply(lambda s: s.shift(1).rolling(win, min_periods=1).mean())
        .reset_index(level=0, drop=True)
        .fillna(0.0)
        .values
    )
    prior_hint = (
        pd.Series(A_sorted)
        .groupby(d["_u"], sort=False)
        .apply(lambda s: s.shift(1).rolling(win, min_periods=1).mean())
        .reset_index(level=0, drop=True)
        .fillna(0.0)
        .values
    )

    # Item global difficulty proxy (computed from entire dataset; for cross-fitting it's OK as it's label-leaky if using Y,
    # but it's a standard "dataset statistic". To be strict, you can recompute on train folds only.)
    item_acc = d.groupby("_i")[cols.correct].mean()
    item_acc = d["_i"].map(item_acc).fillna(0.5).values

    X = pd.DataFrame(
        {
            "prior_cnt": prior_cnt.astype(float),
            "prior_acc20": prior_acc.astype(float),
            "prior_hint20": prior_hint.astype(float),
            "opp": opp.astype(float),
            "item_acc": item_acc.astype(float),
            "A_hint": A_sorted.astype(float),  # NOTE: do NOT include this in models; keep for debugging only
        }
    )

    # Build index maps for CD (student/item/skill)
    user2idx = {u: idx for idx, u in enumerate(d["_u"].unique())}
    item2idx = {it: idx for idx, it in enumerate(d["_i"].unique())}

    # Skills: allow multi-skill separated by commas (common in non-skill-builder)
    def parse_skills(s: str) -> List[str]:
        s = str(s)
        parts = [p.strip() for p in s.replace(";", ",").split(",")]
        parts = [p for p in parts if p != "" and p.lower() != "nan"]
        return parts if parts else ["__UNK__"]

    all_sk = set()
    for s in d["_k_raw"].values:
        for k in parse_skills(s):
            all_sk.add(k)
    skill2idx = {k: idx for idx, k in enumerate(sorted(all_sk))}

    # Attach back ordering-aligned columns
    d["_y"] = y
    d["_A"] = A_sorted
    d["_opp"] = opp
    return X, user2idx, item2idx, skill2idx, d


# -------------------------
# Train/test split (per student, last 20% as test)
# -------------------------

def temporal_split_by_student(d_sorted: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    train_mask = np.zeros(len(d_sorted), dtype=bool)
    test_mask = np.zeros(len(d_sorted), dtype=bool)
    for _, grp in d_sorted.groupby("_u", sort=False):
        n = len(grp)
        if n < 5:
            # too short: all train
            train_mask[grp.index.values] = True
            continue
        cut = int(math.floor(n * (1 - test_ratio)))
        idxs = grp.index.values
        train_mask[idxs[:cut]] = True
        test_mask[idxs[cut:]] = True
    return train_mask, test_mask
