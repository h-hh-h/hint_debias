from typing import Dict, List

import numpy as np
import pandas as pd


def build_k_index_per_row(d_sorted: pd.DataFrame, skill2idx: Dict[str, int]) -> np.ndarray:
    """Per-row skill id (minimal runnable: pick first skill)."""

    def first_skill(s: str) -> str:
        s = str(s)
        parts = [p.strip() for p in s.replace(";", ",").split(",")]
        parts = [p for p in parts if p and p.lower() != "nan"]
        return parts[0] if parts else "__UNK__"

    unk = skill2idx.get("__UNK__", 0)
    k_idx = d_sorted["_k_raw"].map(first_skill).map(lambda x: skill2idx.get(x, unk)).values
    return k_idx.astype(np.int64)


# -------------------------
# Q-matrix build (item -> multi-skill)
# -------------------------

def build_q_matrix(d_sorted: pd.DataFrame, item2idx: Dict[str, int], skill2idx: Dict[str, int]) -> np.ndarray:
    def parse_skills(s: str) -> List[str]:
        s = str(s)
        parts = [p.strip() for p in s.replace(";", ",").split(",")]
        parts = [p for p in parts if p != "" and p.lower() != "nan"]
        return parts if parts else ["__UNK__"]

    n_item = len(item2idx)
    n_skill = len(skill2idx)
    Q = np.zeros((n_item, n_skill), dtype=np.float32)

    # aggregate skills per item
    tmp = d_sorted[["_i", "_k_raw"]].copy()
    tmp["_skills"] = tmp["_k_raw"].map(parse_skills)
    grp = tmp.groupby("_i")["_skills"].sum()  # list concat

    for it, skills in grp.items():
        ii = item2idx[it]
        for k in set(skills):
            if k in skill2idx:
                Q[ii, skill2idx[k]] = 1.0

    empty = Q.sum(axis=1) == 0
    if empty.any():
        unk = skill2idx.get("__UNK__", None)
        if unk is not None:
            Q[empty, unk] = 1.0
    return Q
