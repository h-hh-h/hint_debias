import argparse
import hashlib
import json
import os
import pickle
from typing import Dict

import numpy as np
import pandas as pd

from .config import CACHE_SUBDIR, CACHE_VERSION


def file_fingerprint_fast(path: str) -> str:
    """Fast fingerprint: file size + mtime for local caching."""
    st = os.stat(path)
    s = f"{st.st_size}-{int(st.st_mtime)}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def cache_key(dataset_name: str, csv_path: str, args: argparse.Namespace) -> str:
    """Build cache key from preprocessing inputs to avoid stale caches."""
    fp = file_fingerprint_fast(csv_path)
    propensity_model = str(getattr(args, "propensity_model", "lr"))
    user_keep_ratio = float(getattr(args, "dataset_user_keep_ratio", 1.0))
    user_keep_map = str(getattr(args, "dataset_user_keep_map", ""))
    user_sample_seed = int(getattr(args, "dataset_user_sample_seed", getattr(args, "seed", 42)))
    sample_cfg = f"{dataset_name}|r={user_keep_ratio:.8g}|m={user_keep_map}|s={user_sample_seed}"
    sample_cfg_sig = hashlib.md5(sample_cfg.encode("utf-8")).hexdigest()[:10]
    key = (
        f"{dataset_name}__{fp}__{CACHE_VERSION}__seed{args.seed}"
        f"__k5_eps0.05__split0.2__ps{propensity_model}__us{sample_cfg_sig}"
    )
    return key


def cache_paths(root: str, key: str) -> Dict[str, str]:
    base = os.path.join(root, CACHE_SUBDIR, key)
    os.makedirs(base, exist_ok=True)
    return {
        "base": base,
        "meta": os.path.join(base, "meta.json"),
        "d_sorted": os.path.join(base, "d_sorted.parquet"),
        "X": os.path.join(base, "X.parquet"),
        "arrays": os.path.join(base, "arrays.npz"),
        "maps": os.path.join(base, "maps.pkl"),
        "Q": os.path.join(base, "Q.npy"),
    }


def save_cache(
    paths: Dict[str, str],
    d_sorted: pd.DataFrame,
    X: pd.DataFrame,
    arrays: Dict[str, np.ndarray],
    maps: Dict,
    Q: np.ndarray,
    meta: Dict,
):
    d_sorted.to_parquet(paths["d_sorted"], index=False)
    X.to_parquet(paths["X"], index=False)
    np.savez_compressed(paths["arrays"], **arrays)
    with open(paths["maps"], "wb") as f:
        pickle.dump(maps, f)
    np.save(paths["Q"], Q)
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_cache(paths: Dict[str, str]):
    need = ["d_sorted", "X", "arrays", "maps", "Q"]
    if not all(os.path.exists(paths[k]) for k in need):
        return None
    d_sorted = pd.read_parquet(paths["d_sorted"])
    X = pd.read_parquet(paths["X"])
    arrays = dict(np.load(paths["arrays"]))
    with open(paths["maps"], "rb") as f:
        maps = pickle.load(f)
    Q = np.load(paths["Q"])
    meta = None
    if os.path.exists(paths["meta"]):
        with open(paths["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
    return d_sorted, X, arrays, maps, Q, meta
