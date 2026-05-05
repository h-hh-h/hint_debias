import glob
import os
import subprocess
import zipfile
from typing import List

from .logging_utils import get_logger

logger = get_logger()


def run_cmd(cmd: List[str]) -> None:
    logger.info("[CMD] %s", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.stdout:
        logger.info(p.stdout.rstrip("\n"))
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def ensure_downloaded(dataset_name: str, root: str) -> str:
    out_dir = os.path.join(root, dataset_name)
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        return out_dir

    archive_zip = os.path.join(root, f"{dataset_name}.zip")
    if os.path.exists(archive_zip):
        os.makedirs(out_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(archive_zip, "r") as zf:
                zf.extractall(out_dir)
            logger.info("[DATA] extracted local archive by zipfile: %s -> %s", archive_zip, out_dir)
            return out_dir
        except Exception:
            # Some ZIPs can be listed/extracted by bsdtar but fail with zipfile.
            run_cmd(["tar", "-xf", archive_zip, "-C", out_dir])
            logger.info("[DATA] extracted local archive by tar: %s -> %s", archive_zip, out_dir)
            return out_dir

    os.makedirs(root, exist_ok=True)
    # EduData CLI:
    #   edudata download <dataset> [dir]
    run_cmd(["edudata", "download", dataset_name, out_dir])
    return out_dir


def pick_main_csv(folder: str) -> str:
    preferred = os.path.join(folder, "prepared_main.csv")
    if os.path.exists(preferred):
        return preferred

    ednet_user_csvs = glob.glob(os.path.join(folder, "**", "u*.csv"), recursive=True)
    if len(ednet_user_csvs) >= 100:
        raise FileNotFoundError(
            "Detected raw EdNet-KT3 user logs (u*.csv) without prepared_main.csv. "
            "Please run scripts/prepare_extra_datasets_for_hint_debias.py first."
        )

    prepared_csvs = sorted(glob.glob(os.path.join(folder, "**", "*prepared*.csv"), recursive=True))
    if prepared_csvs:
        return prepared_csvs[0]

    # Heuristic: pick the largest CSV file under folder
    csvs = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    if csvs:
        csvs = sorted(csvs, key=lambda p: os.path.getsize(p), reverse=True)
        return csvs[0]

    txts = glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True)
    if txts:
        txts = sorted(txts, key=lambda p: os.path.getsize(p), reverse=True)
        return txts[0]

    raise FileNotFoundError(f"No CSV/TXT found under {folder}")
