"""Manifest generation, schema definition, and dataset split logic."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

MANIFEST_COLUMNS = [
    "clip_id",
    "dataset",
    "file_path",
    "sha256",
    "label",
    "skater_id",
    "skater_id_source",
    "camera_id",
    "duration_sec",
    "frame_count",
    "fps",
    "width",
    "height",
    "split_published",
    "split_holdout",
]

def compute_sha256(path: Path) -> str:
    """Compute sha256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def inspect_video_metadata(path: Path) -> Tuple[int, float, int, int, float]:
    """Inspect video file returning (frame_count, fps, width, height, duration_sec)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0, 0.0, 0, 0, 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return frame_count, fps, width, height, duration

def generate_skater_disjoint_split(
    df: pd.DataFrame,
    skater_col: str = "skater_id",
    test_size: float = 0.25,
    seed: int = 42,
) -> pd.Series:
    """Generate a split where skater_id is disjoint across train/test splits."""
    import numpy as np

    skaters = sorted(df[skater_col].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(skaters)

    n_test = max(1, int(len(skaters) * test_size))
    test_skaters = set(skaters[:n_test])

    return df[skater_col].apply(lambda s: "test" if s in test_skaters else "train")


SKATEBOARDML_TAR_URL = "https://codeload.github.com/LightningDrop/SkateboardML/tar.gz/refs/heads/master"


def fetch_skateboardml(dest_dir: Path | str = "data/raw/skateboardml", force: bool = False) -> Path:
    """Download and extract the SkateboardML dataset tarball.

    Skips the download when clips already exist locally, so the manifest can be
    rebuilt cheaply. Pass ``force=True`` to re-download.
    """
    import tarfile
    import urllib.request

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    existing = [p for p in dest.rglob("*") if p.is_file() and p.suffix.lower() == ".mov"]
    if existing and not force:
        print(f"Found {len(existing)} existing clips in {dest}; skipping download (use force=True to refresh).")
        return dest

    print(f"Streaming SkateboardML archive from {SKATEBOARDML_TAR_URL}...")
    req = urllib.request.Request(
        SKATEBOARDML_TAR_URL,
        headers={"User-Agent": "SkateID/0.3.0"},
    )
    with urllib.request.urlopen(req) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                name = member.name
                if name.startswith("SkateboardML-master/Tricks/") and member.isfile():
                    subpath = name.replace("SkateboardML-master/Tricks/", "")
                    target_path = dest / subpath
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(target_path, "wb") as f_out:
                        f_in = tar.extractfile(member)
                        if f_in:
                            f_out.write(f_in.read())
                elif name in ("SkateboardML-master/trainlist02.txt", "SkateboardML-master/testlist02.txt"):
                    target_path = dest / Path(name).name
                    with open(target_path, "wb") as f_out:
                        f_in = tar.extractfile(member)
                        if f_in:
                            f_out.write(f_in.read())
    print(f"Extracted SkateboardML clips into {dest}")
    return dest


def build_manifest(raw_dir: Path | str = "data/raw/skateboardml", out_csv: Path | str = "data/manifest.csv") -> pd.DataFrame:
    """Build standardized manifest DataFrame and save to CSV."""
    import glob
    import re
    raw_path = Path(raw_dir)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clips = sorted(list(set(glob.glob(str(raw_path / "**" / "*.mov"), recursive=True) + glob.glob(str(raw_path / "**" / "*.MOV"), recursive=True))))
    if not clips:
        raise FileNotFoundError(f"No video clips found under {raw_path}")

    train_txt = raw_path / "trainlist02.txt"
    test_txt = raw_path / "testlist02.txt"
    pub_train = set()
    pub_test = set()
    if train_txt.exists():
        with open(train_txt) as f:
            pub_train = {l.strip().lower().replace("\\", "/") for l in f if l.strip()}
    if test_txt.exists():
        with open(test_txt) as f:
            pub_test = {l.strip().lower().replace("\\", "/") for l in f if l.strip()}

    rows = []
    for c in clips:
        norm_path = c.replace("\\", "/")
        parts = norm_path.split("/")
        rel_path = f"{parts[-2]}/{parts[-1]}"
        trick_label = parts[-2].lower()

        sha = compute_sha256(Path(c))
        frames, fps, w, h, dur = inspect_video_metadata(Path(c))

        clip_id = f"sbml_{Path(c).stem.lower()}"
        nums = re.findall(r"\d+", Path(c).name)
        num = int(nums[0]) if nums else 0
        # NOTE: SkateboardML does not publish skater identities, so this ID is a
        # synthetic bucket over the clip number. It must never be presented as a
        # real person identifier; see `skater_id_source` below.
        skater_id = f"skater_{num % 8:02d}"
        skater_id_source = "synthetic_clip_number"

        if rel_path.lower() in pub_train:
            split_pub = "train"
        elif rel_path.lower() in pub_test:
            split_pub = "test"
        else:
            split_pub = "train"

        # PLACEHOLDER holdout split (~25% test). Because skater_id is synthetic
        # this is NOT a genuine skater-disjoint split and must not be reported as
        # a leakage-free benchmark. A real clean split arrives with datasets that
        # ship per-skater labels.
        split_holdout = "test" if skater_id in ("skater_00", "skater_01") else "train"

        rows.append({
            "clip_id": clip_id,
            "dataset": "skateboardml",
            "file_path": norm_path,
            "sha256": sha,
            "label": trick_label,
            "skater_id": skater_id,
            "skater_id_source": skater_id_source,
            "camera_id": f"cam_{w}x{h}",
            "duration_sec": round(dur, 2),
            "frame_count": frames,
            "fps": round(fps, 1),
            "width": w,
            "height": h,
            "split_published": split_pub,
            "split_holdout": split_holdout,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df

