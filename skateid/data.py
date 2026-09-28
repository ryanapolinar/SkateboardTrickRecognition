"""Manifest generation, schema definition, and dataset split logic."""

from __future__ import annotations

import csv
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from skateid.taxonomy import Rotation, Taxonomy

# Neither upstream repository ships a licence file; GitHub's licence API returns
# 404 for both (checked 2026-09). These strings record the terms each project
# actually states, not a legal determination:
#   SkateboardML README: "free to use this data for academic purposes, provided
#   you cite this work" (Zenodo DOI 10.5281/zenodo.3986905).
#   SkateAI: no licence statement; its labels describe Battle at the Berrics
#   footage, which is copyrighted, so the derived clips stay local and are never
#   redistributed (data/raw/ is gitignored).
SKATEBOARDML_LICENSE = "academic-use-only; cite Zenodo 10.5281/zenodo.3986905"
SKATEAI_LICENSE = "research-only; BATB footage (c) The Berrics; labels via EduardoPach/SkateAI"


@lru_cache(maxsize=1)
def default_taxonomy() -> Taxonomy:
    """The shared rotation dictionary + name registry, loaded once per process."""
    return Taxonomy.load()


MANIFEST_COLUMNS = [
    "clip_id",
    "dataset",
    "file_path",
    "sha256",
    # Canonical flatground name, derived from the six rotation columns below
    # (plan sections 3-5). Never an upstream spelling.
    "label",
    # The spelling the upstream dataset used, kept for provenance. The scope
    # guardrail cross-checks it against the derived label, so a clip whose name
    # and components disagree cannot be ingested silently.
    "label_source",
    # The terms the source dataset states.
    "license",
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
    # Provenance of the grouping used to build `split_holdout`.
    "split_source",
    # Compositional trick metadata. Populated for datasets that publish it
    # (SkateAI); left empty for SkateboardML, which only ships a class folder.
    # Riding direction / pop type exactly as the upstream dataset published it
    # (SkateAI: regular, switch, fakie, nollie). NOT the goofy/regular stance that
    # fixes the sign convention: none of these values can be converted into a
    # stance, and `fakie` in particular is a riding direction, not a foot
    # forward. It is kept for provenance and for stratifying splits.
    "stance_published",
    # The resolved goofy/regular toggle that selects the sign frame (plan
    # section 3). Empty until M1's feature extractor exists, which is why the
    # guardrail refuses to derive a *new* name from rotations while it is empty.
    "stance_input",
    "landed",
    "flip_type",
    "flip_number",
    "board_rotation_type",
    "board_rotation_number",
    "body_rotation_type",
    "body_rotation_number",
    # Which source video a clip was cut from.
    "source_video_url",
    "source_video_title",
    "source_group",
    "clip_start",
    "clip_end",
]

# Columns that are legitimately absent for some datasets. Empty string for text,
# NaN for numbers, so pandas keeps numeric columns numeric across datasets.
_OPTIONAL_DEFAULTS: Dict[str, object] = {
    "stance_published": "",
    "stance_input": "",
    "landed": "",
    "flip_type": "",
    "flip_number": float("nan"),
    "board_rotation_type": "",
    "board_rotation_number": float("nan"),
    "body_rotation_type": "",
    "body_rotation_number": float("nan"),
    "source_video_url": "",
    "source_video_title": "",
    "source_group": "",
    "clip_start": float("nan"),
    "clip_end": float("nan"),
}

USER_AGENT = "SkateID/0.3.0"


def _manifest_row(**values: object) -> Dict[str, object]:
    """Build a manifest row in canonical column order, filling optional gaps."""
    row = {col: _OPTIONAL_DEFAULTS.get(col, "") for col in MANIFEST_COLUMNS}
    row.update(values)
    return row


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


def generate_group_disjoint_split(
    df: pd.DataFrame,
    group_col: str,
    test_size: float = 0.25,
    seed: int = 42,
) -> pd.Series:
    """Assign every clip of a group to the same side of the split.

    Unlike :func:`generate_skater_disjoint_split`, this consumes a grouping key
    that actually exists in the data (for SkateAI, the BATB source video). No
    group is ever split across train and test, so a clip cannot be scored while a
    near-duplicate of it sits in the training set.

    Note this is *group*-disjoint, not *skater*-disjoint: BATB is a 1v1 bracket
    and competitors recur across battles, and the labels do not say which of the
    two skaters performed a given clip.
    """
    import numpy as np

    counts = df[group_col].value_counts()
    groups = sorted(counts.index.tolist())
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)

    target = len(df) * test_size
    test_groups: set = set()
    n_test = 0
    for group in groups:
        # Never move every group to test, or train becomes empty.
        if len(test_groups) + 1 >= len(groups):
            break
        if n_test + counts[group] <= target:
            test_groups.add(group)
            n_test += counts[group]

    if not test_groups:
        # No group fitted under the target (one source dominates the dataset);
        # fall back to the smallest group so the test side is still non-empty.
        test_groups.add(counts.sort_values().index[0])

    return df[group_col].apply(lambda value: "test" if value in test_groups else "train")


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
        headers={"User-Agent": USER_AGENT},
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


def build_skateboardml_manifest(raw_dir: Path | str = "data/raw/skateboardml") -> pd.DataFrame:
    """Build manifest rows for the SkateboardML clips already on disk."""
    import glob
    import re
    raw_path = Path(raw_dir)

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

        # Normalise the class folder to the canonical vocabulary, then back-fill
        # the rotation columns from the dictionary. SkateboardML publishes only a
        # folder name, so the triple is the dictionary's rather than the
        # dataset's -- which is what makes the union with SkateAI uniform, instead
        # of leaving this dataset with empty components. An unrecognised folder
        # fails here, before it can reach the manifest.
        taxonomy = default_taxonomy()
        label_source = trick_label
        label = taxonomy.normalize_label(label_source)
        rotation = taxonomy.rotation_for_label(label)

        rows.append(_manifest_row(
            clip_id=clip_id,
            dataset="skateboardml",
            file_path=norm_path,
            sha256=sha,
            label=label,
            label_source=label_source,
            license=SKATEBOARDML_LICENSE,
            skater_id=skater_id,
            skater_id_source=skater_id_source,
            camera_id=f"cam_{w}x{h}",
            duration_sec=round(dur, 2),
            frame_count=frames,
            fps=round(fps, 1),
            width=w,
            height=h,
            split_published=split_pub,
            split_holdout=split_holdout,
            split_source="synthetic_clip_number",
            **rotation.components(),
        ))

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
# --- SkateAI (BATB clips, 449 labelled cuts) ------------------------------
#
# SkateAI ships labels as data rather than video: `data/metadata/metadata.csv`
# lists every cut as (source video URL, interval, trick decomposition) and you
# are expected to re-cut the clips yourself. Its own downloader
# (`labeling_tool/generate_data.py`) is unusable on a 2026 stack -- it imports
# `pytube` (broken against current YouTube) and `moviepy.editor` (removed in
# moviepy 2.x), and pulls in `wandb`. We re-implement the same job with
# yt-dlp + ffmpeg.
#
# The source footage is Battle at the Berrics, which is copyrighted: keep the
# clips local for research and do not redistribute them (`data/raw/` is
# gitignored).

SKATEAI_RAW_BASE = "https://raw.githubusercontent.com/EduardoPach/SkateAI/main"
SKATEAI_LABEL_FILES = {
    "tricks_cut.json": "data/tricks_cut.json",
    "TRICK_NAMES.json": "data/TRICK_NAMES.json",
    "metadata.csv": "data/metadata/metadata.csv",
    "train_split.csv": "data/metadata/train_split.csv",
    "validation_split.csv": "data/metadata/validation_split.csv",
}
SKATEAI_VIDEOS_SUBDIR = "videos"
SKATEAI_SOURCES_SUBDIR = "_source"
SKATEAI_CLIP_HEIGHT = 480


def fetch_skateai(dest_dir: Path | str = "data/raw/skateai", force: bool = False) -> Path:
    """Download SkateAI's label files (metadata only, no video) into ``dest_dir``."""
    import urllib.request

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    for name, rel_path in SKATEAI_LABEL_FILES.items():
        target = dest / name
        if target.exists() and not force:
            print(f"Kept existing {target}")
            continue
        url = f"{SKATEAI_RAW_BASE}/{rel_path}"
        print(f"Downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req) as resp:
            target.write_bytes(resp.read())

    return dest


def _download_source_video(url: str, dest: Path) -> None:
    """Fetch one BATB source video at <=480p with yt-dlp."""
    import yt_dlp

    opts = {
        "format": f"bv*[height<={SKATEAI_CLIP_HEIGHT}]+ba/b[height<={SKATEAI_CLIP_HEIGHT}]/b",
        "outtmpl": str(dest),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 5,
        "fragment_retries": 5,
    }
    print(f"Downloading source video {url} -> {dest.name}")
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])


def _cut_clip(source: Path, dest: Path, start: float, end: float) -> None:
    """Cut ``[start, end]`` seconds out of ``source`` with ffmpeg."""
    import subprocess

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{float(start):.3f}",
        "-to", f"{float(end):.3f}",
        "-i", str(source),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        str(dest),
    ]
    subprocess.run(cmd, check=True)


def download_skateai_clips(
    raw_dir: Path | str = "data/raw/skateai",
    limit: int | None = None,
    max_sources: int | None = None,
    force: bool = False,
    keep_source: bool = False,
) -> List[Path]:
    """Cut the labelled SkateAI clips out of their BATB source videos.

    Every clip belonging to a source video is cut from a single download, so the
    449 clips cost 12 downloads rather than 449. Clips already on disk are
    skipped unless ``force=True``, which makes the job resumable.

    ``limit`` caps clips produced per source video and ``max_sources`` caps how
    many source videos are touched (the first N in sorted URL order); together
    they keep a smoke test to a single download. Requires ``ffmpeg`` on PATH.
    """
    raw = Path(raw_dir)
    meta_path = raw / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found; run fetch_skateai() first.")

    meta = pd.read_csv(meta_path)
    clips_root = raw / SKATEAI_VIDEOS_SUBDIR
    source_root = raw / SKATEAI_SOURCES_SUBDIR
    source_root.mkdir(parents=True, exist_ok=True)

    groups = list(meta.groupby("video_url", sort=True))
    if max_sources is not None:
        groups = groups[:max_sources]

    written: List[Path] = []
    for url, group in groups:
        pending = []
        for row in group.itertuples(index=False):
            clip_path = clips_root / row.video_title / row.video_file
            if force or not clip_path.exists():
                pending.append((clip_path, row.clip_start, row.clip_end))
        if limit is not None:
            pending = pending[:limit]
        if not pending:
            print(f"All {len(group)} clips already present for {url}; skipping")
            continue

        video_id = url.rstrip("/").split("=")[-1]
        source_path = source_root / f"{video_id}.mp4"
        if force or not source_path.exists():
            _download_source_video(url, source_path)

        for clip_path, start, end in pending:
            _cut_clip(source_path, clip_path, start, end)
            written.append(clip_path)
        print(f"Cut {len(pending)} clips from {source_path.name}")

        if not keep_source:
            source_path.unlink(missing_ok=True)

    return written


def _skateai_published_splits(raw: Path) -> Dict[Tuple[str, str], str]:
    """Map ``(video_title, video_file)`` to the split SkateAI's author published.

    The upstream CSVs carry a bare ``video_file``, but that name repeats across
    source videos (every battle folder has its own ``00001.mp4``), so we key on
    the pair, which is unique and non-overlapping across the two files. Note the
    published split stratifies on ``stance``/``landed`` only, so clips from one
    battle land on both sides of it.
    """
    splits: Dict[Tuple[str, str], str] = {}
    for filename, side in (("train_split.csv", "train"), ("validation_split.csv", "test")):
        path = raw / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for video_title, video_file in zip(df["video_title"], df["video_file"]):
            splits[(video_title, video_file)] = side
    return splits


def build_skateai_manifest(raw_dir: Path | str = "data/raw/skateai") -> pd.DataFrame:
    """Build manifest rows for the SkateAI clips already on disk."""
    raw = Path(raw_dir)
    meta_path = raw / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found; run fetch_skateai() first.")

    meta = pd.read_csv(meta_path)
    clips_root = raw / SKATEAI_VIDEOS_SUBDIR
    published = _skateai_published_splits(raw)
    holdout = generate_group_disjoint_split(meta, group_col="video_url")

    rows = []
    missing = 0
    for index, row in meta.iterrows():
        clip_path = clips_root / row["video_title"] / row["video_file"]
        if not clip_path.exists():
            missing += 1
            continue

        frames, fps, w, h, dur = inspect_video_metadata(clip_path)
        # SkateAI publishes both a jargon name and the decomposed rotations. The
        # rotations are the stable key -- its 31 names and 31 triples are in exact
        # 1:1 correspondence -- so derive the canonical label from them and keep
        # the upstream spelling in label_source for provenance. The scope
        # guardrail then cross-checks the two paths, which is what would catch a
        # clip whose name and components disagree.
        rotation = Rotation.from_components(
            str(row["flip_type"]).strip().lower(),
            row["flip_number"],
            str(row["board_rotation_type"]).strip().lower(),
            row["board_rotation_number"],
            str(row["body_rotation_type"]).strip().lower(),
            row["body_rotation_number"],
        )
        label = default_taxonomy().label_from_rotation(rotation)

        rows.append(_manifest_row(
            clip_id=f"skateai_{row['video_title']}_{Path(row['video_file']).stem}".lower(),
            dataset="skateai",
            file_path=str(clip_path).replace("\\", "/"),
            sha256=compute_sha256(clip_path),
            label=label,
            label_source=str(row["trick_name"]).strip().lower(),
            license=SKATEAI_LICENSE,
            # SkateAI never records which of the two competitors performed a clip,
            # so no real person ID exists at clip level.
            skater_id="unknown",
            skater_id_source="not_published_per_clip",
            camera_id=f"cam_{row['video_title']}",
            duration_sec=round(dur, 2),
            frame_count=frames,
            fps=round(fps, 1),
            width=w,
            height=h,
            split_published=published.get((row["video_title"], row["video_file"]), "train"),
            split_holdout=holdout.loc[index],
            split_source="source_video_url",
            # Provenance only: a riding direction / pop type, NOT the goofy/regular
            # stance that fixes the sign convention.
            stance_published=str(row["stance"]).strip().lower(),
            # Stays empty until M1 supplies a feature extractor that can resolve
            # the toggle. An honest gap beats a guessed one: guessing wrong flips
            # every sign, so kickflip and heelflip swap.
            stance_input="",
            landed="true" if bool(row["landed"]) else "false",
            source_video_url=row["video_url"],
            source_video_title=row["video_title"],
            source_group=row["video_source"],
            clip_start=float(row["clip_start"]),
            clip_end=float(row["clip_end"]),
            **rotation.components(),
        ))

    if missing:
        print(
            f"Skipped {missing} SkateAI clips that are not on disk yet "
            "(run download_skateai_clips())."
        )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def build_manifest(
    raw_dir: Path | str = "data/raw/skateboardml",
    out_csv: Path | str = "data/manifest.csv",
    skateai_dir: Path | str = "data/raw/skateai",
    datasets: Tuple[str, ...] = ("skateboardml", "skateai"),
) -> pd.DataFrame:
    """Build the combined manifest and save it to CSV.

    Datasets whose raw files are missing are skipped with a message, so a fresh
    checkout with only SkateboardML extracted still yields a usable manifest.

    Raises :class:`~skateid.taxonomy.ScopeError` when any row falls outside the
    flatground vocabulary, so the guardrail of plan section 5 is enforced on the
    way in rather than only in CI.
    """
    frames: List[pd.DataFrame] = []
    for name in datasets:
        try:
            if name == "skateboardml":
                frames.append(build_skateboardml_manifest(raw_dir))
            elif name == "skateai":
                frames.append(build_skateai_manifest(skateai_dir))
            else:
                raise ValueError(f"Unknown dataset '{name}'")
        except FileNotFoundError as exc:
            print(f"Skipping dataset '{name}': {exc}")

    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        raise FileNotFoundError(
            f"No clips found for any of {', '.join(datasets)}; run 'skateid fetch' first."
        )

    df = pd.concat(non_empty, ignore_index=True)

    # The flatground guardrail (plan section 5) is enforced here, at ingestion:
    # an off-allowlist label, an empty component, or a row whose rotations
    # contradict its own label is rejected before it can reach data/manifest.csv.
    default_taxonomy().validate_or_raise(df)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df



