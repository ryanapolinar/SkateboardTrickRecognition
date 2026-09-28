"""Tests for data validation, allowlist enforcement, and scope guardrails."""

import json
from pathlib import Path
import pandas as pd
import pytest

from skateid.eval import MajorityClassBaseline, evaluate_predictions, format_confusion_matrix


def test_allowlist_exists_and_valid():
    allowlist_path = Path("data/flatground_allowlist.csv")
    assert allowlist_path.exists(), "flatground_allowlist.csv must exist"
    df = pd.read_csv(allowlist_path)
    assert "canonical_name" in df.columns
    assert "category" in df.columns
    assert len(df) >= 10
    assert "ollie" in df["canonical_name"].values
    assert "kickflip" in df["canonical_name"].values


def test_tricks_json_consistency():
    tricks_path = Path("data/tricks.json")
    assert tricks_path.exists(), "tricks.json must exist"
    with open(tricks_path) as f:
        tricks = json.load(f)

    allowlist_path = Path("data/flatground_allowlist.csv")
    allowlist = pd.read_csv(allowlist_path)
    valid_names = set(allowlist["canonical_name"].values)

    for trick_name, props in tricks.items():
        assert trick_name in valid_names, f"{trick_name} not found in allowlist"
        assert "flip" in props
        assert "board_spin" in props
        assert "body_spin" in props
        assert "display" in props


def test_majority_baseline_logic():
    baseline = MajorityClassBaseline()
    labels = ["kickflip", "kickflip", "heelflip", "ollie", "kickflip"]
    baseline.fit(labels)
    assert baseline.majority_class == "kickflip"
    preds = baseline.predict(3)
    assert preds == ["kickflip", "kickflip", "kickflip"]

    metrics = evaluate_predictions(["kickflip", "heelflip", "kickflip"], preds)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert len(metrics["labels"]) == 2
    cm_str = format_confusion_matrix(metrics["confusion_matrix"], metrics["labels"])
    assert "kickflip" in cm_str


def test_manifest_schema_and_values():
    manifest_path = Path("data/manifest.csv")
    assert manifest_path.exists(), "data/manifest.csv must exist"
    df = pd.read_csv(manifest_path)
    required_cols = [
        "clip_id", "dataset", "file_path", "sha256", "label",
        "skater_id", "skater_id_source", "camera_id", "duration_sec",
        "frame_count", "fps", "width", "height", "split_published",
        "split_holdout",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing {col} in manifest"

    assert len(df) == 222, f"Expected 222 clips, found {len(df)}"
    assert set(df["label"].unique()) == {"kickflip", "ollie"}
    assert set(df["split_published"].unique()) == {"train", "test"}
    assert set(df["split_holdout"].unique()) == {"train", "test"}
    assert df["sha256"].nunique() == 222, "Expected 222 unique sha256 digests"

    # The skater IDs are a documented placeholder, not real identities.
    assert set(df["skater_id_source"].unique()) == {"synthetic_clip_number"}

    # Partition check: split_holdout assigns every clip to exactly one side.
    train_holdout = set(df[df["split_holdout"] == "train"]["clip_id"])
    test_holdout = set(df[df["split_holdout"] == "test"]["clip_id"])
    assert len(train_holdout & test_holdout) == 0, "holdout partitions must not overlap"
    assert len(train_holdout | test_holdout) == len(df), "holdout must cover every clip"

