"""Tests for the B1/B2 baselines, the frame sampler, and answer resolution.

These are hermetic: clips are synthesised with OpenCV, so the whole B1/B2 path is
exercised on a fresh checkout with no dataset and no torch. The synthetic numbers
are plumbing checks, never results.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skateid.baselines import (
    LinearProbeBaseline,
    MotionStatsEmbedder,
    available_embedders,
    available_vlms,
    build_embedder,
    build_vlm_prompt,
    resolve_answer,
    run_b1,
    run_b2,
)
from skateid.taxonomy import Taxonomy
from skateid.video import DEFAULT_FRAME_COUNT, sample_frames


def _write_clip(path: Path, frames: int = 12, size=(64, 48), speed: float = 1.0, seed: int = 0) -> Path:
    """Write a tiny motion clip: a bright square crossing a noisy background."""
    import cv2

    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (width, height))
    assert writer.isOpened(), "OpenCV could not open an MJPG writer in this build"
    background = np.random.RandomState(seed).randint(40, 90, size=(height, width, 3), dtype=np.uint8)
    for index in range(frames):
        canvas = background.copy()
        x = int(index * speed * 3) % max(width - 12, 1)
        cv2.rectangle(canvas, (x, height // 3), (x + 10, height // 3 + 8), (235, 235, 235), -1)
        writer.write(canvas)
    writer.release()
    assert path.exists() and path.stat().st_size > 0
    return path


def _manifest(rows, taxonomy: Taxonomy) -> pd.DataFrame:
    """Build a minimal manifest whose rows satisfy the taxonomy's requirements."""
    records = []
    for index, (path, label, split) in enumerate(rows):
        record = {
            "clip_id": f"test{index}",
            "dataset": "synthetic",
            "file_path": str(path),
            "sha256": f"{index:064d}",
            "label": label,
            "label_source": label,
            "license": "test-only",
            "split_holdout": split,
        }
        record.update(taxonomy.rotation_for_label(label).components())
        records.append(record)
    return pd.DataFrame(records)


@pytest.fixture
def taxonomy() -> Taxonomy:
    return Taxonomy.load()


@pytest.fixture
def synthetic_manifest(tmp_path, taxonomy) -> pd.DataFrame:
    """Two classes, one clip each side of the split."""
    rows = [
        (_write_clip(tmp_path / "c0.avi", speed=2.2, seed=0), "kickflip", "train"),
        (_write_clip(tmp_path / "c1.avi", speed=0.4, seed=1), "ollie", "train"),
        (_write_clip(tmp_path / "c2.avi", speed=2.0, seed=2), "kickflip", "test"),
        (_write_clip(tmp_path / "c3.avi", speed=0.5, seed=3), "ollie", "test"),
    ]
    return _manifest(rows, taxonomy)


# --- frame sampling --------------------------------------------------------


def test_sample_frames_shape_dtype_and_padding(tmp_path):
    clip = _write_clip(tmp_path / "clip.avi", frames=12)

    frames = sample_frames(clip, count=5, size=(32, 24))
    assert frames.shape == (5, 24, 32, 3), "size is (width, height), frames are (T, H, W, 3)"
    assert frames.dtype == np.uint8

    # A clip shorter than the requested grid is padded by repeating the last
    # frame, so every clip yields a fixed-shape array.
    padded = sample_frames(clip, count=40, size=(32, 24))
    assert padded.shape == (40, 24, 32, 3)
    assert np.array_equal(padded[-1], padded[-2])


def test_sample_frames_rejects_a_missing_file(tmp_path):
    with pytest.raises(OSError):
        sample_frames(tmp_path / "nope.mp4")


# --- B1: frozen embedder + linear probe ------------------------------------


def test_motion_stats_embedder_is_finite_and_reacts_to_motion():
    embedder = MotionStatsEmbedder()
    still = np.zeros((DEFAULT_FRAME_COUNT, 32, 32, 3), dtype=np.uint8)
    vector = embedder.embed(still)
    assert vector.shape == (embedder.dim,)
    assert np.isfinite(vector).all()

    moving = still.copy()
    moving[DEFAULT_FRAME_COUNT // 2 :] = 255
    assert not np.allclose(vector, embedder.embed(moving)), "features must see the motion"


def test_motion_stats_embedder_handles_a_single_frame():
    vector = MotionStatsEmbedder().embed(np.zeros((1, 16, 16, 3), dtype=np.uint8))
    assert vector.shape == (MotionStatsEmbedder.dim,)
    assert np.isfinite(vector).all()


def test_embedder_registry_reports_what_is_usable():
    statuses = available_embedders()
    assert statuses["motion_stats"][0], "the dependency-free floor must always be usable"
    with pytest.raises(KeyError):
        build_embedder("not_an_embedder")

    # Optional backends either work, or say exactly what is missing. Never silent.
    usable, reason = statuses["videomae"]
    if not usable:
        assert "missing optional dependencies" in reason
        with pytest.raises(RuntimeError):
            build_embedder("videomae")


def test_linear_probe_separates_separable_data():
    rng = np.random.RandomState(0)
    features = np.vstack([rng.normal(0, 0.3, (10, 4)), rng.normal(4, 0.3, (10, 4))])
    labels = ["kickflip"] * 10 + ["ollie"] * 10

    probe = LinearProbeBaseline().fit(features, labels)
    assert probe.classes_ == ["kickflip", "ollie"]
    assert probe.predict(features) == labels


def test_count_tag_separates_sampling_grids():
    """Features depend on the sampling grid, so the cache key must depend on it too."""
    from skateid.baselines import count_tag

    assert count_tag(16, (112, 112)) != count_tag(16, (224, 224))
    assert count_tag(8, (112, 112)) != count_tag(16, (112, 112))
    assert count_tag(16, (112, 112)) == "16f_112x112"


def test_backend_declares_its_required_input_geometry():
    """Fixed-geometry backbones must state it, so callers cannot feed a wrong grid."""
    from skateid.baselines import TorchvisionEmbedder, VideoMAEEmbedder

    assert MotionStatsEmbedder.input_size is None, "motion_stats is size-agnostic"
    assert TorchvisionEmbedder.input_size == (224, 224)
    assert VideoMAEEmbedder.input_size == (224, 224)
    # VideoMAE's temporal position embeddings are fixed at its training length:
    # 8 frames gives 784 positions where the model has 1568.
    assert VideoMAEEmbedder.input_frames == 16


def test_run_b1_end_to_end_on_synthetic_clips(tmp_path, synthetic_manifest):
    result = run_b1(
        synthetic_manifest,
        embedder="motion_stats",
        split_col="split_holdout",
        cache_dir=tmp_path / "cache",
        log=None,
    )
    assert result["status"] == "ok", result.get("reason")
    assert result["dim"] == MotionStatsEmbedder.dim
    assert (result["n_train"], result["n_test"]) == (2, 2)
    assert result["clips_missing"] == 0
    assert set(result["predictions"]) <= {"kickflip", "ollie"}
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    # Features are cached per clip, so re-scoring does not re-decode the videos.
    # The directory carries the sampling grid, so a later run at another resolution
    # cannot silently reuse these.
    from skateid.baselines import count_tag

    assert (tmp_path / "cache" / f"motion_stats_{count_tag(16, (112, 112))}" / "test0.npy").exists()


def test_run_b1_skips_an_unknown_backend(tmp_path, synthetic_manifest):
    result = run_b1(
        synthetic_manifest, embedder="not_an_embedder", cache_dir=tmp_path / "cache", log=None
    )
    assert result["status"] == "skipped"
    assert "unknown embedder" in result["reason"]


def test_run_b1_skips_when_no_clip_is_on_disk(tmp_path, taxonomy):
    manifest = _manifest(
        [
            (tmp_path / "absent0.avi", "kickflip", "train"),
            (tmp_path / "absent1.avi", "ollie", "test"),
        ],
        taxonomy,
    )
    result = run_b1(manifest, embedder="motion_stats", cache_dir=tmp_path / "cache", log=None)
    assert result["status"] == "skipped"
    assert "on disk" in result["reason"]


# --- B2: VLM zero-shot -----------------------------------------------------


def test_vlm_prompt_offers_only_rotation_expressible_names(taxonomy):
    prompt = build_vlm_prompt(taxonomy)
    for name in taxonomy.dictionary.expressible_names():
        assert name in prompt, f"{name} must be offered to the model"
    # Rotation-free entries cannot be checked against a triple, so they are not
    # offered as answerable names.
    assert "impossible" not in prompt
    assert "half_cab" not in prompt
    assert available_vlms()["mock"][0] is True


def test_resolve_answer_is_word_bounded(taxonomy):
    assert resolve_answer("treflip", taxonomy) == "tre_flip"
    assert resolve_answer("a backside flip", taxonomy) == "bs_180_kickflip"
    assert resolve_answer("it is a 360 flip I think", taxonomy) == "tre_flip"
    assert resolve_answer("kick flip", taxonomy) == "kickflip"

    # Off-vocabulary text is an abstention, not a guess.
    assert resolve_answer("unknown", taxonomy) is None
    assert resolve_answer("", taxonomy) is None
    assert resolve_answer("it was sick", taxonomy) is None
    # "nollie" is a stance, not a trick: the word boundary stops it becoming ollie.
    assert resolve_answer("nollie", taxonomy) is None


def test_run_b2_scores_a_scripted_model(tmp_path, synthetic_manifest):
    scripted = ["kickflip", "ollie"]
    calls = {"count": 0}

    def answer(prompt, images):
        assert images and images[0][:2] == b"\xff\xd8", "frames must be JPEG-encoded"
        assert "tre_flip" in prompt, "the prompt must carry the vocabulary"
        value = scripted[calls["count"] % len(scripted)]
        calls["count"] += 1
        return value

    result = run_b2(
        synthetic_manifest, backend="mock", split_col="split_holdout", log=None, answer=answer
    )
    assert result["status"] == "ok", result.get("reason")
    assert result["n_test"] == 2
    assert result["abstentions"] == 0
    assert result["resolved_rate"] == 1.0
    assert result["metrics"]["accuracy"] == 1.0, "the scripted answers match the labels"
    assert result["answers"][0]["raw"] == "kickflip"


def test_run_b2_counts_abstentions_as_wrong(tmp_path, synthetic_manifest):
    result = run_b2(
        synthetic_manifest,
        backend="mock",
        split_col="split_holdout",
        log=None,
        answer="it was sick",
    )
    assert result["status"] == "ok"
    assert result["abstentions"] == result["n_test"]
    assert result["resolved_rate"] == 0.0
    assert set(result["predictions"]) == {"unknown"}
    assert result["metrics"]["accuracy"] == 0.0


def test_run_b2_reports_a_transport_error_instead_of_crashing(tmp_path, synthetic_manifest):
    """A dead local server must be a one-line message, not a 60-line traceback."""

    def boom(prompt, images):
        raise ConnectionRefusedError("No connection could be made")

    result = run_b2(
        synthetic_manifest, backend="mock", split_col="split_holdout", log=None, answer=boom
    )
    assert result["status"] == "failed"
    assert "errored on" in result["reason"]
    assert "No connection" in result["reason"]
    # It stops at the first failure rather than repeating the request 112 times.
    assert result["n_scored"] == 0


def test_run_b2_skips_a_backend_without_credentials(tmp_path, synthetic_manifest):
    if os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is set, so the missing-key path cannot be exercised")
    result = run_b2(synthetic_manifest, backend="openai", log=None)
    assert result["status"] == "skipped"
    assert "OPENAI_API_KEY" in result["reason"]
