"""Tests for data validation, allowlist enforcement, and scope guardrails."""

import json
import tempfile
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

import skateid
from skateid.data import generate_group_disjoint_split
from skateid.eval import (
    MajorityClassBaseline,
    evaluate_predictions,
    format_confusion_matrix,
)
from skateid.features import (
    EXTRACTOR_VERSION,
    KEYPOINT_COUNT,
    KEYPOINT_NAMES,
    active_window,
    body_features,
    board_angle,
    board_axes,
    board_axis_angle,
    board_corners,
    board_features,
    board_features_from_angles,
    board_summary,
    cache_key,
    load_features,
    net_sweep,
    save_features,
    segment_board,
    unwrap_angles,
    windowed_sweep,
)
from skateid.stance import (
    CONFIDENCE_FLOOR,
    SUGGESTION_COLUMN,
    STANCE_COLUMN,
    StanceSuggestion,
    apply_to_manifest,
    suggest_from_land_foot,
    suggest_from_sequence,
    trick_for_both_stances,
)
from skateid.taxonomy import ROTATION_COLUMNS, Rotation, ScopeError, Taxonomy


def test_interpreter_is_supported():
    """Guard against running the suite with an interpreter the package rejects."""
    assert skateid.SUPPORTED_PYTHON == ">=3.10"
    assert sys.version_info[:2] >= skateid.MIN_PYTHON, (
        f"skateid requires Python {skateid.SUPPORTED_PYTHON} but tests are running "
        f"on {sys.version_info.major}.{sys.version_info.minor}"
    )


def test_allowlist_exists_and_valid():
    allowlist_path = Path("data/flatground_allowlist.csv")
    assert allowlist_path.exists(), "flatground_allowlist.csv must exist"
    df = pd.read_csv(allowlist_path)
    assert "canonical_name" in df.columns
    assert "category" in df.columns
    assert "aliases" in df.columns
    # The v1 vocabulary: every flatground trick the model can name, plus the
    # documented rotation-free entries (half_cab, full_cab, impossible, none) and
    # the six added only to close the stance mirror.
    assert len(df) == 45, f"expected 45 canonical names, found {len(df)}"
    assert df["canonical_name"].is_unique, "canonical names must be unique"
    assert set(df["category"]) == {"flatground"}, "v1 is flatground-only by definition"
    for name in ("ollie", "kickflip", "tre_flip", "bs_bigspin_kickflip"):
        assert name in set(df["canonical_name"]), f"{name} missing from the allowlist"


def test_tricks_json_is_the_rotation_dictionary():
    """Every entry either carries all three rotations or says why it cannot."""
    with open("data/tricks.json", encoding="utf-8") as handle:
        tricks = json.load(handle)

    entries = {name: props for name, props in tricks.items() if not name.startswith("_")}
    assert len(entries) == 45, f"expected 45 dictionary entries, found {len(entries)}"

    for name, props in entries.items():
        assert "display" in props, f"{name} has no display name"
        if props.get("rotation_expressible", True):
            for axis in ("flip", "board_spin", "body_spin"):
                assert axis in props, f"{name} is missing the {axis} axis"
                assert isinstance(props[axis], int) and not isinstance(props[axis], bool)
                assert abs(props[axis]) <= 3, f"{name}.{axis} exceeds the +/-3 quantizer clamp"
        else:
            assert props.get("note"), f"{name} is not expressible and must explain why"


def test_documented_sign_convention_matches_the_data():
    """backside is positive: the convention SkateAI's own components imply."""
    taxonomy = Taxonomy.load()
    assert taxonomy.rotation_for_label("tre_flip") == Rotation(flip=1, board_spin=2, body_spin=0)
    assert taxonomy.rotation_for_label("hardflip") == Rotation(flip=1, board_spin=-1, body_spin=0)
    assert taxonomy.rotation_for_label("inward_heelflip") == Rotation(flip=-1, board_spin=1, body_spin=0)
    assert taxonomy.rotation_for_label("laser_flip") == Rotation(flip=-1, board_spin=-2, body_spin=0)
    assert taxonomy.rotation_for_label("bs_180") == Rotation(flip=0, board_spin=1, body_spin=1)
    assert taxonomy.rotation_for_label("fs_360") == Rotation(flip=0, board_spin=-2, body_spin=-2)


def test_rotation_free_tricks_cannot_be_derived():
    """Half cab and friends are allowlisted but must fail loudly, not silently."""
    taxonomy = Taxonomy.load()
    for name in ("half_cab", "full_cab", "impossible", "none"):
        with pytest.raises(ScopeError):
            taxonomy.rotation_for_label(name)
    with pytest.raises(ScopeError) as excinfo:
        taxonomy.rotation_for_label("impossible")
    assert "no rotation triple" in str(excinfo.value)


def test_off_allowlist_spellings_are_rejected():
    taxonomy = Taxonomy.load()
    for raw in ("kickflop", "900", "grind", "boardslide", ""):
        with pytest.raises(ScopeError):
            taxonomy.normalize_label(raw)


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
        "clip_id", "dataset", "file_path", "sha256", "label", "label_source",
        "license",
        "skater_id", "skater_id_source", "camera_id", "duration_sec",
        "frame_count", "fps", "width", "height", "split_published",
        "split_holdout", "split_source", "stance_published", "stance_input", "landed",
        "flip_type", "flip_number", "board_rotation_type",
        "board_rotation_number", "body_rotation_type", "body_rotation_number",
        "source_video_url", "source_video_title", "source_group",
        "clip_start", "clip_end",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing {col} in manifest"

    assert df["clip_id"].nunique() == len(df), "clip_id must be unique"
    assert df["file_path"].nunique() == len(df), "file_path must be unique"

    # Every clip lands on exactly one side of both split columns, regardless of
    # which dataset it came from.
    for split_col in ("split_published", "split_holdout"):
        assert set(df[split_col].unique()) == {"train", "test"}
        train = set(df[df[split_col] == "train"]["clip_id"])
        test = set(df[df[split_col] == "test"]["clip_id"])
        assert len(train & test) == 0, f"{split_col} partitions must not overlap"
        assert len(train | test) == len(df), f"{split_col} must cover every clip"

    # SkateboardML is always present in this checkout.
    sbml = df[df["dataset"] == "skateboardml"]
    assert len(sbml) == 222, f"Expected 222 SkateboardML clips, found {len(sbml)}"
    assert set(sbml["label"].unique()) == {"kickflip", "ollie"}
    assert sbml["sha256"].nunique() == 222, "Expected 222 unique SkateboardML digests"

    # The SkateboardML skater IDs are a documented placeholder, not real
    # identities, and its holdout grouping is synthetic.
    assert set(sbml["skater_id_source"].unique()) == {"synthetic_clip_number"}
    assert set(sbml["split_source"].unique()) == {"synthetic_clip_number"}


# --- the scope guardrail (plan section 5) ----------------------------------
#
# Until this milestone the guardrail was only a promise in the plan: nothing read
# the allowlist or tricks.json, and `label` carried whatever spelling the upstream
# dataset used. These tests are the enforcement.


def test_manifest_passes_the_scope_guardrail():
    df = pd.read_csv("data/manifest.csv")
    taxonomy = Taxonomy.load()
    assert taxonomy.validate_frame(df) == []


def test_manifest_labels_are_canonical_not_upstream_spellings():
    df = pd.read_csv("data/manifest.csv")
    taxonomy = Taxonomy.load()
    canonicals = set(taxonomy.allowlist.canonical_names())

    assert set(df["label"]) <= canonicals

    # Upstream spellings survive in label_source so provenance is not lost, and
    # exactly three of SkateAI's 31 happen to already be canonical.
    skateai = df[df["dataset"] == "skateai"]
    assert set(skateai["label_source"]) & canonicals == {"kickflip", "heelflip", "hardflip"}
    assert "treflip" in set(skateai["label_source"])
    assert "tre_flip" in set(skateai["label"])
    assert "treflip" not in set(df["label"]), "upstream spelling leaked into label"


def test_manifest_has_fully_populated_rotations_for_every_dataset():
    """Both datasets carry a triple, so the union of them is uniform."""
    df = pd.read_csv("data/manifest.csv")
    assert df[list(ROTATION_COLUMNS)].notna().all().all(), "every row needs a full triple"
    for column in ("flip_number", "board_rotation_number", "body_rotation_number"):
        assert df[column].between(0, 3).all(), f"{column} is outside the quantizer clamp"

    # SkateboardML publishes only a folder name, so its triples come from the
    # dictionary -- which is exactly why they agree with the canonical rows.
    sbml = df[df["dataset"] == "skateboardml"]
    ollie = sbml[sbml["label"] == "ollie"].iloc[0]
    assert (ollie["flip_number"], ollie["board_rotation_number"], ollie["body_rotation_number"]) == (0, 0, 0)
    kickflip = sbml[sbml["label"] == "kickflip"].iloc[0]
    assert (kickflip["flip_type"], kickflip["flip_number"]) == ("kickflip", 1)


def test_manifest_records_licensing_terms():
    """Neither upstream repo ships a licence file, so the stated terms are recorded."""
    df = pd.read_csv("data/manifest.csv")
    assert df["license"].notna().all() and df["license"].str.len().gt(0).all()
    assert set(df["license"]) == {
        "academic-use-only; cite Zenodo 10.5281/zenodo.3986905",
        "research-only; BATB footage (c) The Berrics; labels via EduardoPach/SkateAI",
    }


def test_guardrail_rejects_off_allowlist_label():
    """The negative case: prove the gate actually closes."""
    df = pd.read_csv("data/manifest.csv")
    doctored = df.copy()
    doctored.loc[doctored.index[0], "label"] = "kickflop"  # not a trick at all
    violations = Taxonomy.load().validate_frame(doctored)
    assert any("off the flatground allowlist" in violation for violation in violations)


def test_guardrail_rejects_label_that_contradicts_its_rotations():
    df = pd.read_csv("data/manifest.csv")
    row = df.index[df["dataset"] == "skateboardml"][0]  # a kickflip folder
    doctored = df.copy()
    doctored.loc[row, "label"] = "heelflip"
    violations = Taxonomy.load().validate_frame(doctored)
    assert any("disagrees with its own rotations" in violation for violation in violations)


def test_guardrail_rejects_empty_impossible_and_inconsistent_components():
    df = pd.read_csv("data/manifest.csv")
    taxonomy = Taxonomy.load()
    row = df.index[0]

    blanked = df.copy()
    blanked.loc[row, "flip_type"] = None
    assert any("are empty" in violation for violation in taxonomy.validate_frame(blanked))

    over_rotated = df.copy()
    over_rotated.loc[row, "flip_number"] = 7
    assert any("exceeds" in violation for violation in taxonomy.validate_frame(over_rotated))

    inconsistent = df.copy()
    inconsistent.loc[row, "flip_type"] = "none"
    inconsistent.loc[row, "flip_number"] = 1
    assert any("cannot carry number" in violation for violation in taxonomy.validate_frame(inconsistent))


def test_guardrail_rejects_a_clip_whose_name_and_components_disagree():
    """The cross-check that makes normalising by name alone insufficient."""
    df = pd.read_csv("data/manifest.csv")
    skateai = df[df["dataset"] == "skateai"].copy()
    skateai.loc[skateai.index[0], "label_source"] = "heelflip"  # the row is a treflip
    violations = Taxonomy.load().validate_frame(skateai)
    assert any("resolves to" in violation for violation in violations)


# --- stance: the sign frame is a separate input (plan sections 3 and 7) -----


def test_mirroring_a_rotation_is_the_geometric_mirror():
    """A kickflip is +360 for a regular rider and -360 for a goofy one."""
    kickflip = Rotation(flip=1, board_spin=0, body_spin=0)
    assert kickflip.mirrored() == Rotation(flip=-1, board_spin=0, body_spin=0)
    assert Taxonomy.load().label_from_rotation(kickflip.mirrored()) == "heelflip"

    # All three axes flip, not just the board: the body yaw mirrors too.
    bigspin = Rotation(flip=0, board_spin=2, body_spin=1)
    assert bigspin.mirrored() == Rotation(flip=0, board_spin=-2, body_spin=-1)

    # Mirroring twice is the identity, so the operation is an involution.
    for rotation in (Rotation(2, -1, 1), Rotation(-1, 2, 0), Rotation()):
        assert rotation.mirrored().mirrored() == rotation


def test_stance_mirror_is_a_bijection_over_the_whole_vocabulary():
    """Every name mirrors to exactly one name, and every name is mirrored once.

    This is what makes plan section 7's mirror-with-label-swap augmentation
    total: there is no clip whose mirrored label has nowhere to land, and no two
    tricks that collide on the same partner. Six names were added purely to
    close the map (the frontside/backside partners of bigflip, biggerflip, the
    180 double flips, tre double flip and hard double flip); none is published by
    an ingested dataset, and each carries a `note` saying so.
    """
    taxonomy = Taxonomy.load()
    closure = taxonomy.mirror_closure()
    assert closure["unmirrored"] == [], closure["unmirrored"]

    names = taxonomy.dictionary.expressible_names()
    partners = {}
    for name in names:
        rotation = taxonomy.dictionary.rotation_for_label(name)
        partner = taxonomy.name_for_mirrored(rotation)
        assert partner is not None, f"{name} has no mirrored partner"
        partners.setdefault(partner, []).append(name)

    # Surjective (every name is hit) and injective (no name is hit twice).
    assert set(partners) == set(names)
    collisions = {key: value for key, value in partners.items() if len(value) > 1}
    assert collisions == {}, collisions

    # Spot-check the skate vocabulary, including the two self-mirrors.
    assert taxonomy.name_for_mirrored(Rotation(flip=1)) == "heelflip"
    assert taxonomy.name_for_mirrored(Rotation(flip=1, board_spin=2)) == "laser_flip"
    # bigflip (flip +1, bs 360, body 180) pairs with the pre-existing bigheel.
    assert taxonomy.name_for_mirrored(Rotation(flip=1, board_spin=2, body_spin=1)) == (
        "fs_bigspin_heelflip"
    )
    # ...while the newly added fs_bigspin_kickflip is the partner of the inward
    # heelflip bigspin, which is a different trick with the opposite flip axis.
    assert taxonomy.name_for_mirrored(Rotation(flip=-1, board_spin=2, body_spin=1)) == (
        "fs_bigspin_kickflip"
    )
    # ollie is the only self-mirror, and necessarily so: the all-zero triple is
    # its own mirror. Every other trick pairs with a different name -- a double
    # kickflip mirrors to a double heelflip, not to itself.
    self_mirrors = sorted(name for name, hits in partners.items() if name in hits)
    assert self_mirrors == ["ollie"]
    assert taxonomy.name_for_mirrored(Rotation(flip=2)) == "double_heelflip"


def test_stance_suggestions_never_silently_fill_the_manifest():
    """The single most dangerous quiet failure in this project, pinned down.

    A wrong stance does not crash: it negates every sign, swapping kick<->heel
    and fs<->bs on every clip. So the rules are that a suggestion is *recorded*
    but never *promoted* unless it clears a bar, and that a user override always
    wins. Both halves are asserted here.
    """
    frame = pd.DataFrame(
        {
            "clip_id": ["a", "b", "c"],
            "label": ["kickflip", "heelflip", "ollie"],
            "stance_input": ["", "", ""],
        }
    )

    # No input at all: nothing is written. The empty state is the safe default.
    assert apply_to_manifest(frame)[STANCE_COLUMN].isna().all()

    # A single frame's evidence is 0.5, and even a unanimous multi-frame vote is
    # capped at 0.75 -- both below CONFIDENCE_FLOOR, so neither is promoted.
    weak = suggest_from_land_foot(True)
    assert weak.stance == "regular" and not weak.confident
    unanimous = suggest_from_sequence([True] * 16)
    assert unanimous.stance == "regular"
    assert unanimous.confidence <= CONFIDENCE_FLOOR, "a unanimous vote still gets reviewed"
    assert apply_to_manifest(frame, suggestions={"a": unanimous})[STANCE_COLUMN].isna().all()

    # ...but the suggestion is still visible, which is the point of recording it.
    out = apply_to_manifest(frame, suggestions={"a": unanimous})
    assert out[SUGGESTION_COLUMN].iloc[0] == unanimous.confidence
    assert out[SUGGESTION_COLUMN].iloc[1] == "", "no suggestion for a clip that got none"

    # An override is the only path that writes, and it wins over a suggestion
    # that says the opposite.
    out = apply_to_manifest(
        frame, overrides={"a": "goofy"}, suggestions={"a": unanimous, "b": weak}
    )
    assert out[STANCE_COLUMN].iloc[0] == "goofy"

    # A riding direction cannot masquerade as a stance, here or via override.
    for bad in ("fakie", "switch", "nollie"):
        with pytest.raises(ValueError, match="toggle"):
            apply_to_manifest(frame, overrides={"a": bad})

    # A confident suggestion is promoted, so the mechanism is not dead code.
    strong = StanceSuggestion("regular", CONFIDENCE_FLOOR, "hand-checked")
    assert apply_to_manifest(frame, suggestions={"a": strong})[STANCE_COLUMN].iloc[0] == "regular"

    # And the result still passes the scope guardrail. Checked against the real
    # manifest rather than the toy frame above, since the guardrail also checks
    # the rotation columns the toy frame has no business having.
    taxonomy = Taxonomy.load("data/tricks.json", "data/flatground_allowlist.csv")
    manifest = pd.read_csv("data/manifest.csv", dtype={"clip_id": str})
    clip = manifest["clip_id"].iloc[0]
    updated = apply_to_manifest(manifest, overrides={clip: "goofy"})
    assert updated[STANCE_COLUMN].iloc[0] == "goofy"
    assert taxonomy.validate_frame(updated) == []
    # Every other row is still empty, so the guardrail sees no new violation and
    # the rest of the dataset is untouched.
    assert updated[STANCE_COLUMN].iloc[1:].isna().all()


def test_stance_vote_does_not_act_on_a_tie():
    """A tied vote is a coin flip wearing a lab coat. It must not be promoted."""
    tied = suggest_from_sequence([True, False, True, False])
    assert not tied.confident
    assert apply_to_manifest(
        pd.DataFrame({"clip_id": ["a"], "stance_input": [""]}), suggestions={"a": tied}
    )[STANCE_COLUMN].isna().all()

    # A majority is readable, and reports its own support.
    majority = suggest_from_sequence([True, True, True, False])
    assert majority.stance == "regular" and "3/4" in majority.reason


def test_stance_input_is_never_derived_from_published_riding_direction():
    """`stance_published=regular` means *natural stance*, not goofy-vs-regular.

    The two vocabularies share the word 'regular' and mean different things, and
    conflating them is the exact corruption the guardrail exists to stop. This
    asserts the module offers no path from one to the other.
    """
    import ast
    import skateid.stance as stance_module

    # No *code* in the module touches the published column. The AST is used
    # rather than a text search because the prose above explains, in detail,
    # exactly why that column must not be read -- and a grep would flag its own
    # explanation. Only identifiers and string literals outside docstrings count.
    tree = ast.parse(Path(stance_module.__file__).read_text(encoding="utf-8"))
    # Collect the *node* holding each docstring, so its text is compared by
    # identity rather than by value (get_docstring cleans and dedents, so the
    # value would not match the literal).
    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ) and node.body and isinstance(node.body[0], ast.Expr):
            value = node.body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                docstring_nodes.add(id(value))
    offenders = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "stance_published" in node.value
        and id(node) not in docstring_nodes
    ]
    assert offenders == [], f"the module reads the published column: {offenders}"

    # And the two columns are independent in practice: 144 SkateAI clips publish
    # 'regular' with no goofy/regular information recoverable from it.
    frame = pd.DataFrame(
        {"clip_id": ["a"], "stance_published": ["regular"], "stance_input": [""]}
    )
    assert apply_to_manifest(frame)[STANCE_COLUMN].isna().all()


def _synthetic_pose(scale: float = 1.0, offset: np.ndarray = np.zeros(2)) -> np.ndarray:
    """A plausible 17-keypoint pose, placed and scaled as requested.

    Laid out so the hips, shoulders and torso are all present -- `body_features`
    needs the torso length to define its unit, and a fixture without one would
    exercise only the fallback path.
    """
    canonical = {
        "nose": (0, -0.5), "left_eye": (-0.05, -0.5), "right_eye": (0.05, -0.5),
        "left_ear": (-0.1, -0.5), "right_ear": (0.1, -0.5),
        "left_shoulder": (-0.2, 0.0), "right_shoulder": (0.2, 0.0),
        "left_elbow": (-0.3, 0.3), "right_elbow": (0.3, 0.3),
        "left_wrist": (-0.35, 0.6), "right_wrist": (0.35, 0.6),
        "left_hip": (-0.15, 1.0), "right_hip": (0.15, 1.0),
        "left_knee": (-0.2, 1.5), "right_knee": (0.2, 1.5),
        "left_ankle": (-0.2, 2.0), "right_ankle": (0.2, 2.0),
    }
    out = np.zeros((KEYPOINT_COUNT, 3), dtype=np.float32)
    for index, name in enumerate(KEYPOINT_NAMES):
        x, y = canonical[name]
        out[index, :2] = np.array([x, y]) * scale * 100.0 + offset
        out[index, 2] = 0.9
    return out


def test_body_features_are_relative_and_record_missingness():
    """Plan section 7: measure everything relative to the rider, and scale it.

    The point is that camera distance and angle stop mattering, so a feature
    vector for a skater filling the frame must equal one for a skater far away --
    that is the whole claim of the representation.
    """
    near = _synthetic_pose(scale=1.0, offset=np.array([500.0, 300.0]))
    far = _synthetic_pose(scale=0.25, offset=np.array([50.0, 80.0]))

    a, b = body_features(near), body_features(far)
    assert a.shape == (KEYPOINT_COUNT * 3,)
    assert np.allclose(a, b, atol=0.05), "features must be translation and scale invariant"

    # Confidence is carried per keypoint, so the model can learn to discount an
    # unreliable one instead of being handed a confident average.
    assert a[2] > 0 and b[2] > 0

    # An occluded keypoint is recorded as missing (0,0,0), not averaged in. A
    # smooth, confident-looking coordinate for a hidden elbow is the failure.
    occluded = near.copy()
    elbow = KEYPOINT_NAMES.index("left_elbow")
    occluded[elbow, 2] = 0.05
    out = body_features(occluded)
    assert out[elbow * 3 : elbow * 3 + 3].tolist() == [0.0, 0.0, 0.0]

    # An unusable pose returns zeros rather than raising, and the origin/hip
    # fallback does not divide by zero.
    assert body_features(np.zeros((17, 3))).shape == (17 * 3,)
    assert body_features(None).tolist() == [0.0] * (KEYPOINT_COUNT * 3)


def test_board_box_carries_location_but_not_rotation():
    """A COCO skateboard box locates the board; it cannot measure its rotation.

    An axis-aligned box is the same shape for a board tilted 45 deg as for one
    flat, so it cannot be a rotation signal. This pins down that
    `board_features` reports location/size and says so, rather than handing a
    downstream model numbers that look like an angle and are not.
    """
    frame_shape = (480, 640, 3)
    box = np.array([100.0, 50.0, 300.0, 200.0])
    out = board_features(box, frame_shape)
    assert out.shape == (5,)
    assert np.isclose(out[0], 200.0 / 640)   # centre x, normalised
    assert np.isclose(out[1], 125.0 / 480)   # centre y
    assert np.isclose(out[2], 200.0 / 640)   # width
    assert np.isclose(out[3], 150.0 / 480)   # height
    assert out[4] == 1.0, "visible"

    # No board detected is a visibly empty frame, not a silent zero.
    missing = board_features(None, frame_shape)
    assert missing.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]

    # Rotation is not in there, by design. If this ever becomes true the module's
    # docstring and the M2 gate both need revisiting.
    assert len(out) == 5


def test_feature_cache_is_keyed_on_the_extractor_version():
    """M0's cache keyed on clip_id alone and silently reused stale features.

    Every downstream score still came out, so the bug was invisible except as
    inexplicably flat results. The version is part of the key so that changing the
    feature definition cannot reuse the previous run's numbers.
    """
    assert cache_key("clip1") != cache_key("clip1", EXTRACTOR_VERSION + 1)
    assert cache_key("clip1") == cache_key("clip1")

    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory)
        key = cache_key("clip1")
        body = np.arange(6, dtype=np.float32)
        board = np.arange(5, dtype=np.float32)
        save_features(cache, key, body, board)
        loaded = load_features(cache, key)
        assert np.array_equal(loaded[0], body) and np.array_equal(loaded[1], board)

        # A miss, and a corrupt file, both read as None rather than crashing a
        # 671-clip pass on clip 400.
        assert load_features(cache, "nope") is None
        (cache / f"{key}.npz").write_bytes(b"truncated")
        assert load_features(cache, key) is None


def test_every_trick_can_be_named_for_both_stances():
    """The stance ambiguity is a clean binary, and it is fully answerable.

    Rather than guessing goofy-vs-regular, name the trick under both readings. This
    works only because the mirror is a bijection over the whole dictionary, so
    this asserts that on every expressible name rather than trusting the claim.
    """
    taxonomy = Taxonomy.load("data/tricks.json", "data/flatground_allowlist.csv")
    closure = taxonomy.mirror_closure()
    assert closure["unmirrored"] == [], "mirror is not total; the both-stances claim fails"

    for name in taxonomy.dictionary.expressible_names():
        result = trick_for_both_stances(name, taxonomy)
        assert not result["unresolved"], name
        assert result["regular"] == name
        assert result["goofy"] in taxonomy.allowlist.canonical_names()

    # The pair is genuinely two different tricks, not one name echoed back.
    kickflip = trick_for_both_stances("kickflip", taxonomy)
    assert kickflip["regular"] == "kickflip"
    assert kickflip["goofy"] == "heelflip", kickflip
    assert kickflip["stance_dependent"]

    # And it round-trips: naming the goofy reading as regular gives the original.
    assert trick_for_both_stances(kickflip["goofy"], taxonomy)["goofy"] == "kickflip"

    # The mirror is an involution across the whole dictionary, so this is not a
    # happy accident on one example.
    for name in taxonomy.dictionary.expressible_names():
        result = trick_for_both_stances(name, taxonomy)
        assert trick_for_both_stances(result["goofy"], taxonomy)["goofy"] == name


def _synthetic_board_image(angles_degrees: List[float], size: int = 120) -> List[np.ndarray]:
    """Render a white board on black, rotated to each given angle.

    A synthetic board with a *known* angle is the only way to test the angle
    pipeline without trusting a real clip -- if the extractor cannot recover 45
    deg from a clean synthetic board, no amount of real footage will help, and
    the bug is in the code rather than the segmentation.
    """
    import cv2

    frames = []
    for angle in angles_degrees:
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        layer = frame.copy()
        cv2.rectangle(layer, (10, 52), (110, 68), (255, 255, 255), -1)
        # -angle, not angle: image rows increase downward, so a positive visual
        # rotation is a negative angle in atan2(dy, dx) terms. The extractor
        # measures 45.00 deg for a board at -45, which is a convention, not an
        # error -- but the fixture has to state which one it is using.
        matrix = cv2.getRotationMatrix2D((size / 2, size / 2), -angle, 1.0)
        frame = cv2.warpAffine(layer, matrix, (size, size))
        frames.append(frame)
    return frames


def test_board_angle_recovers_a_known_rotation():
    """The whole point of minAreaRect: a rotating board keeps its own angle.

    An axis-aligned bounding box would return 0 deg for every one of these. The
    rotated rectangle is what recovers the signal, so this asserts the recovery is
    accurate on inputs whose true angle is known exactly.
    """
    for true_angle in (0, 15, 30, 45, 60, -30, -60):
        frames = _synthetic_board_image([true_angle])
        box = np.array([0.0, 0.0, 120.0, 120.0])
        mask = segment_board(frames[0], box)
        assert mask is not None, f"no mask at {true_angle} deg"
        assert board_corners(mask) is not None, f"no corners at {true_angle} deg"
        measured = board_axis_angle(mask)
        # Angles are mod 180 because a rectangle has no facing; compare on the
        # shortest way round so 0 and 180 are the same orientation.
        delta = abs((measured - true_angle + 90.0) % 180.0 - 90.0)
        assert delta < 6.0, f"true {true_angle}, measured {measured:.1f}"


def test_signed_sweep_separates_a_flip_from_a_heelflip():
    """The mirror pair is separated by the *sign* of the sweep, not its size.

    This is the test the whole board stream exists to pass. A kickflip and a
    heelflip trace the same magnitude of rotation in opposite directions, so any
    feature that ignored direction -- a mean angle, an absolute sweep, a
    bounding-box aspect ratio -- would score them identically.
    """
    kick = _synthetic_board_image([0, 45, 90, 135, 179, -135, -90, -45])
    heel = _synthetic_board_image([0, -45, -90, -135, -179, 135, 90, 45])

    def measure(frames):
        angles = []
        for frame in frames:
            mask = segment_board(frame, np.array([0.0, 0.0, 120.0, 120.0]))
            angles.append(board_axis_angle(mask))
        return net_sweep(angles)

    forward, backward = measure(kick), measure(heel)
    assert forward > 0, f"kickflip should sweep positive, got {forward}"
    assert backward < 0, f"heelflip should sweep negative, got {backward}"
    assert abs(forward + backward) < 60, "mirror sweeps should be near-equal and opposite"
    # The synthetic sequence sweeps a full 360, and the magnitude must be right as
    # well as the sign -- a feature that only caught direction would still fail
    # here. This is the "one kickflip" the manifest's rotation field encodes.
    assert abs(abs(forward) - 360) < 60, f"one full rotation expected, got {forward}"


def test_angles_are_unwrapped_across_the_wrap_boundary():
    """A per-frame angle in (-90, 90] cannot be averaged or differenced naively.

    The board crosses the wrap point mid-flip, so the raw series looks like noise
    even when the motion is perfectly smooth. Unwrapping is what turns it back
    into a continuous sweep -- without it, the sign of the rotation is destroyed
    by arithmetic rather than by anything in the video.
    """
    smooth = [0, 30, 60, 89, 61, 30, 0, -30, -60, -89, -61, -30, 0]
    unwrapped = unwrap_angles(smooth)
    # A smooth rotation must produce monotonically rising then falling values, not
    # jumps of 180 at the boundary.
    assert max(abs(b - a) for a, b in zip(unwrapped, unwrapped[1:])) < 90, unwrapped
    # A rise-and-return series travels 360 deg of path even though it ends where it
    # started. Under the old endpoint-difference definition this read 0; the whole
    # point of the fix is that a rotation out and back is a real rotation. The
    # test below covers the case that must stay 0: a genuinely static board.
    assert net_sweep(smooth) == 360.0, "out and back is 360 deg of travel, not zero"
    assert net_sweep([5, 5, 5, 5]) == 0.0, "a static board has no rotation"
    assert unwrap_angles([]) == []


def test_segmentation_refuses_rather_than_guesses():
    """A failed measurement must be None, never a confident wrong angle.

    A blank box has no board in it. Returning a 0-degree angle for it would be
    indistinguishable from an ollie -- a board that genuinely did not rotate --
    and would poison the sweep for the whole clip.
    """
    blank = np.zeros((120, 120, 3), dtype=np.uint8)
    assert segment_board(blank, np.array([0.0, 0.0, 120.0, 120.0])) is None

    # A box too small to segment is also a refusal, not a division by zero.
    assert segment_board(blank, np.array([0.0, 0.0, 3.0, 3.0])) is None


def test_sweep_measures_a_closed_loop_not_the_endpoints():
    """A full flip ends where it started, so endpoint differencing reports ~0.

    This is the bug that survived two measurement rounds: `net_sweep` originally
    returned the endpoint difference, so a textbook kickflip -- the board rotating
    360 deg and arriving back at its starting angle -- scored as no rotation at
    all. Found by reading a real trajectory (223 deg travelled, 27 deg endpoint
    difference) rather than by any aggregate, and missed by the existing
    synthetic tests because that series happened to start and end apart.

    The starting and ending angle here are *identical on purpose*.
    """
    # 0 -> 180 -> 360 -> back to 0: a full revolution that closes on itself.
    loop = [0, 45, 90, 135, 179, -135, -90, -45, 0, 45, 90, 135, 179, -135, -90, -45, 0]
    forward = net_sweep(loop)
    backward = net_sweep([-a for a in loop])

    assert forward > 0, f"a positive loop must read positive, got {forward}"
    assert backward < 0, f"a negative loop must read negative, got {backward}"
    # A closed loop that travelled 720 deg must not read as 0.
    assert abs(forward) >= 180, f"closed loop collapsed to {forward}"
    assert abs(forward + backward) < 1e-6, "mirror loops must be equal and opposite"

    # The real kickflip trajectory that exposed the bug: 223 deg travelled, 27 deg
    # between endpoints. Endpoint differencing gives ~-27; the path does not.
    observed = [21, 21, 23, 27, 21, 20, 22, 27, 26, 27, 23, 22, 28, 23, 25, 26, 27, 23,
                21, 19, 18, 16, 32, 49, 39, 31, 17, -21, -41, -43, -40, -20, -19, -18,
                -17, -12, -13, -13, -8, -8, -6, -8, -8, -5, -6]
    measured = net_sweep(observed)
    assert abs(measured) >= 180, f"a 223 deg trajectory must not read as {measured}"
    assert abs(measured) <= 360, measured


@pytest.mark.xfail(
    strict=True,
    reason="active_window selects on total variation, which a slow drift beats; "
    "12.12 measured this collapsing kick-vs-heel separation from -108 to +2 deg. "
    "Flip to peak |dtheta/dt| and this will xpass, which strict=True will then flag.",
)
def test_active_window_finds_the_flip_not_the_drift():
    """Documents the *current* behaviour, which is wrong, so the bug stays visible.

    The premise is right -- a flip is a fast event in a mostly-static clip, so the
    sample budget should go where the board actually rotates. The implementation
    is not: `active_window` maximises total variation, and a slow drift across
    the clip accumulates more total variation than a real 0.3 s flip, so it
    selects the drift. Measured on real clips this collapsed kick-vs-heel
    separation from -108 deg to +2 deg (plan 12.12).

    Marked xfail(strict) rather than deleted: the suite stays green, but the
    moment the selection function is fixed this *xpasses* and pytest complains,
    so the bug cannot be quietly forgotten.
    """
    drift = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]          # slow, 2 deg/frame
    flip = [0, 5, 60, -20, 70, -60, 30, -10, 20, -5, 8, 2]        # fast, large swings
    series = drift[:5] + flip + drift[5:]
    flip_start, flip_end = 5, 5 + len(flip)

    window = active_window(series)
    assert window is not None
    start, end = window

    # These hold today: the window is produced, it starts at or after the flip
    # begins, and windowed_sweep tolerates missing frames instead of crashing on
    # them (the first implementation raised TypeError feeding None into net_sweep).
    assert flip_start <= start, f"window {start} starts before the flip at {flip_start}"
    assert end <= len(series)
    assert windowed_sweep(series)[0] != 0.0
    assert windowed_sweep([3, None, 3, None, 3, 3])[0] == 0.0, "static board, no rotation"

    # This is the known bug, and the reason for the xfail: the window overruns the
    # flip into the trailing drift, because the objective rewards total travel
    # rather than rate.
    assert end <= flip_end, (
        f"window ends at {end}, past the flip's end at {flip_end}: active_window "
        "selects on total variation and so prefers the slow drift"
    )


def _synthetic_board_mask(long_px: float, short_px: float) -> np.ndarray:
    """A flat rectangular mask with the requested axis lengths."""
    import cv2

    size = int(max(long_px, short_px)) + 20
    mask = np.zeros((size, size), np.uint8)
    centre = size // 2
    cv2.rectangle(
        mask,
        (centre - int(long_px / 2), centre - int(short_px / 2)),
        (centre + int(long_px / 2), centre + int(short_px / 2)),
        255,
        -1,
    )
    return mask


def test_foreshortening_tracks_a_roll_that_the_angle_cannot_see():
    """The feature the plan specified and M2 never built (plan section 7.1).

    A kickflip rotates about the board's **long** axis. In the image that
    *foreshortens* the board -- it gets narrower -- while the long-axis **angle
    barely changes**. So a model built on the angle alone is nearly blind to the
    most common trick in the dataset, which is what six rounds of M2 work measured
    the hard way (plan 12.8-12.15).

    The ratio must fall monotonically as the board rolls edge-on, and it must be
    self-normalising: a 4:1 board seen broadside is ~0.80, a fully rolled square
    is 0.50.
    """
    broadside = board_axes(_synthetic_board_mask(100, 25))
    assert abs(broadside["foreshortening"] - 100 / 125) < 0.02
    assert broadside["aspect"] > 3.5, "a real board is about 4:1"

    ratios = [board_axes(_synthetic_board_mask(100, short))["foreshortening"]
              for short in (25, 50, 75, 100)]
    # Monotonic decrease is the property that matters: a dip and recovery is
    # exactly the shape a kickflip should produce.
    assert ratios == sorted(ratios, reverse=True), ratios
    assert ratios[0] - ratios[-1] > 0.25, f"roll produced only {ratios}"

    # Self-normalising: scaling the board must not move the ratio, so the feature
    # does not depend on camera distance.
    small = board_axes(_synthetic_board_mask(40, 10))["foreshortening"]
    large = board_axes(_synthetic_board_mask(400, 100))["foreshortening"]
    assert abs(small - large) < 1e-6, (small, large)

    # A degenerate mask is unmeasured, not "flat".
    assert board_axes(np.zeros((40, 40), np.uint8)) is None


@pytest.mark.parametrize(
    "series",
    [
        pytest.param([float("nan"), 0.0, 45.0, float("nan")], id="nan-padded"),
        pytest.param([float("nan")] * 4, id="all-nan"),
        pytest.param([0.0, float("nan"), float("nan")], id="mostly-nan"),
    ],
)
def test_sweep_survives_unmeasurable_frames(series):
    """An unmeasured frame is NaN, and NaN must not crash the sweep.

    Found by running the v2 extraction: `board_features_from_angles` writes NaN
    for a frame it could not measure, and that NaN flowed into net_sweep's
    arithmetic, where it poisoned every sum and finally failed as
    "cannot convert float NaN to integer" -- thousands of frames from the cause,
    on most clips. A gap in the data is normal (a third of frames are
    unmeasurable), so it has to be a supported input, not an exception.
    """
    sweep = net_sweep(series)
    assert np.isfinite(sweep), sweep
    assert isinstance(sweep, float)


def test_board_feature_stream_handles_nan_angles():
    """The assembled board stream must survive a mostly-unmeasured clip."""
    filled = np.zeros((6, 5), np.float32)
    out = board_features_from_angles(filled, [None, None, None, 0.0, None, None], 6)
    assert out.shape == (6, 5 + 6 + 2)
    assert np.isfinite(out).all(), "NaN in the feature vector propagates to the LR"
    # Coverage is reported so a clip with no measurements is visibly empty.
    assert out[0, -1] == pytest.approx(1 / 6)


def test_board_summary_compresses_to_four_scalars_and_keeps_the_dip():
    """The board stream must shrink to a handful of scalars to be usable.

    The full 2640-dim stream measurably *lowered* holdout macro-F1 (0.2048 ->
    0.1476, plan 12.17) because 2448 pose dims already outnumber 110 training
    rows. So the compression is not cosmetic -- it is what makes the board stream
    usable at all.

    The quantity preserved is **dip depth**, which is what plan 12.16 actually
    demonstrated on real clips. `net_sweep` does not survive: it is an
    accumulated-path measure and gave identical means for kick-family and all
    other classes (-1.15 / -1.14).
    """
    frames = 48
    # A flat board that dips mid-clip, the shape a kickflip produces. Column 5+i
    # holds frame i's foreshortening.
    flat = np.full(frames, 0.83, dtype=np.float32)
    flat[22:26] = 0.59
    board = np.zeros((frames, 55), dtype=np.float32)
    for i in range(frames):
        board[i, 5 + i] = flat[i]
    summary = board_summary(board)

    assert summary.shape == (4,)
    assert summary[0] == pytest.approx(0.83 - 0.59, abs=0.01), "dip depth"
    assert 0.4 < summary[1] < 0.6, f"dip should be mid-clip, got {summary[1]}"
    assert summary[2] == pytest.approx(1.0), "all frames measured"
    assert summary[3] == pytest.approx(0.83, abs=0.01), "peak foreshortening"

    # A flat board with no dip has (near) zero depth -- so dip is a real
    # discriminator and not an artefact of every clip having some variance.
    quiet = np.zeros((frames, 55), dtype=np.float32)
    for i in range(frames):
        quiet[i, 5 + i] = 0.80
    assert board_summary(quiet)[0] == pytest.approx(0.0)

    # A block of NaN is "never measured": four explicit zeros, so an unmeasurable
    # clip is visibly empty rather than looking like a board that never moved.
    unmeasured = np.full((frames, 55), np.nan, dtype=np.float32)
    for i in range(frames):
        unmeasured[i, 5 + i] = np.nan
    assert board_summary(unmeasured).tolist() == [0, 0, 0, 0]

    # A block measured as literal zero is a *different* thing -- the board was seen
    # and did not roll -- so it reports full coverage with zero dip depth. Telling
    # those two apart is the entire point of writing NaN rather than 0.
    flat_zero = np.zeros((frames, 55), dtype=np.float32)
    for i in range(frames):
        flat_zero[i, 5 + i] = 0.0
    zero_summary = board_summary(flat_zero)
    assert zero_summary[0] == pytest.approx(0.0)
    assert zero_summary[2] == pytest.approx(1.0), "every frame was measured"

    # NaN gaps must not be mistaken for the dip.
    gapped = board.copy()
    for i in range(frames):
        if i == 23:
            gapped[i, 5 + i] = np.nan
    assert np.isfinite(board_summary(gapped)).all()
    assert board_summary(gapped)[0] == pytest.approx(0.83 - 0.59, abs=0.01)


def test_confusion_matrix_folds_low_support_classes_and_stays_readable():
    """M0's named deliverable is 'prints a confusion matrix'; it has to be readable.

    A 23-class matrix is ~576 characters wide, which no terminal shows in one
    piece, so the table keeps the highest-support classes and folds the rest into
    an (other) row and column, saying so. The fold must preserve the totals.
    """
    # 14 classes, descending support, one false positive each.
    labels = [f"c{i:02d}" for i in range(14)]
    cm = [[0] * 14 for _ in range(14)]
    for i, support in enumerate(range(14, 0, -1)):
        cm[i][i] = support
        cm[i][13] = 1
    total = sum(sum(row) for row in cm)

    folded = format_confusion_matrix(cm, labels, max_labels=5)
    assert "c00" in folded and "c13" not in folded, "keeps the highest-support class"
    assert "9 lower-support class(es) folded" in folded
    assert "(other x9)" in folded

    # The visible numbers must still add up to the whole matrix, so the (other)
    # row and column really do carry the folded classes. Drop the first two lines
    # (header, rule) and the last (the fold note). Row labels can contain spaces
    # ("(other x9)"), so read the last 6 tokens rather than skipping by count.
    body = folded.splitlines()[2:-1]
    assert len(body) == 6, "5 kept classes plus the (other) row"
    visible = [int(token) for line in body for token in line.split()[-6:]]
    assert sum(visible) == total, (sum(visible), total)

    # max_labels=None prints everything, with no fold note.
    full = format_confusion_matrix(cm, labels, max_labels=None)
    assert "c13" in full and "folded" not in full
    assert len(full.splitlines()) == len(labels) + 2

    # A matrix that already fits is printed untouched.
    small = format_confusion_matrix([[1, 0], [0, 1]], ["a", "b"], max_labels=5)
    assert "folded" not in small and len(small.splitlines()) == 4

    # And it is narrow enough to read.
    assert max(len(line) for line in folded.splitlines()) < 200


def test_guardrail_rejects_a_riding_direction_used_as_a_stance():
    """The dangerous error is a plausible-looking wrong value, not a missing one."""
    df = pd.read_csv("data/manifest.csv")
    taxonomy = Taxonomy.load()

    # An all-empty column round-trips as float64/NaN, so cast before assigning.
    doctored = df.copy()
    doctored["stance_input"] = doctored["stance_input"].fillna("").astype(object)
    doctored.loc[doctored.index[0], "stance_input"] = "fakie"
    violations = taxonomy.validate_frame(doctored)
    assert any("stance_input must be one of" in v for v in violations)
    assert any("riding directions" in v for v in violations)

    # 'regular' is legal in both vocabularies, so it must not trip the leak check.
    ok = df.copy()
    ok["stance_input"] = ok["stance_input"].fillna("").astype(object)
    ok.loc[ok.index[0], "stance_input"] = "regular"
    assert not any("stance" in v for v in taxonomy.validate_frame(ok))

    # A valid toggle is accepted.
    goofy = df.copy()
    goofy["stance_input"] = goofy["stance_input"].fillna("").astype(object)
    goofy.loc[goofy.index[0], "stance_input"] = "goofy"
    assert not any("stance" in v for v in taxonomy.validate_frame(goofy))


def test_manifest_keeps_published_riding_direction_apart_from_stance():
    from skateid.taxonomy import PUBLISHED_RIDING_VALUES

    df = pd.read_csv("data/manifest.csv")

    # SkateAI publishes riding directions; SkateboardML publishes nothing at all.
    skateai = df[df["dataset"] == "skateai"]
    assert set(skateai["stance_published"]) <= set(PUBLISHED_RIDING_VALUES)
    assert "fakie" in set(skateai["stance_published"])
    assert df[df["dataset"] == "skateboardml"]["stance_published"].isna().all()

    # stance_input is empty for every row until M1 can resolve the toggle. It reads
    # back as NaN because an all-empty CSV column has no string dtype.
    assert df["stance_input"].isna().all()


def test_group_disjoint_split_never_splits_a_group():
    df = pd.DataFrame({"group": ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4})
    split = generate_group_disjoint_split(df, group_col="group", test_size=0.25, seed=0)

    assert set(split.unique()) == {"train", "test"}
    test_groups = set(df.loc[split == "test", "group"])
    train_groups = set(df.loc[split == "train", "group"])
    assert test_groups and train_groups, "both sides of the split must be populated"
    assert not (test_groups & train_groups), "a group must never appear on both sides"

    # Deterministic for a fixed seed.
    again = generate_group_disjoint_split(df, group_col="group", test_size=0.25, seed=0)
    assert split.tolist() == again.tolist()


def test_skateai_labels_are_consistent():
    meta_path = Path("data/raw/skateai/metadata.csv")
    if not meta_path.exists():
        pytest.skip("SkateAI labels not fetched (run 'skateid fetch --dataset skateai')")
    meta = pd.read_csv(meta_path)

    assert len(meta) == 449, f"Expected 449 SkateAI clips, found {len(meta)}"
    # `video_file` alone repeats across battles; the title+file pair is the key.
    keys = list(zip(meta["video_title"], meta["video_file"]))
    assert len(set(keys)) == len(keys), "SkateAI clip keys must be unique"

    assert meta["trick_name"].notna().all()
    assert set(meta["stance"].unique()) <= {"regular", "switch", "fakie", "nollie"}
    # Those are riding directions, NOT goofy/regular stances: none of them can
    # fix the sign frame, which is why the manifest keeps them apart.
    assert "goofy" not in set(meta["stance"].unique())
    assert set(meta["flip_type"].unique()) <= {"none", "kickflip", "heelflip"}
    assert set(meta["landed"].astype(bool).unique()) <= {True, False}


def test_skateai_holdout_is_video_disjoint():
    manifest_path = Path("data/manifest.csv")
    assert manifest_path.exists(), "data/manifest.csv must exist"
    df = pd.read_csv(manifest_path)
    skateai = df[df["dataset"] == "skateai"]
    if skateai.empty:
        pytest.skip("SkateAI clips not downloaded yet (run 'skateid fetch --with-clips')")

    assert set(skateai["split_source"].unique()) == {"source_video_url"}
    assert set(skateai["skater_id_source"].unique()) == {"not_published_per_clip"}

    # No source video may contribute clips to both sides of the holdout split.
    train_videos = set(skateai[skateai["split_holdout"] == "train"]["source_video_url"])
    test_videos = set(skateai[skateai["split_holdout"] == "test"]["source_video_url"])
    assert train_videos and test_videos
    assert not (train_videos & test_videos), "a source video must not leak across the split"

