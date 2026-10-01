"""Command line interface for skateid."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from skateid.eval import evaluate_predictions, format_confusion_matrix
from skateid.stance import trick_for_both_stances
from skateid.taxonomy import Taxonomy
from skateid.video import sample_frames

#: Value in the `split_holdout` column meaning "held out". The column holds
#: train/test; the *published* column is a different partition of the same clips,
#: so reading the wrong one would score on clips the model trained on.
HOLDOUT_VALUE = "test"

from skateid.baselines import (
    available_embedders,
    available_vlms,
    embedder_names,
    run_b1,
    run_b2,
    vlm_names,
)
from skateid.taxonomy import Taxonomy
from skateid.data import (
    build_manifest,
    default_taxonomy,
    download_skateai_clips,
    fetch_skateai,
    fetch_skateboardml,
)
from skateid.eval import MajorityClassBaseline, evaluate_predictions, format_confusion_matrix

def fetch_cmd(args: argparse.Namespace) -> int:
    datasets = ("skateboardml", "skateai") if args.dataset == "all" else (args.dataset,)
    raw_dir: Path | str = args.dest
    skateai_dir: Path | str = args.skateai_dest

    if "skateboardml" in datasets:
        raw_dir = fetch_skateboardml(args.dest, force=args.force)

    if "skateai" in datasets:
        skateai_dir = fetch_skateai(args.skateai_dest, force=args.force)
        if args.with_clips:
            print("Cutting SkateAI clips from their BATB source videos (this is the slow part)...")
            new_clips = download_skateai_clips(
                skateai_dir,
                limit=args.limit,
                max_sources=args.max_sources,
                force=args.force,
            )
            print(f"Produced {len(new_clips)} new SkateAI clips.")

    print(f"Building manifest at {args.manifest}...")
    try:
        df = build_manifest(
            raw_dir=raw_dir,
            out_csv=args.manifest,
            skateai_dir=skateai_dir,
            datasets=datasets,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        if "skateai" in datasets and not args.with_clips:
            print("SkateAI publishes labels only, so its clips must be cut on demand;")
            print("re-run with --with-clips (slow; needs ffmpeg on PATH).")
        return 1

    per_dataset = df["dataset"].value_counts().to_dict() if "dataset" in df.columns else {}
    print(f"Manifest created successfully with {len(df)} clips {per_dataset}.")
    return 0

def train_cmd(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: manifest file '{manifest_path}' does not exist.")
        print("Run 'skateid fetch' first or create a manifest.")
        return 1

    df = pd.read_csv(manifest_path)
    split_col = f"split_{args.split}"
    if split_col not in df.columns:
        print(f"Error: split column '{split_col}' not found in manifest. Available: {[c for c in df.columns if c.startswith('split_')]}")
        return 1

    if args.dataset != "all":
        df = df[df["dataset"] == args.dataset]
        if df.empty:
            print(f"Error: manifest has no '{args.dataset}' clips.")
            return 1

    train_df = df[df[split_col] == "train"]
    if len(train_df) == 0:
        print(f"Error: no samples in training split '{args.split}'.")
        return 1

    model = MajorityClassBaseline()
    model.fit(train_df["label"])
    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"b0_majority_{args.dataset}_{args.split}.json"
    with open(out_file, "w") as f:
        json.dump({"majority_class": model.majority_class, "classes": model.classes_}, f, indent=2)

    print(f"Trained B0 Majority model. Dominant class: '{model.majority_class}'. Saved to {out_file}")
    return 0

def eval_cmd(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: manifest file '{manifest_path}' not found.")
        return 1

    df = pd.read_csv(manifest_path)
    split_col = f"split_{args.split}"
    if split_col not in df.columns:
        print(f"Error: split column '{split_col}' not found in manifest.")
        return 1

    if args.dataset != "all":
        df = df[df["dataset"] == args.dataset]
        if df.empty:
            print(f"Error: manifest has no '{args.dataset}' clips.")
            return 1

    test_df = df[df[split_col] == "test"]
    if len(test_df) == 0:
        print(f"Warning: split '{args.split}' has 0 test samples. Evaluating on full manifest.")
        test_df = df

    # Name the checkpoint after its dataset and split: a single shared file would
    # let eval silently score a checkpoint trained on a different split.
    model_path = Path(args.checkpoint_dir) / f"b0_majority_{args.dataset}_{args.split}.json"
    if not model_path.exists():
        print(f"Model file {model_path} not found. Running ad-hoc baseline fit on train split.")
        train_df = df[df[split_col] == "train"]
        if len(train_df) == 0:
            train_df = df
        model = MajorityClassBaseline().fit(train_df["label"])
    else:
        with open(model_path) as f:
            data = json.load(f)
        model = MajorityClassBaseline()
        model.majority_class = data["majority_class"]
        model.classes_ = data["classes"]

    preds = model.predict(len(test_df))
    metrics = evaluate_predictions(test_df["label"], preds)

    print(f"\n================ Evaluation Results (Split: {args.split}) ================")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro F1:  {metrics['macro_f1']:.4f}")
    print("\nConfusion Matrix:")
    # --max-labels 0 means "print every class", which format_confusion_matrix
    # spells as None.
    print(format_confusion_matrix(
        metrics["confusion_matrix"],
        metrics["labels"],
        max_labels=args.max_labels or None,
    ))
    print("\nClassification Report:")
    print(metrics["report"])
    return 0

def validate_cmd(args: argparse.Namespace) -> int:
    """Enforce the flatground scope guardrail against an existing manifest."""
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: manifest file '{manifest_path}' does not exist. Run 'skateid fetch' first.")
        return 1

    df = pd.read_csv(manifest_path)
    if args.dataset != "all":
        df = df[df["dataset"] == args.dataset]
    if df.empty:
        print(f"Error: manifest has no '{args.dataset}' clips.")
        return 1

    taxonomy = default_taxonomy()
    summary = taxonomy.describe()
    print("Flatground vocabulary:")
    print(f"  canonical names:      {summary['canonical_names']}")
    print(f"  rotation-expressible: {summary['rotation_expressible']}")
    print(f"  aliases:              {summary['aliases']}")
    print(f"  documented as rotation-free: {', '.join(summary['rotation_free'])}")
    print()

    violations = taxonomy.validate_frame(df)
    print(f"Checked {len(df)} rows from '{args.dataset}' against the dictionary and registry.")
    if violations:
        print(f"FAILED: {len(violations)} violation(s):")
        for violation in violations:
            print(f"  - {violation}")
        return 1
    print("PASSED: every label, component and rotation triple is in scope.")
    return 0

def baselines_cmd(args: argparse.Namespace) -> int:
    """Run B1 (frozen embedder probe) and/or B2 (VLM zero-shot)."""
    if not (args.b1 or args.b2):
        print("Available B1 embedders (name: usable - reason):")
        for name, (usable, reason) in sorted(available_embedders().items()):
            print(f"  {name:14} {'yes' if usable else 'no ':3} {reason}")
        print("\nAvailable B2 VLM backends (name: usable - reason):")
        for name, (usable, reason) in sorted(available_vlms().items()):
            print(f"  {name:14} {'yes' if usable else 'no ':3} {reason}")
        print("\nChoose one with --b1 and/or --b2, e.g. 'skateid baselines --b1 --embedder videomae'.")
        print("'motion_stats' is a weak floor and 'mock' is a plumbing test: neither is a result.")
        return 0

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: manifest file '{manifest_path}' does not exist. Run 'skateid fetch' first.")
        return 1
    df = pd.read_csv(manifest_path)
    if args.dataset != "all":
        df = df[df["dataset"] == args.dataset]
        if df.empty:
            print(f"Error: manifest has no '{args.dataset}' clips.")
            return 1

    split_col = f"split_{args.split}"
    if split_col not in df.columns:
        print(f"Error: split column '{split_col}' not found. Available: "
              f"{[c for c in df.columns if c.startswith('split_')]}")
        return 1

    results = []
    if args.b1:
        print(f"\n================ B1: frozen '{args.embedder}' + linear probe "
              f"({args.dataset}/{args.split}) ================")
        result = run_b1(df, embedder=args.embedder, split_col=split_col, limit=args.limit)
        results.append(result)
        if result["status"] != "ok":
            print(f"{result['status'].capitalize()}: {result['reason']}")
        else:
            print(f"Features: {result['dim']}-d | train {result['n_train']} | "
                  f"test {result['n_test']} | classes {result['n_classes']}")
            print(f"Accuracy:  {result['metrics']['accuracy']:.4f}")
            print(f"Macro F1:  {result['metrics']['macro_f1']:.4f}  "
                  f"(majority floor on this split: {result['majority_macro_f1']:.4f})")

    if args.b2:
        print(f"\n================ B2: '{args.vlm}' zero-shot "
              f"({args.dataset}/{args.split}) ================")
        result = run_b2(df, backend=args.vlm, split_col=split_col, limit=args.limit)
        results.append(result)
        if result["status"] != "ok":
            print(f"{result['status'].capitalize()}: {result['reason']}")
        else:
            print(f"Model: {result['model']} | clips {result['n_test']} | "
                  f"classes {result['n_classes']}")
            print(f"Resolved: {result['resolved_rate']:.2%} "
                  f"({result['abstentions']} abstentions scored as wrong)")
            print(f"Accuracy:  {result['metrics']['accuracy']:.4f}")
            print(f"Macro F1:  {result['metrics']['macro_f1']:.4f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Metrics dicts hold numpy-free primitives, but the confusion matrix list
        # is nested ints; drop the per-clip payload to keep the report readable.
        trimmed = [
            {key: value for key, value in result.items()
             if key not in ("predictions", "test_labels", "answers")}
            for result in results
        ]
        with open(out_path, "w") as handle:
            json.dump({"dataset": args.dataset, "split": args.split, "results": trimmed}, handle, indent=2)
        print(f"\nWrote {out_path}")
    return 0


def _segmented_angle(model, image, device: int):
    """Board angle from a YOLO *segmentation* mask, or ``None``.

    Thin wrapper over :func:`skateid.features.segmented_angle`, kept here so the
    oracle's call sites read the same way. Section 12.9 recorded that this arm
    segments the board far more reliably than Otsu; the Otsu arm is retained as a
    comparison, not as the default.
    """
    from .features import segmented_angle

    return segmented_angle(model, image, device)


def oracle_cmd(args) -> int:
    """M2's oracle check: does the measured board angle track the real rotation?

    Run *before* training anything on this feature. It answers one question with
    real footage rather than a model score: for clips whose manifest label already
    says how many flips happened, does the measured signed sweep agree?

    The bar is deliberately low and deliberately explicit. A sweep that merely
    *varies* across clips proves nothing -- noise varies too. What matters is
    whether the sign separates kick-family from heel-family, because that is the
    axis pose cannot see and the entire reason this stream exists.
    """
    from . import features

    frame = pd.read_csv(args.manifest, dtype={"clip_id": str})
    frame = frame[frame["dataset"] == "skateai"]
    if args.limit:
        frame = frame.head(args.limit)
    elif args.per_class:
        # A few clips per class, so the table is not 112 kickflips.
        frame = frame.groupby("label", group_keys=False).head(args.per_class)

    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    board_model = YOLO("yolo11n.pt").to(device)
    # Optional learned segmenter. Section 12.9 shows it detects the board far more
    # often than Otsu does, and that this does NOT rescue the angle -- but it is
    # exposed so that result is reproducible rather than a one-off observation.
    seg_model = YOLO(args.segmenter).to(device) if args.segmenter else None

    rows = []
    for _, record in frame.iterrows():
        key = features.cache_key(record["clip_id"], features.EXTRACTOR_VERSION)

        try:
            batch = sample_frames(record["file_path"], count=args.frames, size=(640, 640))
        except (OSError, ValueError):
            continue
        if args.two_pass and seg_model is not None:
            # Locate the flip, then re-decode just that window at native fps.
            sweep, span, measured_n, native_n = features.measure_clip_two_pass(
                record["file_path"], float(record["fps"] or 30.0), seg_model
            )
            start, end = f"{span[0]:.2f}-{span[1]:.2f}s"
            coverage = measured_n / max(native_n, 1)
        else:
            angles = []
            for image in batch:
                if seg_model is not None:
                    angles.append(_segmented_angle(seg_model, image, device))
                else:
                    angles.append(features._measure_angle(
                        image, features.detect_board(board_model, image, device=device)))
            usable = [a for a in angles if a is not None]
            if args.window:
                sweep, w_start, w_end = features.windowed_sweep(angles)
                start, end = str(w_start), str(w_end)
            else:
                sweep = features.net_sweep(usable) if len(usable) >= 2 else 0.0
                start, end = 0, len(angles) - 1
            coverage = len(usable) / max(args.frames, 1)
        rows.append({
            "clip_id": record["clip_id"],
            "label": record["label"],
            "flip": int(record["flip_number"]),
            "flip_type": str(record["flip_type"]),
            "board_spin": int(record["board_rotation_number"]),
            "expected_sign": -1 if "heel" in str(record["flip_type"]) else (1 if "kick" in str(record["flip_type"]) else 0),
            "measured_sweep": sweep,
            "window": f"{start}-{end}",
            "angle_coverage": coverage,
        })

    report = pd.DataFrame(rows)
    report.to_csv(args.out, index=False)
    if report.empty:
        print("no clips could be measured", file=sys.stderr)
        return 2

    report["sign_ok"] = (
        (report["expected_sign"] == 0)
        | (np.sign(report["measured_sweep"]) == report["expected_sign"])
    )
    kick = report[report["expected_sign"] == 1]
    heel = report[report["expected_sign"] == -1]

    print(f"measured {len(report)} clips, {args.frames} frames each\n")
    print(f"{'label':<22}{'exp':>4}{'sweep':>9}{'cover':>7}  sign")
    for _, r in report.head(args.show).iterrows():
        mark = "ok" if r["sign_ok"] else ("--" if r["expected_sign"] == 0 else "MISS")
        print(f"{r['label']:<22}{r['expected_sign']:>4}{r['measured_sweep']:>9.0f}"
              f"{r['angle_coverage']:>7.2f}  {mark}")

    print(f"\nangle measured on {report['angle_coverage'].mean():.1%} of frames on average")
    print(f"kick-family clips: {len(kick)}, mean sweep {kick['measured_sweep'].mean():+.0f} deg, "
          f"correct sign {kick['sign_ok'].mean():.0%}")
    print(f"heel-family clips: {len(heel)}, mean sweep {heel['measured_sweep'].mean():+.0f} deg, "
          f"correct sign {heel['sign_ok'].mean():.0%}")
    if len(kick) and len(heel):
        print(f"\nkick - heel separation: "
              f"{kick['measured_sweep'].mean() - heel['measured_sweep'].mean():+.0f} deg "
              f"({report['sign_ok'].mean():.0%} of all clips get the right sign)")
    print(f"\nwrote {args.out}")
    return 0


def probe_cmd(args) -> int:
    """Score cached pose features against the holdout split. This is M1's gate.

    Pose features flattened in time order, logistic regression, scored on the
    video-disjoint holdout. The comparison that matters is against the best
    *measured* holdout score in this project -- VideoMAE at 0.0401 -- not against
    an absolute number, because the question M1 asks is whether structured
    features extract more than a frozen generic probe.

    The board stream is opt-in via --with-board. It contributes location and size,
    not rotation (see features.board_features), so it is measured separately
    rather than folded into a "pose+board" headline.
    """
    from . import features

    frame = pd.read_csv(args.manifest, dtype={"clip_id": str})
    if args.dataset != "all":
        frame = frame[frame["dataset"] == args.dataset]

    taxonomy = default_taxonomy()
    # The split columns hold train/test; "holdout" in this project means the
    # project's own video-disjoint holdout, which is the `test` value of
    # split_holdout. Naming it explicitly avoids reading the *published* split,
    # which is a different partition of the same clips.
    holdout = frame[frame["split_holdout"] == HOLDOUT_VALUE]
    if holdout.empty:
        print(f"error: no rows with split_holdout == {HOLDOUT_VALUE!r} in the manifest", file=sys.stderr)
        return 2

    # Restricted vocabulary (plan 12.13). The 22-class problem is data-limited at
    # ~12 clips/class, not representation-limited, so scoring it measures the
    # dataset rather than the features. The restriction is reported, never applied
    # silently, and holdout rows whose class is excluded are dropped with a count.
    min_clips = args.min_train_clips
    if min_clips:
        train_counts = frame[frame["split_holdout"] != HOLDOUT_VALUE]["label"].value_counts()
        keep = set(train_counts[train_counts >= min_clips].index)
        dropped_classes = len(set(frame["label"]) - keep)
        frame = frame[frame["label"].isin(keep)]
        holdout = holdout[holdout["label"].isin(keep)]
        print(f"restricted vocabulary: {len(keep)} classes with >= {min_clips} training clips "
              f"({dropped_classes} classes excluded) — these stay reachable via the "
              f"rotation heads, not lost")
        if len(keep) < 2:
            print("error: too few classes survive the restriction", file=sys.stderr)
            return 2
        print(f"  classes: {sorted(keep)}")

    cache = Path(args.cache_dir)
    use_board = args.with_board or args.full_board
    X_train, y_train, ids_train = features.load_feature_table(
        frame[frame["split_holdout"] != HOLDOUT_VALUE], cache,
        include_board=use_board, board_summary_dims=not args.full_board,
    )
    X_test, y_test, ids_test = features.load_feature_table(
        holdout, cache, include_board=use_board, board_summary_dims=not args.full_board
    )
    if X_train.size == 0 or X_test.size == 0:
        print("error: no cached features. Run `skateid extract` first.", file=sys.stderr)
        return 2

    print(f"pose features: train {X_train.shape}, holdout {X_test.shape}")
    print(f"classes: train {len(set(y_train))}, holdout {len(set(y_test))}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=3000, C=args.c, class_weight="balanced")
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    metrics = evaluate_predictions(y_test, predictions, labels=sorted(set(y_train) | set(y_test)))
    print()
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro F1:  {metrics['macro_f1']:.4f}")
    baseline = 0.0401
    ratio = metrics["macro_f1"] / baseline if baseline else float("inf")
    print(f"vs VideoMAE floor {baseline}: {ratio:.2f}x  "
          f"(M1 gate is >= 5.0x, i.e. >= {5 * baseline:.4f})")
    print(f"GATE: {'MET' if ratio >= 5 else 'NOT MET'} -- "
          f"{'features beat the frozen probe' if ratio >= 5 else 'stop and fix features before scaling'}")
    print()
    print(format_confusion_matrix(metrics["confusion_matrix"], metrics["labels"], max_labels=12))
    print()
    print(classification_report(y_test, predictions, zero_division=0))

    # A model with no stance input must not be able to name a trick uniquely. If
    # it could, the sign convention is leaking in from somewhere else.
    print("stance-blind naming (what the UI would show without a toggle):")
    shown = 0
    for clip_id, truth, guess in list(zip(ids_test, y_test, predictions))[:args.examples]:
        pair = trick_for_both_stances(guess, taxonomy)
        goofy = pair["goofy"] or "?"
        print(f"  {clip_id[:52]:<52} true={truth:<18} -> {guess}  |  goofy: {goofy}")
        shown += 1
    return 0


def fit_model_cmd(args) -> int:
    """Fit the recogniser on pose features and write a JSON artefact."""
    from .recognize import fit

    taxonomy = default_taxonomy()
    frame = pd.read_csv(args.manifest, dtype={"clip_id": str})
    recognizer, metrics = fit(
        frame, Path(args.cache_dir), taxonomy,
        include_board=args.with_board, min_train_clips=args.min_train_clips,
        margin=args.margin, floor=args.floor,
    )
    out = recognizer.save(Path(args.checkpoint))
    excluded = metrics["excluded_classes"]
    print(f"fitted {metrics['classes']} classes on {metrics['train_clips']} clips "
          f"({metrics['dims']} dims) -> {out}")
    print(f"  restricted vocabulary: {len(excluded)} classes below {args.min_train_clips} "
          f"training clips are excluded; they stay reachable via the rotation heads, "
          f"not lost")
    if metrics["holdout_clips"]:
        print(f"  holdout: {metrics['holdout_clips']} clips | "
              f"macro-F1 {metrics['macro_f1_all_named']:.4f} with abstention | "
              f"abstains {metrics['abstain_rate']:.0%} | "
              f"macro-F1 {metrics['macro_f1_when_named']:.4f} / "
              f"accuracy {metrics['accuracy_when_named']:.4f} when it speaks")
    return 0


def recognize_cmd(args) -> int:
    """Recognise clips from cached features, printing what the model will say."""
    from .recognize import Recognizer

    taxonomy = default_taxonomy()
    frame = pd.read_csv(args.manifest, dtype={"clip_id": str})
    recognizer = Recognizer.load(Path(args.checkpoint), taxonomy,
                                 margin=args.margin, floor=args.floor)

    if args.dataset != "all":
        frame = frame[frame["dataset"] == args.dataset]
    frame = frame[frame["split_holdout"] != "train"] if args.holdout else frame
    if args.label:
        wanted = {item.strip() for item in args.label.split(",")}
        frame = frame[frame["label"].isin(wanted)]
    if args.limit:
        frame = frame.head(args.limit)

    from .features import EXTRACTOR_VERSION, load_features, pose_only_features, cache_key

    cache = Path(args.cache_dir)
    shown, named, correct = 0, 0, 0
    for _, record in frame.iterrows():
        vector = pose_only_features(cache, cache_key(record["clip_id"], EXTRACTOR_VERSION))
        if vector is None:
            continue
        prediction = recognizer.predict(vector, clip_id=record["clip_id"], stance=args.stance)
        shown += 1
        if not prediction.abstained:
            named += 1
            if args.stance != "auto" and prediction.label == record["label"]:
                correct += 1
        if shown <= args.show or args.json:
            if args.json:
                print(json.dumps(prediction.as_dict()))
                continue
            readings = "  ".join(
                f"{stance}: {name}" for stance, name in prediction.readings.items() if name
            )
            print(f"{record['clip_id'][:46]:<46} {prediction.display:<20} "
                  f"{prediction.confidence:.2f}  {prediction.reason}")
            if readings:
                print(f"{'':<46} {readings}")
    if shown and args.stance != "auto":
        print(f"\n{named}/{shown} named ({1 - named / shown:.0%} abstained); "
              f"{correct}/{named} correct when named" if named else
              f"\n0/{shown} named — the model abstained on everything")
    return 0


def extract_cmd(args) -> int:
    """Run pose + board extraction over the manifest and cache the result."""
    from . import features

    frame = pd.read_csv(args.manifest, dtype={"clip_id": str})
    if args.dataset != "all":
        frame = frame[frame["dataset"] == args.dataset]
    if args.limit:
        frame = frame.head(args.limit)

    import torch

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"extracting {len(frame)} clips on {'cuda' if device == 0 else 'cpu'}")

    pose_model = features.load_pose_model(args.pose_model, device=device)
    board_model = None
    if not args.no_board:
        from ultralytics import YOLO

        board_model = YOLO(args.board_model).to(device)
    if pose_model is None and board_model is None:
        print("error: no models available (is ultralytics installed?)", file=sys.stderr)
        return 2

    cache = Path(args.cache_dir)
    rows, done, failed = [], 0, []
    started = time.time()
    for _, record in frame.iterrows():
        key = features.cache_key(record["clip_id"], features.EXTRACTOR_VERSION)
        cached = features.load_features(cache, key)
        if cached is not None:
            body, board = cached
        else:
            try:
                body, board = features.extract_clip(
                    record["clip_id"], Path(record["file_path"]), pose_model, board_model,
                    frames=args.frames, device=device,
                )
                features.save_features(cache, key, body, board)
            except (OSError, ValueError) as error:
                failed.append((record["clip_id"], str(error)))
                continue
        done += 1
        quality = features.body_quality(body)
        rows.append({"clip_id": record["clip_id"], "label": record["label"],
                     "split_holdout": record.get("split_holdout", ""), **quality})

    report = pd.DataFrame(rows)
    report.to_csv(args.out, index=False)
    elapsed = time.time() - started

    print(f"extracted {done}/{len(frame)} in {elapsed:.0f}s ({elapsed / max(done, 1):.2f}s/clip)")
    if failed:
        print(f"WARNING: {len(failed)} clips failed and were skipped, e.g. {failed[:3]}")
    if not report.empty:
        print(f"keypoint fill: mean {report['keypoint_fill'].mean():.3f}, "
              f"min {report['keypoint_fill'].min():.3f}")
        print(f"usable clips (>=50% frames with >=50% keypoints): "
              f"{int(report['usable'].sum())}/{len(report)}")
    print(f"wrote quality report to {args.out}")
    return 0


def stance_cmd(args) -> int:
    """Set or report the goofy/regular toggle.

    Reporting is the default: `stance_input` is empty on every row today, and
    that is the honest state rather than a failure. Writing happens only through
    an explicit `--set`, so this command can never fill the column by accident.
    """
    from . import stance as stance_module

    frame = pd.read_csv(args.manifest, dtype={"clip_id": str})

    overrides = {}
    for item in args.set:
        if "=" not in item:
            print(f"error: --set expects CLIP_ID=STANCE, got {item!r}", file=sys.stderr)
            return 2
        clip_id, value = item.split("=", 1)
        overrides[clip_id.strip()] = value.strip()

    unknown = sorted(set(overrides) - set(frame["clip_id"].astype(str)))
    if unknown:
        print(f"error: no such clip_id in the manifest: {unknown}", file=sys.stderr)
        return 2

    try:
        updated = stance_module.apply_to_manifest(frame, overrides=overrides) if overrides else frame
    except ValueError as error:
        # A riding direction in --set is the mistake this command most invites, so
        # it gets the explanation rather than a traceback.
        print(f"error: {error}", file=sys.stderr)
        return 2
    if overrides:
        updated.to_csv(args.manifest, index=False)
        print(f"wrote {len(overrides)} stance value(s) to {args.manifest}")

    summary = stance_module.stance_summary(updated)
    print(f"stance: {summary['confirmed']}/{summary['rows']} clips confirmed, "
          f"{summary['empty']} empty ({summary['goofy']} goofy, {summary['regular']} regular)")

    taxonomy = Taxonomy.load(args.tricks, args.allowlist)
    print()
    print("stance_published is riding direction (fakie/switch/nollie), NOT goofy-vs-regular.")
    print("A 'fakie kickflip' and a 'regular kickflip' are the same trick, so the published")
    print("column cannot fix the sign frame and is never used as a fallback. It carries no")
    print("information about which foot leads -- every direction contains both kickflips")
    print("and heelflips.")
    print()

    # Both-stances output: the honest default answer. Rather than guessing, name
    # the trick under both readings; the stance only selects between them.
    if args.label:
        labels = [item.strip() for item in args.label.split(",") if item.strip()]
    else:
        labels = frame["label"].value_counts().head(args.top).index.astype(str).tolist()

    print(f"top {len(labels)} tricks, named under both stances:")
    width = max(len(str(label)) for label in labels) + 2
    for label in labels:
        result = stance_module.trick_for_both_stances(label, taxonomy)
        goofy = result["goofy"] or "(unnamed mirror)"
        marker = "" if result["stance_dependent"] else "  <- same name either way"
        print(f"  {str(label):<{width}}{result['regular']:<18}regular{'':<4}| goofy:{goofy}{marker}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="skateid", description="SkateID flatground skateboard trick recognition")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_p = subparsers.add_parser("fetch", help="Download dataset clips and build the manifest")
    fetch_p.add_argument(
        "--dataset",
        choices=["all", "skateboardml", "skateai"],
        default="all",
        help="Which dataset to fetch (default: all)",
    )
    fetch_p.add_argument("--dest", default="data/raw/skateboardml", help="Directory for SkateboardML raw clips")
    fetch_p.add_argument("--skateai-dest", default="data/raw/skateai", help="Directory for SkateAI labels and clips")
    fetch_p.add_argument(
        "--with-clips",
        action="store_true",
        help="Also download the BATB source videos and cut SkateAI's 449 clips (slow; needs ffmpeg)",
    )
    fetch_p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap SkateAI clips produced per source video (smoke test)",
    )
    fetch_p.add_argument(
        "--max-sources",
        type=int,
        default=None,
        help="Cap how many BATB source videos are processed (smoke test)",
    )
    fetch_p.add_argument("--force", action="store_true", help="Re-download files that already exist")
    fetch_p.add_argument("--manifest", default="data/manifest.csv", help="Output path for manifest CSV")

    train_p = subparsers.add_parser("train", help="Train trick recognition model")
    train_p.add_argument("--manifest", default="data/manifest.csv", help="Path to manifest CSV")
    train_p.add_argument("--split", choices=["published", "holdout"], default="published", help="Split to use")
    train_p.add_argument(
        "--dataset",
        choices=["all", "skateboardml", "skateai"],
        default="all",
        help="Restrict to one dataset",
    )
    train_p.add_argument("--checkpoint-dir", default="checkpoints", help="Directory to save model checkpoints")

    eval_p = subparsers.add_parser("eval", help="Evaluate trick recognition model")
    eval_p.add_argument("--manifest", default="data/manifest.csv", help="Path to manifest CSV")
    eval_p.add_argument("--split", choices=["published", "holdout"], default="published", help="Split to use")
    eval_p.add_argument(
        "--dataset",
        choices=["all", "skateboardml", "skateai"],
        default="all",
        help="Restrict to one dataset",
    )
    eval_p.add_argument("--checkpoint-dir", default="checkpoints", help="Directory where model checkpoints are saved")
    eval_p.add_argument(
        "--max-labels",
        type=int,
        default=12,
        help="Show at most N classes in the confusion matrix, folding the rest into "
        "(other). 0 prints every class (very wide past ~10 classes).",
    )

    oracle_p = subparsers.add_parser(
        "oracle", help="Check the measured board angle against known flip labels (M2 step 1)"
    )
    oracle_p.add_argument("--manifest", default="data/manifest.csv", help="Path to manifest CSV")
    oracle_p.add_argument("--frames", type=int, default=12, help="Frames sampled per clip")
    oracle_p.add_argument("--per-class", type=int, default=3, help="Clips per label (0 = all)")
    oracle_p.add_argument("--limit", type=int, default=0, help="Cap total clips")
    oracle_p.add_argument("--show", type=int, default=20, help="Rows to print")
    oracle_p.add_argument("--out", default="data/oracle_board.csv", help="Where to write the report")
    oracle_p.add_argument(
        "--segmenter", default="", help="Use a YOLO *segmentation* model for the board mask "
        "(e.g. yolo11n-seg.pt) instead of Otsu thresholding",
    )
    oracle_p.add_argument(
        "--window", action="store_true",
        help="Measure only the rotation window instead of the whole clip (known "
        "regression, kept for reproducing 12.12)",
    )
    oracle_p.add_argument(
        "--two-pass", action="store_true",
        help="Locate the rotation window, then re-decode it at native fps (the only "
        "variant that adds information rather than selecting from existing samples)",
    )

    probe_p = subparsers.add_parser(
        "probe", help="Score cached pose features against the holdout split (M1's gate)"
    )
    probe_p.add_argument("--manifest", default="data/manifest.csv", help="Path to manifest CSV")
    probe_p.add_argument("--dataset", choices=["all", "skateboardml", "skateai"], default="skateai",
                         help="Restrict to one dataset (default: skateai, the only source with enough classes)")
    probe_p.add_argument("--cache-dir", default="data/cache", help="Feature cache directory")
    probe_p.add_argument(
        "--with-board", action="store_true",
        help="Add the board stream (4 summary scalars by default: dip depth, dip "
        "timing, coverage, peak foreshortening)",
    )
    probe_p.add_argument(
        "--full-board", action="store_true",
        help="Use the full per-frame board stream instead of the 4 summary scalars "
        "(the full stream measurably hurt holdout score; see plan 12.17)",
    )
    probe_p.add_argument("--c", type=float, default=1.0, help="Logistic regression C")
    probe_p.add_argument("--examples", type=int, default=6, help="How many per-clip examples to print")
    probe_p.add_argument(
        "--min-train-clips", type=int, default=0,
        help="Restrict to classes with >= N training clips (plan 12.13). The 22-class "
        "problem is data-limited, not representation-limited. 0 disables it.",
    )
    probe_p.add_argument("--out", default="", help="Optional path to write the metrics as JSON")

    model_p = subparsers.add_parser(
        "fit", help="Fit the recogniser on pose features and save a JSON checkpoint"
    )
    model_p.add_argument("--manifest", default="data/manifest.csv", help="Path to manifest CSV")
    model_p.add_argument("--cache-dir", default="data/cache", help="Feature cache directory")
    model_p.add_argument("--checkpoint", default="checkpoints/recognizer.json", help="Where to save")
    model_p.add_argument("--min-train-clips", type=int, default=15,
                         help="Restricted vocabulary: drop classes below this many training "
                              "clips (plan 12.13). They stay reachable via the rotation heads.")
    model_p.add_argument("--with-board", action="store_true", help="Also use the board summary scalars")
    model_p.add_argument("--margin", type=float, default=0.15, help="Abstain below this top-2 gap")
    model_p.add_argument("--floor", type=float, default=0.35, help="Abstain below this top-1 probability")

    rec_p = subparsers.add_parser("recognize", help="Name clips from cached features, or say 'not sure'")
    rec_p.add_argument("--manifest", default="data/manifest.csv", help="Path to manifest CSV")
    rec_p.add_argument("--cache-dir", default="data/cache", help="Feature cache directory")
    rec_p.add_argument("--checkpoint", default="checkpoints/recognizer.json", help="Model JSON")
    rec_p.add_argument("--dataset", choices=["all", "skateboardml", "skateai"], default="skateai")
    rec_p.add_argument("--stance", default="auto", help="auto (show both readings) | regular | goofy")
    rec_p.add_argument("--limit", type=int, default=None, help="Cap clips")
    rec_p.add_argument("--show", type=int, default=8, help="Rows to print")
    rec_p.add_argument("--holdout", action="store_true", help="Only the holdout split")
    rec_p.add_argument("--label", default="", help="Comma-separated labels to include")
    rec_p.add_argument("--json", action="store_true", help="Machine-readable output")
    rec_p.add_argument("--margin", type=float, default=None, help="Override the saved margin")
    rec_p.add_argument("--floor", type=float, default=None, help="Override the saved floor")

    extract_p = subparsers.add_parser(
        "extract", help="Extract pose + board features for every clip and cache them"
    )
    extract_p.add_argument("--manifest", default="data/manifest.csv", help="Path to manifest CSV")
    extract_p.add_argument(
        "--dataset", choices=["all", "skateboardml", "skateai"], default="all",
        help="Restrict to one dataset",
    )
    extract_p.add_argument("--cache-dir", default="data/cache", help="Where to write .npz features")
    extract_p.add_argument("--out", default="data/feature_quality.csv", help="Quality report CSV")
    extract_p.add_argument("--frames", type=int, default=48, help="Frames sampled per clip")
    extract_p.add_argument("--pose-model", default="yolo11n-pose.pt", help="Pose model weights")
    extract_p.add_argument("--board-model", default="yolo11n.pt", help="Board detection model weights")
    extract_p.add_argument("--no-board", action="store_true", help="Skip the board stream (pose only)")
    extract_p.add_argument("--limit", type=int, default=None, help="Cap clips (smoke test)")

    validate_p = subparsers.add_parser(
        "validate", help="Check a manifest against the flatground scope guardrail"
    )
    validate_p.add_argument("--manifest", default="data/manifest.csv", help="Path to manifest CSV")
    validate_p.add_argument(
        "--dataset",
        choices=["all", "skateboardml", "skateai"],
        default="all",
        help="Restrict to one dataset",
    )

    baselines_p = subparsers.add_parser(
        "baselines", help="Run B1 (frozen embedder probe) and/or B2 (VLM zero-shot)"
    )
    baselines_p.add_argument("--manifest", default="data/manifest.csv", help="Path to manifest CSV")
    baselines_p.add_argument(
        "--split", choices=["published", "holdout"], default="holdout", help="Split to score on"
    )
    baselines_p.add_argument(
        "--dataset",
        choices=["all", "skateboardml", "skateai"],
        default="skateai",
        help="Restrict to one dataset (default: skateai, the only source with enough classes)",
    )
    baselines_p.add_argument("--b1", action="store_true", help="Run the frozen-embedder linear probe")
    baselines_p.add_argument("--b2", action="store_true", help="Run the VLM zero-shot baseline")
    baselines_p.add_argument(
        "--embedder", choices=embedder_names(), default="motion_stats", help="B1 feature extractor"
    )
    baselines_p.add_argument("--vlm", choices=vlm_names(), default="mock", help="B2 backend")
    baselines_p.add_argument("--limit", type=int, default=None, help="Cap clips (smoke test)")
    baselines_p.add_argument("--out", default="", help="Optional path to write the metrics as JSON")

    stance_p = subparsers.add_parser(
        "stance", help="Resolve the goofy/regular toggle (suggest-only; overrides win)"
    )
    stance_p.add_argument("--manifest", default="data/manifest.csv", help="Path to manifest CSV")
    stance_p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="CLIP_ID=STANCE",
        help="Confirm a stance for one clip, e.g. --set clip_0001=goofy. Repeatable. "
        "This is the only path that writes a value.",
    )
    stance_p.add_argument(
        "--label",
        default="",
        help="Comma-separated trick names to show under both stances (default: the most "
        "common labels in the manifest)",
    )
    stance_p.add_argument("--top", type=int, default=8, help="How many labels to show by default")
    stance_p.add_argument("--tricks", default="data/tricks.json", help="Path to the rotation dictionary")
    stance_p.add_argument(
        "--allowlist", default="data/flatground_allowlist.csv", help="Path to the name registry"
    )

    args = parser.parse_args()
    if args.command == "fetch":
        return fetch_cmd(args)
    elif args.command == "train":
        return train_cmd(args)
    elif args.command == "eval":
        return eval_cmd(args)
    elif args.command == "validate":
        return validate_cmd(args)
    elif args.command == "baselines":
        return baselines_cmd(args)
    elif args.command == "stance":
        return stance_cmd(args)
    elif args.command == "extract":
        return extract_cmd(args)
    elif args.command == "probe":
        return probe_cmd(args)
    elif args.command == "oracle":
        return oracle_cmd(args)
    elif args.command == "fit":
        return fit_model_cmd(args)
    elif args.command == "recognize":
        return recognize_cmd(args)
    return 0

if __name__ == "__main__":
    sys.exit(main())

