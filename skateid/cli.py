"""Command line interface for skateid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import pandas as pd

from skateid.baselines import (
    available_embedders,
    available_vlms,
    embedder_names,
    run_b1,
    run_b2,
    vlm_names,
)
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
    return 0

if __name__ == "__main__":
    sys.exit(main())

