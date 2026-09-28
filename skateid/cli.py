"""Command line interface for skateid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import pandas as pd

from skateid.data import build_manifest, fetch_skateboardml
from skateid.eval import MajorityClassBaseline, evaluate_predictions, format_confusion_matrix

def fetch_cmd(args: argparse.Namespace) -> int:
    raw_dir = fetch_skateboardml(args.dest)
    print(f"Building manifest at {args.manifest}...")
    df = build_manifest(raw_dir=raw_dir, out_csv=args.manifest)
    print(f"Manifest created successfully with {len(df)} clips.")
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

    train_df = df[df[split_col] == "train"]
    if len(train_df) == 0:
        print(f"Error: no samples in training split '{args.split}'.")
        return 1

    model = MajorityClassBaseline()
    model.fit(train_df["label"])
    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "b0_majority.json"
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

    test_df = df[df[split_col] == "test"]
    if len(test_df) == 0:
        print(f"Warning: split '{args.split}' has 0 test samples. Evaluating on full manifest.")
        test_df = df

    model_path = Path(args.checkpoint_dir) / "b0_majority.json"
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
    print(format_confusion_matrix(metrics["confusion_matrix"], metrics["labels"]))
    print("\nClassification Report:")
    print(metrics["report"])
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(prog="skateid", description="SkateID flatground skateboard trick recognition")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_p = subparsers.add_parser("fetch", help="Download SkateboardML clips and build manifest")
    fetch_p.add_argument("--dest", default="data/raw/skateboardml", help="Directory to save raw clips")
    fetch_p.add_argument("--manifest", default="data/manifest.csv", help="Output path for manifest CSV")

    train_p = subparsers.add_parser("train", help="Train trick recognition model")
    train_p.add_argument("--manifest", default="data/manifest.csv", help="Path to manifest CSV")
    train_p.add_argument("--split", choices=["published", "holdout"], default="published", help="Split to use")
    train_p.add_argument("--checkpoint-dir", default="checkpoints", help="Directory to save model checkpoints")

    eval_p = subparsers.add_parser("eval", help="Evaluate trick recognition model")
    eval_p.add_argument("--manifest", default="data/manifest.csv", help="Path to manifest CSV")
    eval_p.add_argument("--split", choices=["published", "holdout"], default="published", help="Split to use")
    eval_p.add_argument("--checkpoint-dir", default="checkpoints", help="Directory where model checkpoints are saved")

    args = parser.parse_args()
    if args.command == "fetch":
        return fetch_cmd(args)
    elif args.command == "train":
        return train_cmd(args)
    elif args.command == "eval":
        return eval_cmd(args)
    return 0

if __name__ == "__main__":
    sys.exit(main())

