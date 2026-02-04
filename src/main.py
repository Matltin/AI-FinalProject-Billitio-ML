from __future__ import annotations

import argparse

from .config import Config
from .train import train_pipeline
from .predict import predict_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tripreason",
        description="Train/Predict TripReason (Work/Int) with classic ML. Only this file has main().",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train model and save artifacts.")
    p_train.add_argument("--train_path", type=str, required=True, help="Path to train_data.csv")
    p_train.add_argument("--out_dir", type=str, default="artifacts", help="Output directory for artifacts")
    p_train.add_argument("--test_size", type=float, default=0.2, help="Validation size (group split)")
    p_train.add_argument("--random_state", type=int, default=42, help="Random seed")

    p_pred = sub.add_parser("predict", help="Predict on test and create submission.csv")
    p_pred.add_argument("--test_path", type=str, required=True, help="Path to test_data.csv")
    p_pred.add_argument("--artifacts_dir", type=str, default="artifacts", help="Directory that contains artifacts")
    p_pred.add_argument("--output_path", type=str, default="artifacts/submission.csv", help="Where to write submission")

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "train":
        cfg = Config(random_state=args.random_state, test_size=args.test_size)
        meta = train_pipeline(train_path=args.train_path, out_dir=args.out_dir, cfg=cfg)
        print("✅ Training finished.")
        print(f"Artifacts saved to: {args.out_dir}")
        print(f"Best threshold: {meta['threshold']:.2f}")
        print(f"Val F1 (best threshold): {meta['metrics']['val_f1_at_best_threshold']:.4f}")
        return 0

    if args.command == "predict":
        out = predict_pipeline(
            test_path=args.test_path,
            artifacts_dir=args.artifacts_dir,
            output_path=args.output_path,
        )
        print("✅ Prediction finished.")
        print(f"Submission saved to: {out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())