"""Create a Kaggle submission with the saved model and training statistics.

Run from the project root with ``python -m src.predict``.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from .data import PROJECT_ROOT, inverse_targets, transform_features
from .evaluate import predict
from .model import MLPRegressor


def create_submission(
    checkpoint_path: Path,
    test_csv: Path,
    sample_submission_csv: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Predict DIC and preserve the sample submission's ID order and schema."""
    test = pd.read_csv(test_csv)
    sample = pd.read_csv(sample_submission_csv)
    if sample.columns.tolist() != ["id", "DIC"]:
        raise ValueError("Sample submission must have exactly the columns id,DIC.")
    for name, frame in (("Test data", test), ("Sample submission", sample)):
        if frame.empty or "id" not in frame:
            raise ValueError(f"{name} must contain rows and an id column.")
        if frame["id"].isna().any() or not frame["id"].is_unique:
            raise ValueError(f"{name} IDs must be nonmissing and unique.")
    order = pd.Index(test["id"]).get_indexer(sample["id"])
    if len(test) != len(sample) or (order < 0).any():
        raise ValueError("Test data and sample submission must contain the same IDs.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    torch.set_num_threads(checkpoint["config"]["threads"])
    preprocessing = checkpoint["preprocessing"]
    features = transform_features(test, preprocessing)
    model = MLPRegressor(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions = inverse_targets(predict(model, features), preprocessing)

    submission = sample.copy()
    submission["DIC"] = predictions[order]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "checkpoints" / "baseline" / "model.pt",
    )
    parser.add_argument("--test-csv", type=Path, default=PROJECT_ROOT / "dataset" / "test.csv")
    parser.add_argument(
        "--sample-submission", type=Path,
        default=PROJECT_ROOT / "dataset" / "sample_submission.csv",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "submissions" / "baseline.csv")
    args = parser.parse_args()
    submission = create_submission(args.checkpoint, args.test_csv, args.sample_submission, args.output)
    print(f"Saved {len(submission)} predictions (id,DIC) to {args.output.resolve()}")


if __name__ == "__main__":
    main()
