"""Evaluate the saved baseline on its original validation split.

Run from the project root with ``python -m src.evaluate``.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_squared_error

from .data import (
    DEFAULT_TRAIN_PATH,
    PROJECT_ROOT,
    inverse_targets,
    load_training_data,
    transform_features,
)
from .model import MLPRegressor


def file_sha256(path: Path) -> str:
    """Identify the exact CSV used for the saved split."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@torch.no_grad()
def predict(model: MLPRegressor, features: np.ndarray) -> np.ndarray:
    """Return standardized predictions without constructing gradients."""
    model.eval()
    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    return model(tensor).cpu().numpy().reshape(-1)


def validation_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    """Compute MSE and supplementary RMSE in the original DIC scale."""
    mse = float(mean_squared_error(targets, predictions))
    return {"validation_mse": mse, "validation_rmse": float(np.sqrt(mse))}


def evaluate_checkpoint(checkpoint_path: Path, train_csv: Path) -> dict:
    """Reload the selected weights and reproduce their validation metric."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["train_csv_sha256"] != file_sha256(train_csv):
        raise ValueError("Training CSV differs from the data used for this checkpoint.")
    torch.set_num_threads(checkpoint["config"]["threads"])
    features, targets, ids = load_training_data(train_csv)
    indices = checkpoint["split"]["validation_indices"]
    preprocessing = checkpoint["preprocessing"]
    validation_features = transform_features(features.iloc[indices], preprocessing)
    model = MLPRegressor(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions = inverse_targets(predict(model, validation_features), preprocessing)
    result = validation_metrics(targets.iloc[indices].to_numpy(), predictions)
    result.update(
        best_epoch=checkpoint["best_epoch"],
        validation_rows=len(indices),
        checkpoint=str(checkpoint_path.resolve()),
    )
    if not np.isclose(result["validation_mse"], checkpoint["validation_mse"], rtol=1e-6):
        raise ValueError("Reloaded checkpoint does not reproduce its saved validation MSE.")
    # Assert row alignment so a reordered CSV cannot silently change the holdout.
    if ids.iloc[indices].tolist() != checkpoint["split"]["validation_ids"]:
        raise ValueError("Validation IDs differ from the saved split.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "checkpoints" / "baseline" / "model.pt",
    )
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_PATH)
    args = parser.parse_args()
    print(json.dumps(evaluate_checkpoint(args.checkpoint, args.train_csv), indent=2))


if __name__ == "__main__":
    main()
