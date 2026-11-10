"""Train the reproducible 64 -> 32 MLP baseline.

Run from the project root with ``python -m src.train``.
"""

import argparse
import json
import platform
import random
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import (
    DEFAULT_TRAIN_PATH,
    PROJECT_ROOT,
    fit_preprocessing,
    inverse_targets,
    load_training_data,
    transform_features,
    transform_targets,
)
from .evaluate import file_sha256, predict, validation_metrics
from .model import MLPRegressor


def train(args: argparse.Namespace) -> dict:
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1:
        raise ValueError("epochs, batch-size, and threads must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if not 0 < args.validation_size < 1:
        raise ValueError("validation-size must be between 0 and 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)

    features, targets, ids = load_training_data(args.train_csv)
    
    # データセットを訓練集合とテスト集合に分割する
    train_indices, validation_indices = train_test_split(
        np.arange(len(features)), test_size=args.validation_size, random_state=args.seed,
    )
    
    # 前処理
    preprocessing = fit_preprocessing(features.iloc[train_indices], targets.iloc[train_indices])
    train_features = transform_features(features.iloc[train_indices], preprocessing)
    validation_features = transform_features(features.iloc[validation_indices], preprocessing)
    train_targets = transform_targets(targets.iloc[train_indices], preprocessing)
    validation_targets_raw = targets.iloc[validation_indices].to_numpy(dtype=np.float64)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_targets)),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = MLPRegressor(train_features.shape[1])
    
    # 損失
    criterion = nn.MSELoss()
    
    # 最適化アルゴリズム
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 学習設定の記録
    config = {
        "train_csv": str(args.train_csv.resolve()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "validation_size": args.validation_size,
        "seed": args.seed,
        "threads": args.threads,
        "device": "cpu",
        "hidden_dims": [64, 32],
        "loss": "MSELoss on standardized DIC",
        "optimizer": "Adam",
        "early_stopping": False,
        "selection_metric": "minimum validation MSE in original DIC scale",
    }
    fingerprint = file_sha256(args.train_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_mse = float("inf")
    best_checkpoint = None
    print(
        f"Rows: train={len(train_indices)}, validation={len(validation_indices)}; "
        f"features={train_features.shape[1]}; device=cpu", flush=True,
    )

    # 学習
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch_features, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features), batch_targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch_features)

        train_predictions = inverse_targets(predict(model, train_features), preprocessing)
        validation_predictions = inverse_targets(predict(model, validation_features), preprocessing)
        train_mse = float(mean_squared_error(targets.iloc[train_indices], train_predictions))
        validation_mse = float(mean_squared_error(validation_targets_raw, validation_predictions))
        history.append({
            "epoch": epoch,
            "train_loss_standardized": loss_sum / len(train_indices),
            "train_mse": train_mse,
            "validation_mse": validation_mse,
        })
        
        # MSEが下回ったら、最良チェックポイントを更新
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_checkpoint = {
                "format_version": 1,
                "model_state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
                "input_dim": train_features.shape[1],
                "preprocessing": preprocessing,
                "split": {
                    "train_indices": train_indices.tolist(),
                    "validation_indices": validation_indices.tolist(),
                    "validation_ids": ids.iloc[validation_indices].tolist(),
                },
                "train_csv_sha256": fingerprint,
                "best_epoch": epoch,
                "validation_mse": validation_mse,
            }
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:3d}/{args.epochs}: train MSE={train_mse:.6f}, "
                f"validation MSE={validation_mse:.6f}", flush=True,
            )

    # 学習結果を保存
    torch.save({**best_checkpoint, "config": config}, args.output_dir / "model.pt")
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_predictions = inverse_targets(predict(model, validation_features), preprocessing)
    metrics = validation_metrics(validation_targets_raw, best_predictions)
    mean_prediction = np.full(len(validation_indices), targets.iloc[train_indices].mean())
    metrics.update({
        "best_epoch": best_checkpoint["best_epoch"],
        "last_epoch_validation_mse": history[-1]["validation_mse"],
        "mean_predictor_validation_mse": float(mean_squared_error(validation_targets_raw, mean_prediction)),
        "total_rows": len(features),
        "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "feature_count": train_features.shape[1],
        "feature_columns": preprocessing["feature_columns"],
        "train_csv_sha256": fingerprint,
        "completed_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "package_versions": {
            name: version(name) for name in ["pandas", "numpy", "scikit-learn", "torch", "matplotlib"]
        },
        "config": config,
    })
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    pd.DataFrame({
        "id": ids.iloc[validation_indices].to_numpy(),
        "DIC": validation_targets_raw,
        "predicted_DIC": best_predictions,
    }).to_csv(args.output_dir / "validation_predictions.csv", index=False)

    # 学習曲線の出力
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot([row["epoch"] for row in history], [row["train_mse"] for row in history], label="Train")
    axis.plot([row["epoch"] for row in history], [row["validation_mse"] for row in history], label="Validation")
    axis.axvline(metrics["best_epoch"], color="gray", linestyle="--", label="Best epoch")
    axis.set(xlabel="Epoch", ylabel="MSE (original DIC scale)", title="MLP baseline learning curve", yscale="log")
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "learning_curve.png", dpi=150)
    plt.close(figure)
    print(
        f"Best epoch={metrics['best_epoch']}; validation MSE={metrics['validation_mse']:.8f}; "
        f"RMSE={metrics['validation_rmse']:.8f}\nSaved to {args.output_dir.resolve()}", flush=True,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "baseline")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
