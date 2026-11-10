"""CSV loading and consistent feature names for the CalCOFI baseline."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "dataset" / "train.csv"
TARGET_COLUMN = "DIC"


def feature_frame(
    frame: pd.DataFrame, feature_columns: list[str] | None = None
) -> pd.DataFrame:
    """Return numeric predictors, aligning train/test names and saved order.

    Missing feature values are retained for training-fitted imputation. IDs,
    the target, and CSV index/empty columns are never used as predictors.
    """
    if frame.columns.duplicated().any():
        raise ValueError("CSV contains duplicate column names.")
    if "TA1.x" in frame and "TA1" in frame:
        raise ValueError("Both TA1.x and TA1 are present; the TA1 feature is ambiguous.")

    features = frame.rename(columns={"TA1.x": "TA1"}).drop(
        columns=[
            column
            for column in frame.columns
            if column in {"id", TARGET_COLUMN} or str(column).startswith("Unnamed:")
        ]
    ).copy()
    if feature_columns is not None:
        if not feature_columns or len(set(feature_columns)) != len(feature_columns):
            raise ValueError("Saved feature columns must be nonempty and unique.")
        missing = [column for column in feature_columns if column not in features]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        features = features.loc[:, feature_columns]

    if features.shape[1] == 0:
        raise ValueError("No predictor columns were found.")
    nonnumeric = [
        column
        for column in features
        if not pd.api.types.is_numeric_dtype(features[column])
        or pd.api.types.is_complex_dtype(features[column])
    ]
    if nonnumeric:
        raise ValueError(f"Feature columns must be numeric: {nonnumeric}")
    values = features.to_numpy(dtype=np.float64, na_value=np.nan)
    infinite = features.columns[np.isinf(values).any(axis=0)].tolist()
    if infinite:
        raise ValueError(f"Feature columns contain infinite values: {infinite}")
    return features


def load_training_data(
    path: str | Path = DEFAULT_TRAIN_PATH,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load predictors, finite DIC targets, and unique IDs in CSV row order."""
    frame = pd.read_csv(path)
    if len(frame) < 5:
        raise ValueError("Training data must contain at least 5 rows.")
    if TARGET_COLUMN not in frame:
        raise ValueError(f"Training CSV must contain the target column {TARGET_COLUMN!r}.")
    target = frame[TARGET_COLUMN]
    if (
        not pd.api.types.is_numeric_dtype(target)
        or pd.api.types.is_complex_dtype(target)
        or not np.isfinite(target.to_numpy(dtype=np.float64, na_value=np.nan)).all()
    ):
        raise ValueError("DIC targets must be numeric, nonmissing, and finite.")
    if "id" not in frame:
        raise ValueError("Training CSV must contain an 'id' column.")
    ids = frame["id"]
    if ids.isna().any() or not ids.is_unique:
        raise ValueError("Training IDs must be unique and nonmissing.")
    return feature_frame(frame), target.copy(), ids.copy()


def fit_preprocessing(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Fit imputation and scaling on training rows and return plain metadata."""
    features = feature_frame(X_train)
    if len(features) == 0 or len(features) != len(y_train):
        raise ValueError("Training features and targets must have the same nonzero length.")
    all_missing = features.columns[features.isna().all()].tolist()
    if all_missing:
        raise ValueError(f"Training feature columns are entirely missing: {all_missing}")
    targets = np.asarray(y_train, dtype=np.float64).reshape(-1, 1)
    if not np.isfinite(targets).all():
        raise ValueError("Training targets must be finite.")
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(features)
    feature_scaler = StandardScaler().fit(imputed)
    target_scaler = StandardScaler().fit(targets)
    return {
        "feature_columns": features.columns.tolist(),
        "feature_medians": imputer.statistics_.tolist(),
        "feature_means": feature_scaler.mean_.tolist(),
        "feature_scales": feature_scaler.scale_.tolist(),
        "target_mean": float(target_scaler.mean_[0]),
        "target_scale": float(target_scaler.scale_[0]),
    }


def transform_features(frame: pd.DataFrame, preprocessing: dict) -> np.ndarray:
    """Apply saved training statistics and return finite float32 predictors."""
    features = feature_frame(frame, preprocessing["feature_columns"])
    values = features.to_numpy(dtype=np.float64, na_value=np.nan)
    imputed = np.where(np.isnan(values), preprocessing["feature_medians"], values)
    transformed = (
        (imputed - np.asarray(preprocessing["feature_means"]))
        / np.asarray(preprocessing["feature_scales"])
    ).astype(np.float32)
    if not np.isfinite(transformed).all():
        raise ValueError("Transformed features must be finite.")
    return transformed


def transform_targets(y: pd.Series | np.ndarray, preprocessing: dict) -> np.ndarray:
    """Standardize targets with training statistics into an N-by-1 array."""
    values = np.asarray(y, dtype=np.float64)
    if values.ndim not in {1, 2} or (values.ndim == 2 and values.shape[1] != 1):
        raise ValueError("Targets must be a one-dimensional array or an N-by-1 array.")
    transformed = (
        (values.reshape(-1, 1) - preprocessing["target_mean"])
        / preprocessing["target_scale"]
    ).astype(np.float32)
    if not np.isfinite(transformed).all():
        raise ValueError("Transformed targets must be finite.")
    return transformed


def inverse_targets(values: np.ndarray, preprocessing: dict) -> np.ndarray:
    """Convert standardized predictions back to one-dimensional DIC values."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim not in {1, 2} or (values.ndim == 2 and values.shape[1] != 1):
        raise ValueError("Predictions must be one-dimensional or an N-by-1 array.")
    predictions = (
        values.reshape(-1) * preprocessing["target_scale"]
        + preprocessing["target_mean"]
    )
    if not np.isfinite(predictions).all():
        raise ValueError("Inverse-transformed predictions must be finite.")
    return predictions
