"""
ridge_knn_metric.py - evaluation helpers for the Ridge + KNN models.

Same functions and output format as the shared metrics.py, so the results CSV
lines up with the other models' CSVs. Save as src/models/ridge_knn_metric.py.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

from src.features.dataset import BAND_LABELS, RESULTS_DIR


def _ape(y_true, y_pred):
    """Absolute percentage error per row; rows with a true value of 0 become NaN instead of inf."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    safe = np.where(y_true == 0, np.nan, y_true)
    return np.abs((y_true - y_pred) / safe) * 100


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ape = _ape(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "MAPE_%": float(np.nanmean(ape)),
        "MedianAPE_%": float(np.nanmedian(ape)),
    }


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro_F1": f1_score(y_true, y_pred, average="macro"),
        "Weighted_F1": f1_score(y_true, y_pred, average="weighted"),
    }


def band_confusion(y_true, y_pred) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=BAND_LABELS)
    return pd.DataFrame(cm, index=BAND_LABELS, columns=BAND_LABELS)


def error_by_group(y_true, y_pred, groups) -> pd.DataFrame:
    """MAPE per group (e.g. price band or make) - used for error analysis."""
    df = pd.DataFrame({"y": np.asarray(y_true, float),
                       "p": np.asarray(y_pred, float),
                       "g": np.asarray(groups)})
    df["ape"] = _ape(df["y"], df["p"])
    out = df.groupby("g")["ape"].agg(["count", "mean", "median"])
    return out.rename(columns={"mean": "MAPE_%", "median": "MedianAPE_%"})


def save_metrics(rows: list, filename: str = "ridge_knn_metrics.csv") -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / filename, index=False)
    print(f"\n[+] Metrics saved to: {RESULTS_DIR / filename}")
