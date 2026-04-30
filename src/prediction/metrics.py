"""
metrics

1. STATISTICAL metrics (per-target):
    - RMSE, MAE on the predicted log-returns
    - Directional accuracy: % of times sign(y_pred) == sign(y_true) R^2 vs naive zero baseline

2. BASELINES we have to beat:
    - naive_zero:  always predict return = 0   ("tomorrow == today")
    - naive_mean:  predict the rolling mean of recent returns
    - persistence: predict yesterday's return again

Functions:
    regression_report(y_true, y_pred)        -> dict
    classification_report(y_true, y_pred)    -> dict
    naive_zero_baseline(y_true)              -> pd.Series of zeros
    naive_persistence_baseline(returns)      -> pd.Series (yesterday's return)
    summarize_per_target(df_true, df_pred)   -> pd.DataFrame summary
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------
def regression_report(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """RMSE, MAE, directional accuracy, and R^2-vs-zero on returns."""
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)
    mask = y_true.notna() & y_pred.notna()
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "dir_acc": np.nan, "r2_vs_zero": np.nan}

    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    # Directional accuracy on signed-return targets only. For range/vol targets
    # this metric is meaningless but harmless (always ~1).
    dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))

    # R^2 vs naive zero predictor. Negative means naive zero beats the model.
    ss_res = float(np.sum(err ** 2))
    ss_naive = float(np.sum(y_true ** 2))  # error of always predicting 0
    r2_vs_zero = 1.0 - ss_res / ss_naive if ss_naive > 0 else np.nan

    return {
        "n": int(len(y_true)),
        "rmse": rmse,
        "mae": mae,
        "dir_acc": dir_acc,
        "r2_vs_zero": float(r2_vs_zero) if r2_vs_zero == r2_vs_zero else np.nan,
    }


# ---------------------------------------------------------------------------
# Classification metrics (for the *_direction targets)
# ---------------------------------------------------------------------------
def classification_report(y_true: pd.Series, y_pred_proba: pd.Series, threshold: float = 0.5) -> Dict[str, float]:
    """Accuracy, precision, recall, F1 for binary direction. y_pred_proba can be
    either probabilities in [0,1] or hard 0/1 labels."""
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred_proba).astype(float)
    mask = y_true.notna() & y_pred.notna()
    y_true = y_true[mask].astype(int)
    y_hat = (y_pred[mask] >= threshold).astype(int)
    if len(y_true) == 0:
        return {"n": 0, "accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "base_rate": np.nan}

    tp = int(((y_hat == 1) & (y_true == 1)).sum())
    fp = int(((y_hat == 1) & (y_true == 0)).sum())
    fn = int(((y_hat == 0) & (y_true == 1)).sum())
    tn = int(((y_hat == 0) & (y_true == 0)).sum())
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision and recall and not np.isnan(precision) and not np.isnan(recall)) else np.nan

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy),
        "precision": float(precision) if precision == precision else np.nan,
        "recall": float(recall) if recall == recall else np.nan,
        "f1": float(f1) if f1 == f1 else np.nan,
        "base_rate": float(y_true.mean()),  # what % of the time the market went up
    }


# ---------------------------------------------------------------------------
# Naive baselines
# ---------------------------------------------------------------------------
def naive_zero_baseline(y_true: pd.Series) -> pd.Series:
    """Always predict zero return. The target to beat."""
    return pd.Series(0.0, index=y_true.index)


def naive_persistence_baseline(log_returns: pd.Series) -> pd.Series:
    """Predict tomorrow's return == today's return. Indexed same as input."""
    return log_returns.copy()  # caller is responsible for proper alignment


# ---------------------------------------------------------------------------
# Summary table across multiple targets
# ---------------------------------------------------------------------------
def summarize_per_target(
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    direction_cols: tuple = ("target_a_direction", "target_b_direction"),
) -> pd.DataFrame:
    """Produce a per-target metrics table. Direction targets get classification
    metrics; everything else gets regression metrics."""
    rows = []
    for col in y_pred_df.columns:
        if col not in y_true_df.columns:
            continue
        if col in direction_cols:
            r = classification_report(y_true_df[col], y_pred_df[col])
            r["target"] = col
            r["kind"] = "classification"
        else:
            r = regression_report(y_true_df[col], y_pred_df[col])
            r["target"] = col
            r["kind"] = "regression"
        rows.append(r)
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    cols = ["target", "kind", "n", "rmse", "mae", "dir_acc", "r2_vs_zero",
            "accuracy", "precision", "recall", "f1", "base_rate"]
    cols = [c for c in cols if c in df.columns]
    return df[cols]


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    y_true_ret = pd.Series(rng.normal(0, 0.02, n))
    # A "model" that has slight skill: 55% directional accuracy, low magnitude
    skill_signal = np.sign(y_true_ret) * rng.uniform(0, 1, n)
    flip = rng.random(n) < 0.45  # flip 45% of signs
    skill_signal = np.where(flip, -skill_signal, skill_signal)
    y_pred_ret = pd.Series(skill_signal * 0.005)

    print("Regression report:")
    print(regression_report(y_true_ret, y_pred_ret))
    print()

    y_true_dir = (y_true_ret > 0).astype(int)
    y_pred_proba = pd.Series(rng.uniform(0, 1, n))
    # bias prediction toward truth
    y_pred_proba = 0.4 * y_pred_proba + 0.6 * y_true_dir + rng.normal(0, 0.1, n)
    y_pred_proba = y_pred_proba.clip(0, 1)
    print("Classification report:")
    print(classification_report(y_true_dir, y_pred_proba))
    print()

    # summary across multiple targets
    yt = pd.DataFrame({"target_a_close": y_true_ret, "target_a_direction": y_true_dir})
    yp = pd.DataFrame({"target_a_close": y_pred_ret, "target_a_direction": y_pred_proba})
    print("Summary table:")
    print(summarize_per_target(yt, yp).to_string(index=False))