"""
the lightweight model. less hard to overfit.
one regressor per regression target. direction signals
are derived from sign(predicted_return) instead of a separate classifier.

Workflow:
    1. tune_hyperparams(X_train, y_train, X_val, y_val, target_name, n_trials=30)
          -> dict of best params (Optuna, time-aware, no shuffling)

    2. fit_models(X_train, y_train_df, params_per_target=None)
          -> dict[target_name -> trained Booster]

    3. predict(models, X) -> pd.DataFrame of predictions per target

    4. walk_forward_predict(X_full, y_full, splits, params_per_target,
                            retrain_step="ME") -> pd.DataFrame
          Produces honest out-of-sample predictions for the backtest period
          by retraining at the start of each step.

CLI usage:
    python -m src.model_lgbm <csv_path> [--option a|b|both] [--n-trials 30]
    Runs end-to-end on your CSV: features -> tune -> fit -> walk-forward
    -> metrics report.
"""

from __future__ import annotations
import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Make module runnable as script OR as `python -m src.model_lgbm`
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import load_raw, build_features
from src.prediction.targets import build_targets_a, build_targets_b
from src.prediction.splits import fixed_split, WalkForward
from src.prediction.metrics import summarize_per_target, regression_report, classification_report


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "verbose": -1,
    "seed": 42,
    "feature_pre_filter": False,  # let us change min_data_in_leaf across trials
}

NUM_BOOST_ROUND_DEFAULT = 2000
EARLY_STOPPING_DEFAULT = 100


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def build_dataset(
    raw: pd.DataFrame,
    option: str = "a",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (X, Y) for the chosen option.

    option='a': features without today's open, targets keyed off close[t]
    option='b': features INCLUDING today's open (overnight_gap), targets keyed
                off open[t+1]
    """
    if option == "a":
        X = build_features(raw, include_today_open=False)
        Y = build_targets_a(raw)
    elif option == "b":
        X = build_features(raw, include_today_open=True)
        Y = build_targets_b(raw)
    else:
        raise ValueError("option must be 'a' or 'b'")

    # Align indices and drop rows with any NaN in features OR targets.
    # We must be conservative here: a NaN in any used target means we can't
    # train on that row. NaNs in features will be passed through to LightGBM
    # which handles them natively, but we still need at least the warmup
    # period to elapse for the model to see complete features.
    df = X.join(Y, how="inner")
    return X.loc[df.index], Y.loc[df.index]


def _drop_target_nans(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    mask = y.notna()
    return X.loc[mask], y.loc[mask]


# ---------------------------------------------------------------------------
# Optuna tuning (time-aware: train on train split, score on val split)
# ---------------------------------------------------------------------------
def tune_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 30,
    timeout: Optional[int] = None,
    seed: int = 42,
) -> Dict:
    """Optuna search for one regression target. Returns best params dict."""
    X_train, y_train = _drop_target_nans(X_train, y_train)
    X_val, y_val = _drop_target_nans(X_val, y_val)

    def objective(trial: optuna.Trial) -> float:
        params = dict(DEFAULT_PARAMS)
        params.update({
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "min_data_in_leaf":  trial.suggest_int("min_data_in_leaf", 10, 100),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq":      trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1":         trial.suggest_float("lambda_l1", 1e-8, 1.0, log=True),
            "lambda_l2":         trial.suggest_float("lambda_l2", 1e-8, 1.0, log=True),
            "seed":              seed,
        })
        # Build datasets fresh per trial; LightGBM caches params on Dataset
        # which makes reuse across trials with different params unsafe.
        train_set = lgb.Dataset(X_train, label=y_train, params=params, free_raw_data=False)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set,
                              params=params, free_raw_data=False)
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=NUM_BOOST_ROUND_DEFAULT,
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_DEFAULT, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        preds = booster.predict(X_val, num_iteration=booster.best_iteration)
        rmse = float(np.sqrt(np.mean((y_val.values - preds) ** 2)))
        # store best_iteration so we can replay later
        trial.set_user_attr("best_iteration", int(booster.best_iteration))
        return rmse

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best = dict(DEFAULT_PARAMS)
    best.update(study.best_params)
    best["best_iteration"] = study.best_trial.user_attrs.get("best_iteration", 500)
    best["best_rmse"] = float(study.best_value)
    return best


# ---------------------------------------------------------------------------
# Fit / Predict
# ---------------------------------------------------------------------------
def fit_one(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict,
    num_boost_round: Optional[int] = None,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
) -> lgb.Booster:
    X, y = _drop_target_nans(X, y)
    p = {k: v for k, v in params.items() if k not in ("best_iteration", "best_rmse")}
    train_set = lgb.Dataset(X, label=y)

    callbacks = [lgb.log_evaluation(0)]
    valid_sets = [train_set]
    valid_names = ["train"]

    if X_val is not None and y_val is not None:
        Xv, yv = _drop_target_nans(X_val, y_val)
        if len(yv) > 0:
            valid_sets.append(lgb.Dataset(Xv, label=yv, reference=train_set))
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(EARLY_STOPPING_DEFAULT, verbose=False))

    rounds = num_boost_round or params.get("best_iteration") or NUM_BOOST_ROUND_DEFAULT
    booster = lgb.train(
        p, train_set,
        num_boost_round=rounds,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return booster


def fit_models(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    params_per_target: Dict[str, Dict],
    X_val: Optional[pd.DataFrame] = None,
    Y_val: Optional[pd.DataFrame] = None,
    target_cols: Optional[List[str]] = None,
) -> Dict[str, lgb.Booster]:
    """Train one regressor per (regression) target. Direction is derived later."""
    if target_cols is None:
        target_cols = [c for c in Y_train.columns if not c.endswith("_direction")]

    models: Dict[str, lgb.Booster] = {}
    for tcol in target_cols:
        params = params_per_target.get(tcol, DEFAULT_PARAMS)
        Xv = X_val
        yv = Y_val[tcol] if (Y_val is not None and tcol in Y_val.columns) else None
        models[tcol] = fit_one(X_train, Y_train[tcol], params, X_val=Xv, y_val=yv)
    return models


def predict(models: Dict[str, lgb.Booster], X: pd.DataFrame) -> pd.DataFrame:
    """Predict every regression target. Direction columns are derived from
    sign of the corresponding *_close prediction."""
    out = pd.DataFrame(index=X.index)
    for tname, m in models.items():
        out[tname] = m.predict(X, num_iteration=m.best_iteration)

    # Derive direction targets from sign of next-day-close prediction.
    if "target_a_close" in out.columns:
        out["target_a_direction"] = (out["target_a_close"] > 0).astype(int)
    if "target_b_close" in out.columns:
        out["target_b_direction"] = (out["target_b_close"] > 0).astype(int)

    return out


# ---------------------------------------------------------------------------
# Walk-forward driver
# ---------------------------------------------------------------------------
def walk_forward_predict(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    backtest_idx: pd.DatetimeIndex,
    params_per_target: Dict[str, Dict],
    step: str = "ME",
    expanding: bool = True,
    target_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Retrain at every step boundary and predict the following block."""
    wf = WalkForward(backtest_idx=backtest_idx, full_idx=X.index,
                     step=step, expanding=expanding)
    chunks: List[pd.DataFrame] = []
    folds = list(wf)
    for i, (train_idx, predict_idx) in enumerate(folds):
        Xtr, Ytr = X.loc[train_idx], Y.loc[train_idx]
        Xpr = X.loc[predict_idx]
        models = fit_models(Xtr, Ytr, params_per_target, target_cols=target_cols)
        chunk = predict(models, Xpr)
        chunks.append(chunk)
        if verbose:
            print(f"  fold {i+1}/{len(folds)}  "
                  f"train {train_idx.min().date()}..{train_idx.max().date()} ({len(train_idx)})  "
                  f"predict {predict_idx.min().date()}..{predict_idx.max().date()} ({len(predict_idx)})")
    return pd.concat(chunks).sort_index() if chunks else pd.DataFrame()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _run(option: str, csv_path: str, n_trials: int, train_end: str,
         val_end: str) -> pd.DataFrame:
    print(f"\n========== Option {option.upper()} ==========")
    raw = load_raw(csv_path)
    print(f"Raw rows: {len(raw)}  ({raw.index.min().date()} -> {raw.index.max().date()})")

    X, Y = build_dataset(raw, option=option)
    sp = fixed_split(raw, train_end=train_end, val_end=val_end)
    print(sp)

    target_cols = [c for c in Y.columns if not c.endswith("_direction")]
    params_per_target: Dict[str, Dict] = {}

    print(f"\n[1/3] Optuna tuning ({n_trials} trials per target)")
    for tcol in target_cols:
        print(f"  tuning {tcol}...", end="", flush=True)
        Xt, yt = X.loc[sp.train], Y.loc[sp.train, tcol]
        Xv, yv = X.loc[sp.val],   Y.loc[sp.val, tcol]
        best = tune_hyperparams(Xt, yt, Xv, yv, n_trials=n_trials)
        params_per_target[tcol] = best
        print(f" best_rmse={best['best_rmse']:.6f}  best_iter={best['best_iteration']}")

    # Fit final model on all pre-backtest data (train + val combined).
    # The backtest period (2024-today) is never seen during training — clean OOS test.
    pre_backtest = sp.train.append(sp.val)
    X_pre = X.loc[pre_backtest]
    Y_pre = Y.loc[pre_backtest]

    print(f"\n[2/3] Fitting final model on pre-backtest data "
          f"({pre_backtest.min().date()} -> {pre_backtest.max().date()}, n={len(pre_backtest)})")
    models = fit_models(X_pre, Y_pre, params_per_target, target_cols=target_cols)

    X_bt = X.loc[sp.backtest]
    preds = predict(models, X_bt)

    print(f"\n[3/3] Backtest metrics  "
          f"({sp.backtest.min().date()} -> {sp.backtest.max().date()}, "
          f"n={len(preds)} predictions)")
    Y_bt = Y.loc[preds.index]
    summary = summarize_per_target(Y_bt, preds)
    print(summary.to_string(index=False))

    out_path = f"data/preds_lgbm_option_{option}.csv"
    preds.to_csv(out_path)
    print(f"\nSaved predictions to {out_path}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv_path", help="Path to the OHLCV csv (e.g. data/brent.csv)")
    p.add_argument("--option", choices=["a", "b", "both"], default="both")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--train-end", default="2022-12-31")
    p.add_argument("--val-end", default="2023-12-31")
    args = p.parse_args()

    if args.option in ("a", "both"):
        _run("a", args.csv_path, args.n_trials, args.train_end, args.val_end)
    if args.option in ("b", "both"):
        _run("b", args.csv_path, args.n_trials, args.train_end, args.val_end)


if __name__ == "__main__":
    main()