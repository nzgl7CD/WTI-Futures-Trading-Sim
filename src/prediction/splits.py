"""
data splitting. NEVER shuffles. NEVER lets future data leak into
training.

2 split tools:

1. fixed_split(df, train_end, val_end)
       Single chronological split into train / validation / backtest holdout.
       Defaults match the project plan:
           train:    ...      -> 2022-12-31
           val:      2023-01-01 -> 2023-12-31
           backtest: 2024-01-01 -> end of data

2. WalkForward(start, end, train_window, step, expanding=True)
       Iterator that yields (train_idx, predict_idx) pairs for walk-forward
       retraining inside the backtest period. By default:
           - retrain monthly
           - expanding window (use ALL history up to t for training)
       This mimics how you would actually deploy the model.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Tuple
import pandas as pd


# ---------------------------------------------------------------------------
# Fixed three-way split
# ---------------------------------------------------------------------------
@dataclass
class Split:
    train: pd.DatetimeIndex
    val: pd.DatetimeIndex
    backtest: pd.DatetimeIndex

    def __repr__(self) -> str:
        def _r(idx: pd.DatetimeIndex) -> str:
            if len(idx) == 0:
                return "(empty)"
            return f"{idx.min().date()} -> {idx.max().date()}  n={len(idx)}"
        return (
            f"Split(\n"
            f"  train    = {_r(self.train)}\n"
            f"  val      = {_r(self.val)}\n"
            f"  backtest = {_r(self.backtest)}\n"
            f")"
        )


def fixed_split(
    df: pd.DataFrame,
    train_end: str = "2022-12-31",
    val_end: str = "2023-12-31",
) -> Split:
    """Three-way chronological split. Endpoints are INCLUSIVE."""
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be sorted ascending")

    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    if train_end_ts >= val_end_ts:
        raise ValueError("train_end must be before val_end")

    idx = df.index
    train = idx[idx <= train_end_ts]
    val = idx[(idx > train_end_ts) & (idx <= val_end_ts)]
    backtest = idx[idx > val_end_ts]
    return Split(train=train, val=val, backtest=backtest)


# ---------------------------------------------------------------------------
# Walk-forward iterator
# ---------------------------------------------------------------------------
@dataclass
class WalkForward:
    """
    Walk-forward iterator over a backtest period.

    Parameters
    ----------
    backtest_idx : pd.DatetimeIndex
        Dates we want predictions for (the holdout period).
    full_idx : pd.DatetimeIndex
        Full available index (used to look up training history).
    step : str
        How often to retrain. e.g. 'M' (monthly), 'W' (weekly), 'Q' (quarterly).
        With step='M', we predict all of January with a model trained through
        Dec 31, then predict all of February with a model trained through Jan 31,
        etc.
    expanding : bool
        If True, each retrain uses ALL history up to the cutoff (recommended).
        If False, uses a rolling window of `train_window_days`.
    train_window_days : int
        Used only when expanding=False.
    min_train_days : int
        Skip a fold if fewer than this many training rows are available.
    """
    backtest_idx: pd.DatetimeIndex
    full_idx: pd.DatetimeIndex
    step: str = "ME"  # month-end frequency in modern pandas
    expanding: bool = True
    train_window_days: int = 1500  # ~6 years
    min_train_days: int = 500       # ~2 years

    def __iter__(self) -> Iterator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        if len(self.backtest_idx) == 0:
            return
        # Build retrain cutoffs at the end of each step within the backtest period.
        # Each cutoff means: "train using all data up to and including this date".
        start = self.backtest_idx.min()
        end = self.backtest_idx.max()
        # Cutoff is the day BEFORE each step boundary, so we predict the next chunk
        # using a model trained strictly before it.
        boundaries = pd.date_range(start=start, end=end, freq=self.step)
        # Make sure the final partial chunk is included
        if len(boundaries) == 0 or boundaries[-1] < end:
            boundaries = boundaries.append(pd.DatetimeIndex([end]))

        prev_boundary = start - pd.Timedelta(days=1)
        for boundary in boundaries:
            # predict_idx = backtest dates in (prev_boundary, boundary]
            predict_mask = (self.backtest_idx > prev_boundary) & (self.backtest_idx <= boundary)
            predict_idx = self.backtest_idx[predict_mask]
            if len(predict_idx) == 0:
                prev_boundary = boundary
                continue

            # train_idx = all full_idx dates strictly BEFORE the first predict date
            first_predict = predict_idx.min()
            train_mask = self.full_idx < first_predict
            train_idx = self.full_idx[train_mask]
            if not self.expanding:
                train_idx = train_idx[-self.train_window_days:]

            if len(train_idx) < self.min_train_days:
                prev_boundary = boundary
                continue

            yield train_idx, predict_idx
            prev_boundary = boundary


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.features import load_raw

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_brent.csv"
    raw = load_raw(path)
    sp = fixed_split(raw, train_end="2015-01-09", val_end="2015-01-13")
    print(sp)
    print()

    # Build a fake longer index to test walk-forward
    import numpy as np
    fake_idx = pd.date_range("2015-01-01", "2026-04-01", freq="B")
    fake_df = pd.DataFrame(index=fake_idx, data={"x": np.arange(len(fake_idx))})
    sp2 = fixed_split(fake_df)
    print(sp2)
    print()
    print("Walk-forward folds (monthly retrain) - first 3 and last 1:")
    wf = WalkForward(backtest_idx=sp2.backtest, full_idx=fake_df.index, step="ME")
    folds = list(wf)
    print(f"Total folds: {len(folds)}")
    for i, (tr, pr) in enumerate(folds[:3]):
        print(f"  Fold {i}: train {tr.min().date()}..{tr.max().date()} ({len(tr)})  "
              f"predict {pr.min().date()}..{pr.max().date()} ({len(pr)})")
    if len(folds) > 3:
        tr, pr = folds[-1]
        print(f"  Fold {len(folds)-1}: train {tr.min().date()}..{tr.max().date()} ({len(tr)})  "
              f"predict {pr.min().date()}..{pr.max().date()} ({len(pr)})")