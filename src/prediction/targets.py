"""
builds the prediction targets.

all targets are expressed as LOG RETURNS relative to
a known reference price, so they are stationary and price-scale-free.

we reconstruct dollar prices later via price = ref * exp(target).

2 prediction modes (matching features.py):

    Option A  (decision time = end of day t, predict day t+1)
        ref_price = close[t]
        targets:
            target_a_open       = log(open[t+1]  / close[t])
            target_a_high       = log(high[t+1]  / close[t])
            target_a_low        = log(low[t+1]   / close[t])
            target_a_close      = log(close[t+1] / close[t])
            target_a_vol_range  = (high[t+1] - low[t+1]) / close[t]
            target_a_realvol_5  = rolling std of log_ret over [t+1 .. t+5]
            target_a_direction  = 1 if close[t+1] > close[t] else 0

    Option B  (decision time = open of day t+1, predict rest of day t+1)
        ref_price = open[t+1]   (known at decision time)
        targets:
            target_b_high       = log(high[t+1]  / open[t+1])
            target_b_low        = log(low[t+1]   / open[t+1])
            target_b_close      = log(close[t+1] / open[t+1])
            target_b_vol_range  = (high[t+1] - low[t+1]) / open[t+1]
            target_b_direction  = 1 if close[t+1] > open[t+1] else 0

NB: targets at row index `t` describe the FUTURE (day t+1).
This means the last row of any target frame is always NaN. After joining with
features, drop rows where any used target is NaN.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from fractdiff import frac_diff_ffd, find_min_d


def build_targets_a(df: pd.DataFrame) -> pd.DataFrame:
    """Targets for Option A: predict day t+1 using info up to end of day t."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    ref = c  # close[t]
    nxt_o = o.shift(-1)
    nxt_h = h.shift(-1)
    nxt_l = l.shift(-1)
    nxt_c = c.shift(-1)

    out = pd.DataFrame(index=df.index)
    out["target_a_open"]  = np.log(nxt_o / ref)
    out["target_a_high"]  = np.log(nxt_h / ref)
    out["target_a_low"]   = np.log(nxt_l / ref)
    out["target_a_close"] = np.log(nxt_c / ref)
    out["target_a_vol_range"] = (nxt_h - nxt_l) / ref
    direction = (nxt_c > c).astype("Int8")
    out["target_a_direction"] = direction.where(nxt_c.notna())

    # Forward 5-day realized vol: std of log returns over t+1..t+5.
    # Built so that row t contains the std computed from the next 5 returns.
    log_ret = np.log(c / c.shift(1))
    # std over t+1..t+5  ==  shift(-5) on a trailing rolling std of window 5
    out["target_a_realvol_5"] = log_ret.rolling(5, min_periods=5).std(ddof=0).shift(-5)

    return out


def build_targets_b(df: pd.DataFrame) -> pd.DataFrame:
    """Targets for Option B: predict rest of day t+1 given open[t+1]."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    ref = o.shift(-1)  # open[t+1]

    nxt_h = h.shift(-1)
    nxt_l = l.shift(-1)
    nxt_c = c.shift(-1)

    out = pd.DataFrame(index=df.index)
    out["target_b_high"]  = np.log(nxt_h / ref)
    out["target_b_low"]   = np.log(nxt_l / ref)
    out["target_b_close"] = np.log(nxt_c / ref)
    out["target_b_vol_range"] = (nxt_h - nxt_l) / ref
    direction = (nxt_c > ref).astype("Int8")
    out["target_b_direction"] = direction.where(nxt_c.notna() & ref.notna())
    return out


def build_targets_weekly(df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """
    Predict open[t+horizon] using info up to end of day t.

    horizon is in *trading* days — shift(-horizon) skips weekends/holidays
    automatically because the index only contains trading days.

    ref_price = close[t]
    target = log(open[t+horizon] / close[t])
    """
    c = df["Close"]
    o = df["Open"]
    out = pd.DataFrame(index=df.index)
    out[f"target_open_t{horizon}"] = np.log(o.shift(-horizon) / c)
    return out


def reconstruct_prices_a(close_t: pd.Series, preds: pd.DataFrame) -> pd.DataFrame:
    """Convert Option A log-return predictions back to dollar price levels."""
    out = pd.DataFrame(index=preds.index)
    if "target_a_open"  in preds: out["pred_open"]  = close_t * np.exp(preds["target_a_open"])
    if "target_a_high"  in preds: out["pred_high"]  = close_t * np.exp(preds["target_a_high"])
    if "target_a_low"   in preds: out["pred_low"]   = close_t * np.exp(preds["target_a_low"])
    if "target_a_close" in preds: out["pred_close"] = close_t * np.exp(preds["target_a_close"])
    return out


def reconstruct_prices_b(open_tp1: pd.Series, preds: pd.DataFrame) -> pd.DataFrame:
    """Convert Option B log-return predictions back to dollar price levels."""
    out = pd.DataFrame(index=preds.index)
    if "target_b_high"  in preds: out["pred_high"]  = open_tp1 * np.exp(preds["target_b_high"])
    if "target_b_low"   in preds: out["pred_low"]   = open_tp1 * np.exp(preds["target_b_low"])
    if "target_b_close" in preds: out["pred_close"] = open_tp1 * np.exp(preds["target_b_close"])
    return out


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.features import load_raw

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_brent.csv"
    raw = load_raw(path)

    ta = build_targets_a(raw)
    tb = build_targets_b(raw)

    print("Targets A:")
    print(ta.head())
    print()
    print("Targets B:")
    print(tb.head())
    print()
    # Sanity: last row should be NaN (no t+1 available)
    print("Last row A (should be NaN):")
    print(ta.iloc[-1])