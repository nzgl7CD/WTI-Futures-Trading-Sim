"""
Feature engineering for WTI futures (multi-ticker).

load_raw(path)
    Reads the long-format CSV (one row per ticker per date), pivots it wide,
    and returns a single DataFrame indexed by trading date with:
      - Open/High/Low/Close/Volume  = CL=F values only  (used by targets.py)
      - <TICKER>_<field>  columns   = features from ALL tickers
      - dayofweek, month            = calendar features

build_features(raw, include_today_open)
    Selects the feature columns (everything except CL=F raw OHLCV).
    include_today_open=True adds CLF_overnight_gap (Option B).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

TICKER_LABELS = {
    "CL=F": "CLF",
    "NG=F": "NGF",
    "GC=F": "GCF",
    "ES=F": "ESF",
    "^OVX": "OVX",
    "^VIX": "VIX",
}

# Per-ticker columns to include as features (lowercased in output)
FEAT_COLS = [
    "Close", "High", "Low",
    "log_return", "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    "rsi", "macd", "atr", "bollinger_pband", "sma_20", "ema_12",
]

# These CL=F columns stay as-is for targets.py; not used as features directly
_TARGET_OHLCV = {"Open", "High", "Low", "Close", "Volume"}


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.drop_duplicates(["timestamp", "Ticker"])

    # ---- CL=F OHLCV (index anchor + target source) ----
    clf = (
        df[df["Ticker"] == "CL=F"]
        .set_index("timestamp")
        .sort_index()[["Open", "High", "Low", "Close", "Volume"]]
    )

    # ---- Feature columns from every ticker (pivoted wide) ----
    pieces: list[pd.DataFrame] = []
    for ticker, label in TICKER_LABELS.items():
        sub = (
            df[df["Ticker"] == ticker]
            .set_index("timestamp")
            .sort_index()
        )
        available = [c for c in FEAT_COLS if c in sub.columns]
        piece = sub[available].copy()
        piece.columns = [f"{label}_{c.lower()}" for c in available]
        pieces.append(piece)

    features_wide = pd.concat(pieces, axis=1)

    # ---- Join on CL=F dates (left join keeps only dates CL=F traded) ----
    out = clf.join(features_wide, how="left")

    out["dayofweek"] = out.index.dayofweek
    out["month"] = out.index.month

    if not out.index.is_monotonic_increasing:
        out = out.sort_index()

    return out


def build_features(raw: pd.DataFrame, include_today_open: bool = False) -> pd.DataFrame:
    exclude = _TARGET_OHLCV  # raw OHLCV goes to targets, not features
    cols = [c for c in raw.columns if c not in exclude]
    X = raw[cols].copy()

    if include_today_open:
        # Known at open of day t; legitimate for Option B
        X["CLF_overnight_gap"] = np.log(raw["Open"] / raw["Close"].shift(1))

    return X
