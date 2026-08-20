import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """Weights for fixed-width window fractional differentiation."""
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < threshold:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


def frac_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    """
    Fixed-width window fractional differentiation (López de Prado).
    """
    w = get_weights_ffd(d, threshold)
    width = len(w) - 1

    df = {}
    series_ = series.ffill().dropna()

    for i in range(width, series_.shape[0]):
        loc0 = series_.index[i - width]
        loc1 = series_.index[i]
        if not np.isfinite(series_.loc[loc1]):
            continue
        df[loc1] = np.dot(w.T, series_.loc[loc0:loc1])[0, 0]

    return pd.Series(df, name=f"{series.name}_fracdiff_d{d:.2f}")


def find_min_d(series: pd.Series, d_range=None) -> pd.DataFrame:
    """
    Find the smallest d that makes the series stationary (ADF test).
    """
    if d_range is None:
        d_range = np.linspace(0.0, 1.0, 11)

    results = []
    for d in d_range:
        try:
            diffed = frac_diff_ffd(series, d)
            adf_stat = adfuller(diffed.dropna(), maxlag=1, regression="c", autolag=None)[0]
            results.append({"d": d, "adfStat": adf_stat})
        except Exception:
            continue
    return pd.DataFrame(results)