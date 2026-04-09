"""
Model  — AdrActCnt Signal for Network Demand Discovery
Based on: model_v2_addr (Score 56.78%, Win Rate 71.80%)

vs model_v2_addr:
  MVRV Z-score  : 0.30 → 0.25
  MA200 dev     : 0.30 → 0.25
  Halving prox  : 0.15 (maintained)
  Net flow      : 0.15 (maintained)
  AdrActCnt     : 0.10 → 0.20  ★ strengthened
  exp_strength  : 3.5  (maintained)
"""

import numpy as np
import pandas as pd

from template.model_development_template import allocate_sequential_stable, _clean_array

# ── Price column (source of truth) ───────────────────────────────────────────
PRICE_COL = "PriceUSD_coinmetrics"

# ── Known halving dates (public knowledge, no look-ahead) ────────────────────
HALVING_DATES = pd.to_datetime(
    [
        "2012-11-28",
        "2016-07-09",
        "2020-05-11",
        "2024-04-20",
        "2028-03-15",  # estimated
    ]
)

MIN_W = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _rolling_zscore(
    series: pd.Series, window: int = 365, min_periods: int = 60, clip: float = 3.0
) -> pd.Series:
    """Rolling z-score, clipped to ±clip, NaN filled with 0."""
    mu = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    std = std.replace(0, np.nan)
    z = ((series - mu) / std).clip(-clip, clip)
    return z.fillna(0.0)


def _halving_proximity(index: pd.DatetimeIndex) -> np.ndarray:
    """
    exp(-|days_to_nearest_halving| / 180)
    Range [0, 1]. Only uses past/known halving dates → no look-ahead.
    """
    prox = np.zeros(len(index))
    for i, d in enumerate(index):
        diffs = np.abs((HALVING_DATES - d).days)
        prox[i] = float(np.exp(-diffs.min() / 180.0))
    return prox


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid mapped to (0, 1)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


W_MVRV = 0.25
W_MA200 = 0.25
W_HALVING = 0.15
W_FLOW = 0.15
W_ADDR = 0.20  # ★ strengthened

EXP_STRENGTH = 3.5


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()

    if "CapMVRVCur" in df.columns:
        mvrv_raw = df["CapMVRVCur"].interpolate(method="linear").bfill()
        mvrv_z = _rolling_zscore(mvrv_raw, window=365, min_periods=60, clip=3.0)
    else:
        mvrv_z = pd.Series(0.0, index=df.index)

    price = df[PRICE_COL].interpolate(method="linear").bfill()
    ma200 = price.rolling(200, min_periods=30).mean()
    ma200_dev = (np.log(price / ma200)).clip(-1.0, 1.0)
    ma200_dev = pd.Series(ma200_dev, index=df.index).fillna(0.0)

    halving_prox = pd.Series(_halving_proximity(pd.DatetimeIndex(df.index)), index=df.index)

    if "FlowInExNtv" in df.columns and "FlowOutExNtv" in df.columns:
        net_flow = df["FlowInExNtv"] - df["FlowOutExNtv"]
        net_flow_ma = net_flow.rolling(30, min_periods=5).mean()
        flow_z = -_rolling_zscore(net_flow_ma, window=365, min_periods=60, clip=3.0)
    else:
        flow_z = pd.Series(0.0, index=df.index)

    if "AdrActCnt" in df.columns:
        addr = df["AdrActCnt"].interpolate(method="linear").bfill()
        addr_z = _rolling_zscore(addr, window=365, min_periods=60, clip=3.0)
    else:
        addr_z = pd.Series(0.0, index=df.index)

    sig_mvrv = pd.Series(_sigmoid(np.asarray(mvrv_z.values) * -1), index=df.index)
    sig_ma200 = pd.Series(_sigmoid(np.asarray(ma200_dev.values) * -3), index=df.index)
    sig_halving = halving_prox
    sig_flow = pd.Series(_sigmoid(np.asarray(flow_z.values)), index=df.index)
    sig_addr = pd.Series(_sigmoid(np.asarray(addr_z.values)), index=df.index)

    composite = (
        W_MVRV * sig_mvrv
        + W_MA200 * sig_ma200
        + W_HALVING * sig_halving
        + W_FLOW * sig_flow
        + W_ADDR * sig_addr
    )

    feat = pd.DataFrame(
        {
            "mvrv_z": mvrv_z,
            "ma200_dev": ma200_dev,
            "halving_prox": halving_prox,
            "flow_z": flow_z,
            "addr_z": addr_z,
            "sig_mvrv": sig_mvrv,
            "sig_ma200": sig_ma200,
            "sig_halving": sig_halving,
            "sig_flow": sig_flow,
            "sig_addr": sig_addr,
            "composite": composite,
        },
        index=df.index,
    )

    feat = feat.shift(1).fillna(0.0)
    return feat


def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    full_range = pd.date_range(start_date, end_date, freq="D")
    composite = features_df.reindex(full_range)["composite"].fillna(0.0).values
    composite = _clean_array(np.asarray(composite))
    raw_w = np.clip(np.exp(composite * EXP_STRENGTH), MIN_W, None)

    if locked_weights is not None:
        n_past = len(locked_weights)
    else:
        past_end = min(current_date, end_date)
        n_past = min(len(pd.date_range(start_date, past_end, freq="D")), len(full_range))

    weights = allocate_sequential_stable(raw_w, n_past, locked_weights)
    weights = np.clip(weights, MIN_W, None)
    weights = weights / weights.sum()
    return pd.Series(weights, index=full_range, name="weight")
