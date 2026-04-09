"""
Model — Multi-Signal DCA Strategy
Features (EDA-justified):
  1. MVRV Z-score (365d rolling)         — valuation regime  ★★★
  2. MA200 log deviation                 — trend regime      ★★★
  3. Halving proximity exp(-|days|/180)  — cycle timing      ★★
  4. Net exchange flow z-score (30d MA)  — supply pressure   ★★

Architecture: example_1 standard interface
  - precompute_features(df) -> features_df  [all signals lagged 1 day]
  - compute_window_weights(features_df, start_date, end_date, current_date,
                           locked_weights=None) -> pd.Series [sum=1.0 ±1e-5]
  - allocate_sequential_stable() for weight stability guarantee
"""

import numpy as np
import pandas as pd

from template.model_development_template import (
    allocate_sequential_stable,
    _clean_array,
)

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

# ── Signal weights (must sum to 1.0) ─────────────────────────────────────────
W_MVRV = 0.35  # ★★★ valuation
W_MA200 = 0.35  # ★★★ trend
W_HALVING = 0.15  # ★★  cycle timing
W_FLOW = 0.15  # ★★  supply pressure

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


# ─────────────────────────────────────────────────────────────────────────────
# 1. precompute_features
# ─────────────────────────────────────────────────────────────────────────────


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features from raw CoinMetrics DataFrame.

    Input
    -----
    df : DataFrame returned by load_data()
        - Index : DatetimeIndex named 'time' (daily, sorted)
        - Required columns : PriceUSD_coinmetrics, CapMVRVCur,
                             FlowInExNtv, FlowOutExNtv

    CRITICAL: All signals are lagged 1 day (shift(1)) to prevent
    look-ahead bias — today's weight uses only yesterday's information.

    Returns
    -------
    pd.DataFrame indexed by date with columns:
        mvrv_z, ma200_dev, halving_prox, flow_z,
        sig_mvrv, sig_ma200, sig_halving, sig_flow, composite
    """
    df = df.sort_index()

    # ── 1. MVRV Z-score (365d rolling) ──────────────────────────────────────
    if "CapMVRVCur" in df.columns:
        mvrv_raw = df["CapMVRVCur"].interpolate(method="linear").bfill()
        mvrv_z = _rolling_zscore(mvrv_raw, window=365, min_periods=60, clip=3.0)
    else:
        mvrv_z = pd.Series(0.0, index=df.index)

    # ── 2. MA200 log deviation ───────────────────────────────────────────────
    price = df[PRICE_COL].interpolate(method="linear").bfill()
    ma200 = price.rolling(200, min_periods=30).mean()
    ma200_dev = (np.log(price / ma200)).clip(-1.0, 1.0)
    ma200_dev = pd.Series(ma200_dev, index=df.index).fillna(0.0)

    # ── 3. Halving proximity exp(-|days|/180) ────────────────────────────────
    halving_prox = pd.Series(_halving_proximity(pd.DatetimeIndex(df.index)), index=df.index)

    # ── 4. Net exchange flow z-score (30d MA, sign-inverted) ─────────────────
    #   outflow (negative net flow) = supply leaves exchanges → buy signal
    if "FlowInExNtv" in df.columns and "FlowOutExNtv" in df.columns:
        net_flow = df["FlowInExNtv"] - df["FlowOutExNtv"]
        net_flow_ma = net_flow.rolling(30, min_periods=5).mean()
        flow_z = _rolling_zscore(net_flow_ma, window=365, min_periods=60, clip=3.0)
        flow_z = -flow_z  # invert: strong outflow → positive buy signal
    else:
        flow_z = pd.Series(0.0, index=df.index)

    # ── 5. Convert each signal to (0, 1) buy-strength via sigmoid ───────────
    #   low mvrv  → high buy strength   : sigmoid(-mvrv_z)
    #   below MA  → high buy strength   : sigmoid(-ma200_dev * 3)
    #   near halving → high strength    : already [0,1]
    #   net outflow → high strength      : sigmoid(flow_z)
    sig_mvrv = pd.Series(_sigmoid(np.asarray(mvrv_z.values) * -1), index=df.index)
    sig_ma200 = pd.Series(_sigmoid(np.asarray(ma200_dev.values) * -3), index=df.index)
    sig_halving = halving_prox  # already [0,1]
    sig_flow = pd.Series(_sigmoid(np.asarray(flow_z.values)), index=df.index)

    # ── 6. Composite signal (weighted sum) ───────────────────────────────────
    composite = (
        W_MVRV * sig_mvrv + W_MA200 * sig_ma200 + W_HALVING * sig_halving + W_FLOW * sig_flow
    )

    # ── 7. Build features DataFrame ─────────────────────────────────────────
    feat = pd.DataFrame(
        {
            "mvrv_z": mvrv_z,
            "ma200_dev": ma200_dev,
            "halving_prox": halving_prox,
            "flow_z": flow_z,
            "sig_mvrv": sig_mvrv,
            "sig_ma200": sig_ma200,
            "sig_halving": sig_halving,
            "sig_flow": sig_flow,
            "composite": composite,
        },
        index=df.index,
    )

    # ── CRITICAL: Lag 1 day — prevent look-ahead bias ────────────────────────
    feat = feat.shift(1).fillna(0.0)

    return feat


# ─────────────────────────────────────────────────────────────────────────────
# 2. compute_window_weights
# ─────────────────────────────────────────────────────────────────────────────


def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """
    Return daily DCA weights summing to exactly 1.0 (±1e-5).

    Modes
    -----
    BACKTEST  (locked_weights=None, current_date == end_date):
        All weights come from composite signal.

    PRODUCTION (locked_weights provided):
        Past weights are locked; future dates use uniform allocation
        (handled inside allocate_sequential_stable).

    Parameters
    ----------
    features_df   : output of precompute_features()
    start_date    : first date of investment window
    end_date      : last date of investment window
    current_date  : "today" boundary
    locked_weights: pre-locked weights array (len = n_past)

    Returns
    -------
    pd.Series of weights indexed by date, sum = 1.0
    """
    full_range = pd.date_range(start_date, end_date, freq="D")
    n = len(full_range)

    # ── Align composite signal to window ─────────────────────────────────────
    composite = features_df.reindex(full_range)["composite"].fillna(0.0).values
    composite = _clean_array(np.asarray(composite))

    # ── Raw weights: exp amplification of composite buy-signal ───────────────
    # composite ∈ (0,1) → raw ∈ (1, e^2) range with strength=2
    raw_w = np.exp(composite * 2.0)
    raw_w = np.clip(raw_w, MIN_W, None)

    # ── n_past: how many days are "past" (weights get locked) ────────────────
    if locked_weights is not None:
        n_past = len(locked_weights)
    else:
        # backtest mode: current_date == end_date → all days are past
        past_end = min(current_date, end_date)
        n_past = len(pd.date_range(start_date, past_end, freq="D"))
        n_past = min(n_past, n)

    # ── Stable allocation (past locked, last day absorbs remainder) ──────────
    weights = allocate_sequential_stable(
        raw=raw_w,
        n_past=n_past,
        locked_weights=locked_weights,
    )

    # ── Final safety: clip negatives, renormalize ─────────────────────────────
    weights = np.clip(weights, MIN_W, None)
    weights = weights / weights.sum()

    return pd.Series(weights, index=full_range, name="weight")
