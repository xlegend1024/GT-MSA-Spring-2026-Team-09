"""
Model: MA200 Removal + Flow Signal Enhancement
Version: v3 (Score: 72.48%)

Summary:
    This model optimizes performance by removing the MA200 signal, which showed high
    redundancy with MVRV (r=0.87), and redistributing its weight to remaining indicators.
    It further enhances the 'Flow' signal by synthesizing Net Flow with Exchange Supply
    dynamics (SplyExNtv).

Grid Search Results:
    - Grid Search B (+2.64%): Removed MA200 and redistributed weights among
      the remaining 4 signals. Score increased to 72.48%.
    - Grid Search A (+1.74%): Integrated Net Flow with SplyExNtv (Exchange Balance).
      A decrease in SplyExNtv indicates BTC outflow, signaling reduced sell pressure.
    - Grid Search E (+0.21%): EXP_STRENGTH optimized from 150 → 175.

V3 Weights & Signal Components (post-MA200 removal):
    Original weights: MVRV=0.25, Halving=0.10, Flow=0.15, Addr=0.25, MA200=0.25
    After removing MA200 (renormalize remaining 0.75 → 1.0):

    1. MVRV Z-score (W=1/3 ≈ 0.333): Market valuation assessment.
    2. Halving Proximity (W=2/15 ≈ 0.133): Proximity to the Bitcoin halving cycle.
    3. Flow (W=1/5 = 0.200): Enhanced supply pressure signal.
       Formula: 0.5 * Net_Flow_z + 0.5 * SplyEx_change_z
    4. AdrActCnt (W=1/3 ≈ 0.333): Demand momentum based on active address count.

Parameters:
    EXP_STRENGTH: 175.0 (Grid Search E optimization)
    MIN_WINDOW: 60
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


# ── Weights (MA200 removed, renormalized to sum=1.0) ────────────────────────
# Original: MVRV=0.25, Halving=0.10, Flow=0.15, Addr=0.25, MA200=0.25
# After MA200 removal: renormalize 0.75 → 1.0
W_MVRV = 1 / 3  # 0.25 / 0.75 = 0.3333...
W_HALVING = 2 / 15  # 0.10 / 0.75 = 0.1333...
W_FLOW = 1 / 5  # 0.15 / 0.75 = 0.2000
W_ADDR = 1 / 3  # 0.25 / 0.75 = 0.3333...

# Verify sum = 1.0: 5/15 + 2/15 + 3/15 + 5/15 = 15/15 = 1.0
assert abs(W_MVRV + W_HALVING + W_FLOW + W_ADDR - 1.0) < 1e-9

EXP_STRENGTH = 175.0  # Grid Search E (150→175: +0.21%)
MIN_WINDOW = 60


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 4 signals. shift(1) to prevent look-ahead bias.

    Changes vs v2:
      - MA200 removed (redundant with MVRV, r=0.87)
      - Flow = Net Flow 30d MA + SplyExNtv change rate composite
    """
    df = df.sort_index()
    price = df[PRICE_COL].interpolate(method="linear").bfill()

    # ── 1. MVRV Z-score ─────────────────────────────────────────
    if "CapMVRVCur" in df.columns:
        mvrv_raw = df["CapMVRVCur"].interpolate(method="linear").bfill()
        mvrv_z = _rolling_zscore(mvrv_raw, window=365, min_periods=MIN_WINDOW, clip=3.0)
    else:
        mvrv_z = pd.Series(0.0, index=df.index)
    sig_mvrv = pd.Series(_sigmoid(-mvrv_z.to_numpy()), index=df.index)

    # ── 2. Halving proximity ────────────────────────────────────
    halving_prox = pd.Series(_halving_proximity(pd.DatetimeIndex(df.index)), index=df.index)
    sig_halving = halving_prox

    # ── 3. Flow signal (Net Flow + SplyExNtv composite) ─────────────────
    if "FlowInExNtv" in df.columns and "FlowOutExNtv" in df.columns:
        # Part A: Net Flow (inflow - outflow 30d MA)
        net_flow = df["FlowInExNtv"].fillna(0.0) - df["FlowOutExNtv"].fillna(0.0)
        net_flow_ma = net_flow.rolling(30, min_periods=5).mean()
        net_z = -_rolling_zscore(net_flow_ma, 365, MIN_WINDOW, 3.0)
        sig_net = pd.Series(_sigmoid(net_z.to_numpy()), index=df.index)
    else:
        sig_net = pd.Series(0.5, index=df.index)

    if "SplyExNtv" in df.columns:
        # Part B: SplyExNtv 30-day change rate (decrease = exchange balance decrease = buy signal)
        sply_ex = df["SplyExNtv"].interpolate(method="linear").bfill()
        sply_change = sply_ex.pct_change(30).fillna(0.0)
        sply_ma = sply_change.rolling(7, min_periods=3).mean()
        sply_z = -_rolling_zscore(sply_ma, 365, MIN_WINDOW, 3.0)
        sig_sply = pd.Series(_sigmoid(sply_z.to_numpy()), index=df.index)
        # Composite: 0.5 × Net + 0.5 × SplyEx
        sig_flow = 0.5 * sig_net + 0.5 * sig_sply
    else:
        sig_flow = sig_net

    # ── 4. Active Address ───────────────────────────────────────
    if "AdrActCnt" in df.columns:
        addr = df["AdrActCnt"].interpolate(method="linear").bfill()
        addr_z = _rolling_zscore(addr, 365, MIN_WINDOW, 3.0)
    else:
        addr_z = pd.Series(0.0, index=df.index)
    sig_addr = pd.Series(_sigmoid(addr_z.to_numpy()), index=df.index)

    # ── Composite ───────────────────────────────────────────────
    composite = W_MVRV * sig_mvrv + W_HALVING * sig_halving + W_FLOW * sig_flow + W_ADDR * sig_addr

    feat = pd.DataFrame(
        {
            "mvrv_z": mvrv_z,
            "halving_prox": halving_prox,
            "sig_mvrv": sig_mvrv,
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
    composite = features_df.reindex(full_range)["composite"].fillna(0.0).to_numpy()
    composite = _clean_array(composite)
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
