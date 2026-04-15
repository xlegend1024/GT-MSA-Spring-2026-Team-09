"""V1 Baseline Model — EDA-driven DCA weight computation.

Signals from EDA Executive Key Takeaways:
1. MA200 regime (bull/bear)
2. MVRV valuation zones (<1.0 accumulate, >3.5 reduce)
3. Exchange Net Flow (sustained inflow = caution)
4. Halving proximity (2-6 months pre-halving = increase)
5. Multi-signal confidence (MVRV + MA agreement)

Optional Polymarket overlays (loaded by variant flag):
- Crypto Index
- Trump Index
- US Affairs Index
"""

import logging
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from template.model_development_template import (
    _compute_stable_signal,
    allocate_sequential_stable,
    _clean_array,
)

# =============================================================================
# Constants
# =============================================================================

PRICE_COL = "PriceUSD_coinmetrics"
MVRV_COL = "CapMVRVCur"
FLOW_IN_COL = "FlowInExNtv"
FLOW_OUT_COL = "FlowOutExNtv"

MIN_W = 1e-6
MA_WINDOW = 200
DYNAMIC_STRENGTH = 4.0

# MVRV thresholds from EDA
MVRV_ACCUM = 1.0   # < 1.0 → max accumulation
MVRV_OVERHEAT = 3.5  # > 3.5 → reduce/pause

# Exchange flow: sustained inflow threshold
FLOW_SUSTAINED_DAYS = 5

# Halving dates
HALVING_DATES = pd.to_datetime([
    "2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20",
])

# Signal weights (base model)
W_MVRV = 0.35
W_MA = 0.20
W_FLOW = 0.15
W_HALVING = 0.10
W_CONFIDENCE = 0.20

# Polymarket signal weight (replaces proportional share of base signals)
W_POLY = 0.15  # When enabled, other weights scaled down proportionally

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


# =============================================================================
# Variant Enum
# =============================================================================

class Variant(Enum):
    BASE = "base"
    CRYPTO = "base+crypto"
    TRUMP = "base+trump"
    US_AFFAIRS = "base+us_affairs"
    CRYPTO_US = "base+crypto+us_affairs"
    CRYPTO_TRUMP = "base+crypto+trump"
    TRUMP_US = "base+trump+us_affairs"
    ALL_THREE = "base+crypto+trump+us_affairs"


# =============================================================================
# Polymarket Index Loading
# =============================================================================

def _load_poly_index(filename: str, col: str) -> pd.Series:
    """Load a Polymarket index parquet and return a date-indexed Series."""
    path = DATA_DIR / filename
    if not path.exists():
        logging.warning(f"Polymarket index not found: {path}")
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df[col]


def load_poly_indices() -> dict[str, pd.Series]:
    """Load all three Polymarket index time series."""
    return {
        "crypto": _load_poly_index("polymarket_crypto_index_ts.parquet", "crypto_poly_index"),
        "trump": _load_poly_index("trump_index_ts_interp.parquet", "trump_poly_index"),
        "us_affairs": _load_poly_index("us_affairs_index_ts_interp.parquet", "us_affairs_poly_index"),
    }


# =============================================================================
# Feature Engineering
# =============================================================================

def _halving_proximity(dates: pd.DatetimeIndex) -> pd.Series:
    """Compute halving proximity signal.

    Returns value in [0, 1]: 1.0 when 2-6 months before halving, 0 otherwise.
    """
    signal = pd.Series(0.0, index=dates)
    for hd in HALVING_DATES:
        pre_start = hd - pd.DateOffset(months=6)
        pre_end = hd - pd.DateOffset(months=2)
        mask = (dates >= pre_start) & (dates <= pre_end)
        # Ramp up linearly toward halving
        if mask.any():
            days_to_halving = (hd - dates[mask]).days
            max_days = (hd - pre_start).days
            signal[mask] = np.clip(1.0 - days_to_halving / max_days, 0, 1)
    return signal


def _sustained_inflow(flow_in: pd.Series, flow_out: pd.Series) -> pd.Series:
    """Detect sustained net inflow periods (≥5 consecutive days).

    Returns 1.0 during sustained inflow periods, 0 otherwise.
    """
    net_flow = flow_in - flow_out
    is_inflow = (net_flow > 0).astype(int)
    # Count consecutive inflow days
    groups = (is_inflow != is_inflow.shift()).cumsum()
    run_length = is_inflow.groupby(groups).cumsum()
    return (run_length >= FLOW_SUSTAINED_DAYS).astype(float)


def _confidence_signal(mvrv_z: pd.Series, price_vs_ma: pd.Series) -> pd.Series:
    """Multi-signal confidence from EDA Section 5.

    High confidence when MVRV and MA signals agree in direction.
    """
    # Normalize both to [-1, 1] where negative = buy signal
    z_signal = (-mvrv_z / 4).clip(-1, 1)
    ma_signal = (-price_vs_ma).clip(-1, 1)

    agree = ((z_signal > 0) & (ma_signal > 0)).astype(float) - \
            ((z_signal < 0) & (ma_signal < 0)).astype(float)

    strength = (z_signal.abs() + ma_signal.abs()) / 2
    confidence = agree * strength
    return confidence.fillna(0)


def _poly_change_signal(poly_index: pd.Series, window: int = 30) -> pd.Series:
    """Convert Polymarket index level to z-scored change signal.

    Uses rolling z-score of daily changes as the signal.
    """
    if poly_index.empty:
        return pd.Series(dtype=float)
    change = poly_index.diff()
    mean = change.rolling(window, min_periods=10).mean()
    std = change.rolling(window, min_periods=10).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = ((change - mean) / std).clip(-3, 3)
    return z.fillna(0)


def precompute_features(
    df: pd.DataFrame,
    variant: Variant = Variant.BASE,
) -> pd.DataFrame:
    """Compute all features for the given variant.

    All signal features are lagged by 1 day to prevent look-ahead bias.
    """
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found. Available: {list(df.columns)}")

    price = df[PRICE_COL].loc["2010-07-18":].copy()

    # --- MA200 signal ---
    ma200 = price.rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        price_vs_ma = ((price / ma200) - 1).clip(-1, 1).fillna(0)

    # --- MVRV signal ---
    if MVRV_COL in df.columns:
        mvrv = df[MVRV_COL].loc[price.index]
        mvrv_mean = mvrv.rolling(365, min_periods=180).mean()
        mvrv_std = mvrv.rolling(365, min_periods=180).std()
        mvrv_z = ((mvrv - mvrv_mean) / mvrv_std).clip(-4, 4).fillna(0)

        # Zone classification: <1.0 accumulate, >3.5 reduce
        mvrv_zone_signal = pd.Series(0.0, index=price.index)
        mvrv_zone_signal[mvrv < MVRV_ACCUM] = 1.0   # Buy signal
        mvrv_zone_signal[mvrv > MVRV_OVERHEAT] = -1.0  # Reduce signal
    else:
        mvrv_z = pd.Series(0.0, index=price.index)
        mvrv_zone_signal = pd.Series(0.0, index=price.index)

    # --- Exchange Flow signal ---
    if FLOW_IN_COL in df.columns and FLOW_OUT_COL in df.columns:
        flow_in = df[FLOW_IN_COL].loc[price.index].fillna(0)
        flow_out = df[FLOW_OUT_COL].loc[price.index].fillna(0)
        sustained = _sustained_inflow(flow_in, flow_out)
        # Sustained inflow = caution → negative signal
        flow_signal = -sustained
        # Sustained outflow = accumulate → positive signal
        net_flow = flow_in - flow_out
        is_outflow = (net_flow < 0).astype(int)
        out_groups = (is_outflow != is_outflow.shift()).cumsum()
        out_run = is_outflow.groupby(out_groups).cumsum()
        sustained_outflow = (out_run >= FLOW_SUSTAINED_DAYS).astype(float)
        flow_signal = flow_signal + sustained_outflow * 0.5
        flow_signal = flow_signal.clip(-1, 1)
    else:
        flow_signal = pd.Series(0.0, index=price.index)

    # --- Halving proximity ---
    halving_signal = _halving_proximity(price.index)

    # --- Confidence signal ---
    confidence = _confidence_signal(mvrv_z, price_vs_ma)

    # Build features DataFrame
    features = pd.DataFrame({
        PRICE_COL: price,
        "price_ma200": ma200,
        "price_vs_ma": price_vs_ma,
        "mvrv_z": mvrv_z,
        "mvrv_zone": mvrv_zone_signal,
        "flow_signal": flow_signal,
        "halving_signal": halving_signal,
        "confidence": confidence,
    }, index=price.index)

    # --- Polymarket overlays ---
    if variant != Variant.BASE:
        poly_indices = load_poly_indices()

        needs_crypto = variant in (Variant.CRYPTO, Variant.CRYPTO_US, Variant.CRYPTO_TRUMP, Variant.ALL_THREE)
        needs_trump = variant in (Variant.TRUMP, Variant.CRYPTO_TRUMP, Variant.TRUMP_US, Variant.ALL_THREE)
        needs_us = variant in (Variant.US_AFFAIRS, Variant.CRYPTO_US, Variant.TRUMP_US, Variant.ALL_THREE)

        if needs_crypto:
            crypto_z = _poly_change_signal(poly_indices["crypto"])
            features["poly_crypto"] = crypto_z.reindex(price.index, fill_value=0)

        if needs_trump:
            trump_z = _poly_change_signal(poly_indices["trump"])
            features["poly_trump"] = trump_z.reindex(price.index, fill_value=0)

        if needs_us:
            us_z = _poly_change_signal(poly_indices["us_affairs"])
            features["poly_us_affairs"] = us_z.reindex(price.index, fill_value=0)

    # Lag all signal columns by 1 day (prevent look-ahead bias)
    signal_cols = [c for c in features.columns if c not in [PRICE_COL, "price_ma200"]]
    features[signal_cols] = features[signal_cols].shift(1)
    features = features.fillna(0)

    return features


# =============================================================================
# Dynamic Multiplier
# =============================================================================

def compute_dynamic_multiplier(features: pd.DataFrame, variant: Variant) -> np.ndarray:
    """Compute weight multiplier from EDA-driven signals.

    Combines MA200, MVRV zone, exchange flow, halving proximity,
    and confidence signals. Optionally adds Polymarket overlays.
    """
    price_vs_ma = _clean_array(features["price_vs_ma"].values)
    mvrv_z = _clean_array(features["mvrv_z"].values)
    mvrv_zone = _clean_array(features["mvrv_zone"].values)
    flow_signal = _clean_array(features["flow_signal"].values)
    halving_signal = _clean_array(features["halving_signal"].values)
    confidence = _clean_array(features["confidence"].values)

    # Individual signals (all in "buy = positive" direction)
    ma_signal = -price_vs_ma  # Below MA → buy
    mvrv_signal = -mvrv_z * 0.5 + mvrv_zone * 0.5  # Z-score + zone
    flow_sig = flow_signal  # Already oriented
    halving_sig = halving_signal  # Pre-halving → buy
    conf_sig = confidence  # Agreement → amplify

    # Base weight allocation
    if variant == Variant.BASE:
        combined = (
            mvrv_signal * W_MVRV
            + ma_signal * W_MA
            + flow_sig * W_FLOW
            + halving_sig * W_HALVING
            + conf_sig * W_CONFIDENCE
        )
    else:
        # Scale down base weights to make room for Polymarket
        scale = 1.0 - W_POLY
        base_combined = (
            mvrv_signal * W_MVRV * scale
            + ma_signal * W_MA * scale
            + flow_sig * W_FLOW * scale
            + halving_sig * W_HALVING * scale
            + conf_sig * W_CONFIDENCE * scale
        )

        # Polymarket signal(s)
        poly_combined = np.zeros(len(features))
        poly_count = 0

        if "poly_crypto" in features.columns:
            poly_combined += _clean_array(features["poly_crypto"].values)
            poly_count += 1
        if "poly_trump" in features.columns:
            poly_combined += _clean_array(features["poly_trump"].values)
            poly_count += 1
        if "poly_us_affairs" in features.columns:
            poly_combined += _clean_array(features["poly_us_affairs"].values)
            poly_count += 1

        if poly_count > 0:
            poly_combined /= poly_count  # Average if multiple

        combined = base_combined + poly_combined * W_POLY

    adjustment = combined * DYNAMIC_STRENGTH
    adjustment = np.clip(adjustment, -5, 100)
    multiplier = np.exp(adjustment)
    return np.where(np.isfinite(multiplier), multiplier, 1.0)


# =============================================================================
# Weight Computation API
# =============================================================================

def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    variant: Variant = Variant.BASE,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date window."""
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n

    dyn = compute_dynamic_multiplier(df, variant)
    raw = base * dyn

    if n_past is None:
        n_past = n
    weights = allocate_sequential_stable(raw, n_past, locked_weights)
    return pd.Series(weights, index=df.index)


def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    variant: Variant = Variant.BASE,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date range with lock-on-compute stability."""
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame(
            {col: 0.0 for col in features_df.columns},
            index=missing,
        )
        features_df = pd.concat([features_df, placeholder]).sort_index()

    past_end = min(current_date, end_date)
    if start_date <= past_end:
        n_past = len(pd.date_range(start=start_date, end=past_end, freq="D"))
    else:
        n_past = 0

    weights = compute_weights_fast(
        features_df, start_date, end_date, variant, n_past, locked_weights
    )
    return weights.reindex(full_range, fill_value=0.0)
