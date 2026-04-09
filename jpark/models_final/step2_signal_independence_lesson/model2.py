"""
Model V2 — 5-Signal DCA Strategy (NVT Proxy Addition)

Features (EDA-justified):
  1. MVRV Z-score (365d rolling)         — valuation regime       ★★★  W=0.30
  2. MA200 log deviation                 — trend regime           ★★★  W=0.30
  3. Halving proximity exp(-|days|/180)  — cycle timing           ★★   W=0.15
  4. Net exchange flow z-score (30d MA)  — supply pressure        ★★   W=0.15
  5. NVT Proxy z-score (30d MA, 365d)   — network utility ratio  ★★   W=0.10  ★NEW

EDA Evidence (Section 6.2):
  - NVT proxy median 102.97; < 50 → 1,048 undervaluation days
  - NVT low + volume expansion → durable bottoms
  - Independent signal from MVRV (network utility vs realized value)

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


# ── Signal weights (must sum to 1.0) ─────────────────────────────────────────
W_MVRV = 0.30  # ★★★ valuation  (0.35 → 0.30)
W_MA200 = 0.30  # ★★★ trend      (0.35 → 0.30)
W_HALVING = 0.15  # ★★  cycle timing
W_FLOW = 0.15  # ★★  supply pressure
W_NVT = 0.10  # ★★  network utility ratio  ← NEW

EXP_STRENGTH = 3.5  # keep model_v2_strong amplification


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()

    # ── 1. MVRV Z-score ──────────────────────────────────────────────────────
    if "CapMVRVCur" in df.columns:
        mvrv_raw = df["CapMVRVCur"].interpolate(method="linear").bfill()
        mvrv_z = _rolling_zscore(mvrv_raw, window=365, min_periods=60, clip=3.0)
    else:
        mvrv_z = pd.Series(0.0, index=df.index)

    # ── 2. MA200 log deviation ────────────────────────────────────────────────
    price = df[PRICE_COL].interpolate(method="linear").bfill()
    ma200 = price.rolling(200, min_periods=30).mean()
    ma200_dev = pd.Series(np.log(price / ma200).clip(-1.0, 1.0), index=df.index).fillna(0.0)

    # ── 3. Halving proximity ──────────────────────────────────────────────────
    halving_prox = pd.Series(_halving_proximity(pd.DatetimeIndex(df.index)), index=df.index)

    # ── 4. Net exchange flow z-score (inverted) ───────────────────────────────
    if "FlowInExNtv" in df.columns and "FlowOutExNtv" in df.columns:
        net_flow = df["FlowInExNtv"] - df["FlowOutExNtv"]
        net_flow_ma = net_flow.rolling(30, min_periods=5).mean()
        flow_z = -_rolling_zscore(net_flow_ma, window=365, min_periods=60, clip=3.0)
    else:
        flow_z = pd.Series(0.0, index=df.index)

    # ── 5. NVT Proxy z-score (NEW) ────────────────────────────────────────────
    # NVT proxy = MarketCap / SpotVolume
    # Low NVT = network undervalued relative to usage → buy signal (inverted)
    if "CapMrktCurUSD" in df.columns and "volume_reported_spot_usd_1d" in df.columns:
        nvt_raw = df["CapMrktCurUSD"] / df["volume_reported_spot_usd_1d"]
        nvt_raw = nvt_raw.replace([np.inf, -np.inf], np.nan)
        # 30-day smoothing to reduce daily noise
        nvt_ma = (
            nvt_raw.rolling(30, min_periods=5)
            .mean()
            .interpolate(method="linear")
            .fillna(nvt_raw.median())
        )
        # Z-score then invert: low NVT → positive signal
        nvt_z = -_rolling_zscore(nvt_ma, window=365, min_periods=60, clip=3.0)
    else:
        nvt_z = pd.Series(0.0, index=df.index)

    # ── Signals → (0,1) via sigmoid ──────────────────────────────────────────
    sig_mvrv = pd.Series(_sigmoid(np.asarray(mvrv_z.values) * -1), index=df.index)
    sig_ma200 = pd.Series(_sigmoid(np.asarray(ma200_dev.values) * -3), index=df.index)
    sig_halving = halving_prox
    sig_flow = pd.Series(_sigmoid(np.asarray(flow_z.values)), index=df.index)
    sig_nvt = pd.Series(_sigmoid(np.asarray(nvt_z.values)), index=df.index)

    # ── Weighted composite ────────────────────────────────────────────────────
    composite = (
        W_MVRV * sig_mvrv
        + W_MA200 * sig_ma200
        + W_HALVING * sig_halving
        + W_FLOW * sig_flow
        + W_NVT * sig_nvt
    )

    feat = pd.DataFrame(
        {
            "mvrv_z": mvrv_z,
            "ma200_dev": ma200_dev,
            "halving_prox": halving_prox,
            "flow_z": flow_z,
            "nvt_z": nvt_z,
            "sig_mvrv": sig_mvrv,
            "sig_ma200": sig_ma200,
            "sig_halving": sig_halving,
            "sig_flow": sig_flow,
            "sig_nvt": sig_nvt,
            "composite": composite,
        },
        index=df.index,
    )

    # ── 1-day lag: look-ahead bias prevention ───────────────────────────────────────
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

    raw_w = np.exp(composite * EXP_STRENGTH)
    raw_w = np.clip(raw_w, MIN_W, None)

    if locked_weights is not None:
        n_past = len(locked_weights)
    else:
        past_end = min(current_date, end_date)
        n_past = min(len(pd.date_range(start_date, past_end, freq="D")), len(full_range))

    weights = allocate_sequential_stable(raw_w, n_past, locked_weights)
    weights = np.clip(weights, MIN_W, None)
    weights = weights / weights.sum()
    return pd.Series(weights, index=full_range, name="weight")
