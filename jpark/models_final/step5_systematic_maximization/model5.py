"""
Model — Minimal Regime Split: MVRV-only Bear/Bull Branch


### Core Hypothesis
  The primary cause of exp loss during bull markets = incorrect MVRV signaling.

  MVRV mechanism:
    sig_mvrv = 1 - percentile_rank(MVRV)

    → Bull market: MVRV high → sig_mvrv low → allocation reduced → exp loss
    → Bear market: MVRV low → sig_mvrv high → allocation increased → works well

  Solution:
    w_mvrv_bear (higher) vs w_mvrv_bull (lower)

    → Suppress MVRV influence during bull markets

### Change
  v22 had a single parameter: w_mvrv

  v25 replaces it with:
    w_mvrv_bear + w_mvrv_bull + r_steepness + r_threshold

  Net increase: +3 parameters (13 → 16)

  Compared to v24 (22 parameters):
    -6 parameters → significantly more efficient search space

### Parameters (16 total)

  Regime:
    r_steepness, r_threshold

  Branch:
    w_mvrv_bear, w_mvrv_bull

  Shared:
    w_halving, w_flow, w_addr, w_ex, base_w,
    steepness, threshold,
    c30, c90, c180, c365, c1461

### Warm-start

  bear = v22 best (w_mvrv = 0.1019)

  bull = 0.03
    → MVRV influence nearly ignored during bull markets

  regime boundary:
    r_steepness = 3.0
    r_threshold = 0.3

    (z1461 > 0.3 = bull market)
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


# ── Regime parameters ──────────────────────────────
R_STEEPNESS = 3.0
R_THRESHOLD = 0.3  # z1461 > 0.3 = bull

W_MVRV_BEAR = 0.1019  # v22 best
W_MVRV_BULL = 0.03  # MVRV influence nearly ignored during bull markets

# Shared parameters (v22 best)
W_HALVING = 0.2150
W_FLOW = 0.4011
W_ADDR = 0.3673
W_EX = 0.3454
BASE_W = 0.4949
STEEPNESS = 364.90
THRESHOLD = 0.3610

C30 = 2.9506
C90 = 0.8114
C180 = 6.9989
C365 = 8.9234
C1461 = 4.4286

MIN_WINDOW = 30
PCT_WINDOW = 365


def _zscore(series: pd.Series, window: int, min_periods: int = 30) -> pd.Series:
    m = series.rolling(window, min_periods=min_periods).mean()
    s = series.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    return ((series - m) / s).fillna(0.0).clip(-3, 3)


def _rolling_pct_rank(series: pd.Series, window: int, min_periods: int = 30) -> pd.Series:
    return series.rolling(window, min_periods=min_periods).rank(pct=True).fillna(0.5)


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    price = df[PRICE_COL].interpolate(method="linear").bfill()

    # 1. MVRV
    if "CapMVRVCur" in df.columns:
        mvrv_raw = df["CapMVRVCur"].interpolate(method="linear").bfill()
    else:
        mvrv_raw = pd.Series(1.0, index=df.index)
    sig_mvrv = 1 - _rolling_pct_rank(mvrv_raw, PCT_WINDOW, MIN_WINDOW)

    # 2. Halving
    sig_halving = pd.Series(_halving_proximity(pd.DatetimeIndex(df.index)), index=df.index)

    # 3. Flow
    if "FlowInExNtv" in df.columns and "FlowOutExNtv" in df.columns:
        net_flow = df["FlowInExNtv"].fillna(0.0) - df["FlowOutExNtv"].fillna(0.0)
        net_flow_ma = net_flow.rolling(30, min_periods=5).mean()
        sig_net = 1 - _rolling_pct_rank(net_flow_ma, PCT_WINDOW, MIN_WINDOW)
    else:
        sig_net = pd.Series(0.5, index=df.index)

    if "SplyExNtv" in df.columns:
        sply_ex_abs = df["SplyExNtv"].interpolate(method="linear").bfill()
        sply_change = sply_ex_abs.pct_change(30).fillna(0.0).rolling(7, min_periods=3).mean()
        sig_sply = 1 - _rolling_pct_rank(sply_change, PCT_WINDOW, MIN_WINDOW)
        sig_flow = 0.5 * sig_net + 0.5 * sig_sply
    else:
        sig_flow = sig_net

    # 4. Active Address
    if "AdrActCnt" in df.columns:
        addr_raw = df["AdrActCnt"].interpolate(method="linear").bfill()
    else:
        addr_raw = pd.Series(0.5, index=df.index)
    sig_addr = _rolling_pct_rank(addr_raw, PCT_WINDOW, MIN_WINDOW)

    # 5. Exchange Supply Ratio
    if "SplyExNtv" in df.columns and "SplyCur" in df.columns:
        sply_cur = df["SplyCur"].replace(0, np.nan).interpolate(method="linear").bfill()
        sply_ex = df["SplyExNtv"].interpolate(method="linear").bfill()
        ex_ratio = (
            (sply_ex / sply_cur)
            .replace([np.inf, -np.inf], np.nan)
            .interpolate(method="linear")
            .bfill()
        )
        ex_chg = ex_ratio.pct_change(30).fillna(0.0).rolling(7, min_periods=3).mean()
        sig_ex = 1 - _rolling_pct_rank(ex_chg, PCT_WINDOW, MIN_WINDOW)
    else:
        sig_ex = pd.Series(0.5, index=df.index)

    # z-score
    log_price = pd.Series(np.log(price.clip(lower=1e-9)), index=price.index)
    z30 = _zscore(log_price, 30, min_periods=15)
    z90 = _zscore(log_price, 90, min_periods=30)
    z180 = _zscore(log_price, 180, min_periods=60)
    z365 = _zscore(log_price, 365, min_periods=120)
    z1461 = _zscore(log_price, 1461, min_periods=365)

    feat = pd.DataFrame(
        {
            "sig_mvrv": sig_mvrv,
            "sig_halving": sig_halving,
            "sig_flow": sig_flow,
            "sig_addr": sig_addr,
            "sig_ex": sig_ex,
            "z30": z30,
            "z90": z90,
            "z180": z180,
            "z365": z365,
            "z1461": z1461,
            "z1461_raw": z1461,  # For regime detection only (same values, separate column)
        },
        index=df.index,
    )

    feat = feat.shift(1).fillna(0.0)
    for col in ["sig_mvrv", "sig_halving", "sig_flow", "sig_addr", "sig_ex"]:
        feat[col] = feat[col].replace(0.0, 0.5)

    return feat


def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
    *,
    # Regime detection
    r_steepness: float = R_STEEPNESS,
    r_threshold: float = R_THRESHOLD,
    # MVRV bear/bull branch
    w_mvrv_bear: float = W_MVRV_BEAR,
    w_mvrv_bull: float = W_MVRV_BULL,
    # Shared parameters
    w_halving: float = W_HALVING,
    w_flow: float = W_FLOW,
    w_addr: float = W_ADDR,
    w_ex: float = W_EX,
    base_w: float = BASE_W,
    steepness: float = STEEPNESS,
    threshold: float = THRESHOLD,
    c30: float = C30,
    c90: float = C90,
    c180: float = C180,
    c365: float = C365,
    c1461: float = C1461,
) -> pd.Series:
    """
    v25: w_mvrv only bear/bull branch, rest same as v22.

    regime = sigmoid(r_steepness × (z1461 - r_threshold))
    w_mvrv_eff = w_mvrv_bear × (1-regime) + w_mvrv_bull × regime
    """
    full_range = pd.date_range(start_date, end_date, freq="D")
    df_re = features_df.reindex(full_range)

    def _get(col: str) -> np.ndarray:
        return np.asarray(df_re[col].fillna(0.0).values)

    # ── 1. Regime detection ──────────────────────────────────────────────────
    z1461_arr = _get("z1461_raw")
    regime_logit = np.clip(r_steepness * (z1461_arr - r_threshold), -500, 500)
    regime = 1.0 / (1.0 + np.exp(-regime_logit))  # ∈ [0,1]

    # ── 2. w_mvrv regime interpolation ──────────────────────────────────────────
    w_mvrv_eff = w_mvrv_bear * (1 - regime) + w_mvrv_bull * regime

    # ── 3. On-chain composite ─────────────────────────────────────────
    total_w = w_mvrv_eff + w_halving + w_flow + w_addr + w_ex
    total_w = np.where(total_w < 1e-9, 1.0, total_w)

    onchain = (
        (w_mvrv_eff / total_w) * _get("sig_mvrv")
        + (w_halving / total_w) * _get("sig_halving")
        + (w_flow / total_w) * _get("sig_flow")
        + (w_addr / total_w) * _get("sig_addr")
        + (w_ex / total_w) * _get("sig_ex")
    )

    # ── 4. Z-score composite (v22 same) ───────────────────────────────
    z_combined = -(
        c30 * _get("z30")
        + c90 * _get("z90")
        + c180 * _get("z180")
        + c365 * _get("z365")
        + c1461 * _get("z1461")
    )
    zscore_sig = 1.0 / (1.0 + np.exp(-np.clip(z_combined, -500, 500)))

    # ── 5. Final composite ────────────────────────────────────────────
    zscore_w = 1.0 - base_w
    composite = base_w * onchain + zscore_w * zscore_sig
    composite = _clean_array(composite)

    # ── 6. v22 sigmoid → raw_w ───────────────────────────────────────
    exp_input = np.clip(steepness * (composite - threshold), -700, 700)
    raw_w = np.clip(np.exp(exp_input), MIN_W, None)

    if locked_weights is not None:
        n_past = len(locked_weights)
    else:
        past_end = min(current_date, end_date)
        n_past = min(len(pd.date_range(start_date, past_end, freq="D")), len(full_range))

    weights = allocate_sequential_stable(raw_w, n_past, locked_weights)
    weights = np.clip(weights, MIN_W, None)
    weights = weights / weights.sum()
    return pd.Series(weights, index=full_range, name="weight")
