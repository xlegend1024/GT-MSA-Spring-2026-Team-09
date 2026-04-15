"""V2 Model — Walk-Forward Optimized DCA Strategy.

Extends V1 base signals with:
1. Scipy.optimize SLSQP for signal weight optimization
2. Walk-forward validation (train on past, test on future)
3. Nonlinear MVRV boost at extremes
4. DYNAMIC_STRENGTH as optimizable parameter

Optimization variables (7):
- 5 signal weights (sum=1): W_MVRV, W_MA, W_FLOW, W_HALVING, W_CONFIDENCE
- DYNAMIC_STRENGTH: [2.0, 7.0]
- MVRV_ACCUM threshold: [0.7, 1.3]
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as spo

from template.model_development_template import (
    _compute_stable_signal,
    allocate_sequential_stable,
    _clean_array,
)
from template.prelude_template import BACKTEST_START, BACKTEST_END, WINDOW_OFFSET
from hshin.model.v1.model_v1 import load_poly_indices, _poly_change_signal

W_POLY = 0.15

PRICE_COL = "PriceUSD_coinmetrics"
MVRV_COL = "CapMVRVCur"
FLOW_IN_COL = "FlowInExNtv"
FLOW_OUT_COL = "FlowOutExNtv"

MIN_W = 1e-6
MA_WINDOW = 200

HALVING_DATES = pd.to_datetime([
    "2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20",
])

# Default params (V1 baseline — used before optimization)
DEFAULT_PARAMS = {
    "w_mvrv": 0.35,
    "w_ma": 0.20,
    "w_flow": 0.15,
    "w_halving": 0.10,
    "w_confidence": 0.20,
    "dynamic_strength": 4.0,
    "mvrv_accum": 1.0,
}

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


# =============================================================================
# Feature Engineering (reuse V1 logic, parameterized thresholds)
# =============================================================================

def _halving_proximity(dates: pd.DatetimeIndex) -> pd.Series:
    signal = pd.Series(0.0, index=dates)
    for hd in HALVING_DATES:
        pre_start = hd - pd.DateOffset(months=6)
        pre_end = hd - pd.DateOffset(months=2)
        mask = (dates >= pre_start) & (dates <= pre_end)
        if mask.any():
            days_to_halving = (hd - dates[mask]).days
            max_days = (hd - pre_start).days
            signal[mask] = np.clip(1.0 - days_to_halving / max_days, 0, 1)
    return signal


def _sustained_inflow(flow_in: pd.Series, flow_out: pd.Series, threshold: int = 5) -> pd.Series:
    net_flow = flow_in - flow_out
    is_inflow = (net_flow > 0).astype(int)
    groups = (is_inflow != is_inflow.shift()).cumsum()
    run_length = is_inflow.groupby(groups).cumsum()
    return (run_length >= threshold).astype(float)


def _confidence_signal(mvrv_z: pd.Series, price_vs_ma: pd.Series) -> pd.Series:
    z_signal = (-mvrv_z / 4).clip(-1, 1)
    ma_signal = (-price_vs_ma).clip(-1, 1)
    agree = ((z_signal > 0) & (ma_signal > 0)).astype(float) - \
            ((z_signal < 0) & (ma_signal < 0)).astype(float)
    strength = (z_signal.abs() + ma_signal.abs()) / 2
    return (agree * strength).fillna(0)


def precompute_features(df: pd.DataFrame, use_poly: bool = False) -> pd.DataFrame:
    """Compute base features (same as V1 base). Thresholds applied at weight time."""
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found.")

    price = df[PRICE_COL].loc["2010-07-18":].copy()
    ma200 = price.rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        price_vs_ma = ((price / ma200) - 1).clip(-1, 1).fillna(0)

    if MVRV_COL in df.columns:
        mvrv = df[MVRV_COL].loc[price.index]
        mvrv_raw = mvrv.copy()
        mvrv_mean = mvrv.rolling(365, min_periods=180).mean()
        mvrv_std = mvrv.rolling(365, min_periods=180).std()
        mvrv_z = ((mvrv - mvrv_mean) / mvrv_std).clip(-4, 4).fillna(0)
    else:
        mvrv_raw = pd.Series(1.5, index=price.index)
        mvrv_z = pd.Series(0.0, index=price.index)

    if FLOW_IN_COL in df.columns and FLOW_OUT_COL in df.columns:
        flow_in = df[FLOW_IN_COL].loc[price.index].fillna(0)
        flow_out = df[FLOW_OUT_COL].loc[price.index].fillna(0)
        sustained = _sustained_inflow(flow_in, flow_out)
        net_flow = flow_in - flow_out
        is_outflow = (net_flow < 0).astype(int)
        out_groups = (is_outflow != is_outflow.shift()).cumsum()
        out_run = is_outflow.groupby(out_groups).cumsum()
        sustained_outflow = (out_run >= 5).astype(float)
        flow_signal = (-sustained + sustained_outflow * 0.5).clip(-1, 1)
    else:
        flow_signal = pd.Series(0.0, index=price.index)

    halving_signal = _halving_proximity(price.index)
    confidence = _confidence_signal(mvrv_z, price_vs_ma)

    features = pd.DataFrame({
        PRICE_COL: price,
        "price_ma200": ma200,
        "price_vs_ma": price_vs_ma,
        "mvrv_z": mvrv_z,
        "mvrv_raw": mvrv_raw,
        "mvrv_raw_lagged": mvrv_raw.shift(1).fillna(1.5),
        "flow_signal": flow_signal,
        "halving_signal": halving_signal,
        "confidence": confidence,
    }, index=price.index)

    if use_poly:
        poly_indices = load_poly_indices()
        crypto_z = _poly_change_signal(poly_indices["crypto"])
        features["poly_crypto"] = crypto_z.reindex(price.index, fill_value=0)
        us_z = _poly_change_signal(poly_indices["us_affairs"])
        features["poly_us_affairs"] = us_z.reindex(price.index, fill_value=0)

    signal_cols = [c for c in features.columns if c not in [PRICE_COL, "price_ma200", "mvrv_raw", "mvrv_raw_lagged"]]
    features[signal_cols] = features[signal_cols].shift(1)
    features = features.fillna(0)
    return features


# =============================================================================
# Parameterized Dynamic Multiplier
# =============================================================================

def compute_dynamic_multiplier(features: pd.DataFrame, params: dict) -> np.ndarray:
    """Compute multiplier with optimizable params."""
    price_vs_ma = _clean_array(features["price_vs_ma"].values)
    mvrv_z = _clean_array(features["mvrv_z"].values)
    mvrv_raw = _clean_array(features["mvrv_raw_lagged"].values)
    flow_signal = _clean_array(features["flow_signal"].values)
    halving_signal = _clean_array(features["halving_signal"].values)
    confidence = _clean_array(features["confidence"].values)

    w_mvrv = params["w_mvrv"]
    w_ma = params["w_ma"]
    w_flow = params["w_flow"]
    w_halving = params["w_halving"]
    w_confidence = params["w_confidence"]
    dyn_str = params["dynamic_strength"]
    mvrv_accum = params["mvrv_accum"]

    # MVRV signal with nonlinear boost at extremes
    mvrv_zone = np.zeros_like(mvrv_raw)
    mvrv_zone[mvrv_raw < mvrv_accum] = 1.0 + 0.5 * (mvrv_accum - mvrv_raw[mvrv_raw < mvrv_accum]) ** 2
    mvrv_zone[mvrv_raw > 3.5] = -1.0 - 0.3 * (mvrv_raw[mvrv_raw > 3.5] - 3.5) ** 2
    mvrv_signal = -mvrv_z * 0.5 + np.clip(mvrv_zone, -2, 2) * 0.5

    ma_signal = -price_vs_ma
    flow_sig = flow_signal
    halving_sig = halving_signal
    conf_sig = confidence

    has_poly = "poly_crypto" in features.columns or "poly_us_affairs" in features.columns
    if has_poly:
        scale = 1.0 - W_POLY
        combined = (
            mvrv_signal * w_mvrv * scale
            + ma_signal * w_ma * scale
            + flow_sig * w_flow * scale
            + halving_sig * w_halving * scale
            + conf_sig * w_confidence * scale
        )
        poly_combined = np.zeros(len(features))
        poly_count = 0
        if "poly_crypto" in features.columns:
            poly_combined += _clean_array(features["poly_crypto"].values)
            poly_count += 1
        if "poly_us_affairs" in features.columns:
            poly_combined += _clean_array(features["poly_us_affairs"].values)
            poly_count += 1
        if poly_count > 0:
            poly_combined /= poly_count
        combined += poly_combined * W_POLY
    else:
        combined = (
            mvrv_signal * w_mvrv
            + ma_signal * w_ma
            + flow_sig * w_flow
            + halving_sig * w_halving
            + conf_sig * w_confidence
        )

    adjustment = combined * dyn_str
    adjustment = np.clip(adjustment, -5, 100)
    multiplier = np.exp(adjustment)
    return np.where(np.isfinite(multiplier), multiplier, 1.0)


# =============================================================================
# Weight Computation
# =============================================================================

def compute_weights_fast(features_df, start_date, end_date, params, n_past=None):
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)
    n = len(df)
    base = np.ones(n) / n
    dyn = compute_dynamic_multiplier(df, params)
    raw = base * dyn
    if n_past is None:
        n_past = n
    weights = allocate_sequential_stable(raw, n_past)
    return pd.Series(weights, index=df.index)


def compute_window_weights(features_df, start_date, end_date, current_date, params):
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")
    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame({col: 0.0 for col in features_df.columns}, index=missing)
        features_df = pd.concat([features_df, placeholder]).sort_index()
    past_end = min(current_date, end_date)
    n_past = len(pd.date_range(start=start_date, end=past_end, freq="D")) if start_date <= past_end else 0
    weights = compute_weights_fast(features_df, start_date, end_date, params, n_past)
    return weights.reindex(full_range, fill_value=0.0)


# =============================================================================
# Walk-Forward Optimization
# =============================================================================

def _params_from_vector(x: np.ndarray) -> dict:
    """Convert optimization vector to params dict.
    x[0:5] = signal weights (will be normalized to sum=1)
    x[5] = dynamic_strength
    x[6] = mvrv_accum
    """
    raw_weights = np.abs(x[:5])
    w_sum = raw_weights.sum()
    if w_sum < 1e-10:
        raw_weights = np.ones(5) / 5
    else:
        raw_weights = raw_weights / w_sum
    return {
        "w_mvrv": float(raw_weights[0]),
        "w_ma": float(raw_weights[1]),
        "w_flow": float(raw_weights[2]),
        "w_halving": float(raw_weights[3]),
        "w_confidence": float(raw_weights[4]),
        "dynamic_strength": float(x[5]),
        "mvrv_accum": float(x[6]),
    }


def _vector_from_params(params: dict) -> np.ndarray:
    return np.array([
        params["w_mvrv"], params["w_ma"], params["w_flow"],
        params["w_halving"], params["w_confidence"],
        params["dynamic_strength"], params["mvrv_accum"],
    ])


def _evaluate_params_on_period(
    features_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    params: dict,
    period_start: str,
    period_end: str,
    sample_step: int = 7,
) -> float:
    """Evaluate params on a date range. Returns negative mean excess (for minimization).
    sample_step: evaluate every Nth window to speed up.
    """
    start = pd.to_datetime(period_start)
    end = pd.to_datetime(period_end)
    max_start = end - WINDOW_OFFSET
    start_dates = pd.date_range(start=start, end=max_start, freq="D")[::sample_step]

    excesses = []
    for ws in start_dates:
        we = ws + WINDOW_OFFSET
        price_slice = btc_df[PRICE_COL].loc[ws:we]
        if price_slice.empty or len(price_slice) < 120:
            continue

        weights = compute_weights_fast(features_df, ws, we, params)
        if weights.empty or not np.isclose(weights.sum(), 1.0, atol=1e-4):
            continue

        inv_price = 1e8 / price_slice
        uniform_spd = inv_price.mean()
        dynamic_spd = (weights * inv_price).sum()
        span = inv_price.max() - inv_price.min()
        if span > 0:
            excess = ((dynamic_spd - uniform_spd) / span) * 100
        else:
            excess = 0.0
        excesses.append(excess)

    if not excesses:
        return 0.0
    return -np.mean(excesses)  # Negative because we minimize


def optimize_params(
    features_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    train_start: str,
    train_end: str,
) -> dict:
    """Find optimal params on training period using SLSQP."""
    x0 = _vector_from_params(DEFAULT_PARAMS)

    bounds = [
        (0.05, 0.60),  # w_mvrv
        (0.05, 0.50),  # w_ma
        (0.05, 0.40),  # w_flow
        (0.02, 0.30),  # w_halving
        (0.05, 0.50),  # w_confidence
        (2.0, 7.0),    # dynamic_strength
        (0.7, 1.3),    # mvrv_accum
    ]

    constraints = [{
        "type": "eq",
        "fun": lambda x: np.sum(x[:5]) - 1.0,
    }]

    call_count = [0]

    def objective(x):
        call_count[0] += 1
        params = _params_from_vector(x)
        score = _evaluate_params_on_period(
            features_df, btc_df, params, train_start, train_end, sample_step=14
        )
        if call_count[0] % 10 == 0:
            logging.info(f"  opt iter {call_count[0]}: obj={-score:.4f}")
        return score

    result = spo.minimize(
        objective, x0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 100, "ftol": 1e-6},
    )

    best_params = _params_from_vector(result.x)
    logging.info(f"  Optimization: {call_count[0]} evals, success={result.success}")
    logging.info(f"  Best params: {best_params}")
    return best_params


def walk_forward_optimize(
    features_df: pd.DataFrame,
    btc_df: pd.DataFrame,
) -> list[dict]:
    """Walk-forward optimization with expanding training window.

    Folds:
      Train: 2018-01-01 → 2021-12-31, Test: 2022-01-01 → 2022-12-31
      Train: 2018-01-01 → 2022-12-31, Test: 2023-01-01 → 2023-12-31
      Train: 2018-01-01 → 2023-12-31, Test: 2024-01-01 → 2024-12-31
      Train: 2018-01-01 → 2024-12-31, Test: 2025-01-01 → 2025-12-31
    """
    folds = [
        ("2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        ("2018-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ]

    results = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        logging.info(f"Fold {i+1}: Train [{tr_s} → {tr_e}], Test [{te_s} → {te_e}]")

        best_params = optimize_params(features_df, btc_df, tr_s, tr_e)
        train_score = -_evaluate_params_on_period(features_df, btc_df, best_params, tr_s, tr_e, sample_step=7)
        test_score = -_evaluate_params_on_period(features_df, btc_df, best_params, te_s, te_e, sample_step=1)

        # Compare with V1 default on same test period
        default_test = -_evaluate_params_on_period(features_df, btc_df, DEFAULT_PARAMS, te_s, te_e, sample_step=1)

        results.append({
            "fold": i + 1,
            "train_period": f"{tr_s} → {tr_e}",
            "test_period": f"{te_s} → {te_e}",
            "optimized_params": best_params,
            "train_mean_excess": round(train_score, 4),
            "test_mean_excess": round(test_score, 4),
            "default_test_mean_excess": round(default_test, 4),
            "improvement_vs_default": round(test_score - default_test, 4),
        })

        logging.info(f"  Train excess: {train_score:.4f}, Test excess: {test_score:.4f} (default: {default_test:.4f})")

    return results
