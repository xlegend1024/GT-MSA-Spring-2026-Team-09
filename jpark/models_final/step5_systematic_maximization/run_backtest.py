"""
Run backtest for Step 5: Systematic Maximization

This uses the final optimized parameters from Optuna trial 431 (1000 trials total).

Usage:
    python -m jpark.models_final.step5_systematic_maximization.run_backtest
"""

import logging
from pathlib import Path
import json

import pandas as pd

from template.prelude_template import load_data
from template.backtest_template import run_full_analysis
from jpark.models_final.step5_systematic_maximization.model5 import (
    precompute_features,
    compute_window_weights,
)

# Global precomputed features
_FEATURES_DF = None

# Best parameters from v27 trial 431 (B-soft=88.671, Score=89.17%)
BEST_PARAMS = {
    "r_steepness": 5.477,
    "r_threshold": 0.754,
    "w_mvrv_bear": 0.1754,
    "w_mvrv_bull": 0.0055,
    "w_halving": 0.0538,
    "w_flow": 0.474,
    "w_addr": 0.354,
    "w_ex": 0.192,
    "base_w": 0.4325,
    "steepness": 709.52,
    "threshold": 0.2859,
    "c30": 2.228,
    "c90": 1.045,
    "c180": 8.112,
    "c365": 10.052,
    "c1461": 4.001,
}


def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    """Wrapper adapting compute_window_weights to backtest engine interface."""
    global _FEATURES_DF

    if _FEATURES_DF is None:
        raise ValueError("Features not precomputed. Call precompute_features() first.")

    if df_window.empty:
        return pd.Series(dtype=float)

    start_date = df_window.index.min()
    end_date = df_window.index.max()
    current_date = end_date  # Backtest mode

    return compute_window_weights(
        _FEATURES_DF,
        start_date,
        end_date,
        current_date,
        locked_weights=None,
        r_steepness=BEST_PARAMS["r_steepness"],
        r_threshold=BEST_PARAMS["r_threshold"],
        w_mvrv_bear=BEST_PARAMS["w_mvrv_bear"],
        w_mvrv_bull=BEST_PARAMS["w_mvrv_bull"],
        w_halving=BEST_PARAMS["w_halving"],
        w_flow=BEST_PARAMS["w_flow"],
        w_addr=BEST_PARAMS["w_addr"],
        w_ex=BEST_PARAMS["w_ex"],
        base_w=BEST_PARAMS["base_w"],
        steepness=BEST_PARAMS["steepness"],
        threshold=BEST_PARAMS["threshold"],
        c30=BEST_PARAMS["c30"],
        c90=BEST_PARAMS["c90"],
        c180=BEST_PARAMS["c180"],
        c365=BEST_PARAMS["c365"],
        c1461=BEST_PARAMS["c1461"],
    )


def main():
    global _FEATURES_DF

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=== Step 5: Systematic Maximization (v27) ===")
    logging.info("Optuna 1000 trials, MVRV regime-aware, multi-timescale z-scores")
    logging.info(f"Using best params from trial 431:")
    for k, v in BEST_PARAMS.items():
        logging.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 1. Load data
    btc_df = load_data()

    # 2. Precompute features
    logging.info("Precomputing features...")
    _FEATURES_DF = precompute_features(btc_df)
    logging.info(f"Features computed: {_FEATURES_DF.shape[0]} rows")

    # 3. Output directory
    output_dir = Path(__file__).parent / "output"

    # 4. Save best params
    params_file = output_dir / "best_params_v27.json"
    params_file.parent.mkdir(parents=True, exist_ok=True)
    with open(params_file, "w") as f:
        json.dump(BEST_PARAMS, f, indent=2)
    logging.info(f"Best params saved to {params_file}")

    # 5. Run full backtest & generate artifacts
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="Step5: Systematic Maximization (v27, Trial 431)",
    )


if __name__ == "__main__":
    main()
