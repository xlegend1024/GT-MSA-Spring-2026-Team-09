"""
Run backtest for Step 1: Baseline Foundation

Usage:
    python -m jpark.models_final.step1_baseline_foundation.run_backtest
"""

import logging
from pathlib import Path

import pandas as pd

from template.prelude_template import load_data
from template.backtest_template import run_full_analysis
from jpark.models_final.step1_baseline_foundation.model1 import (
    precompute_features,
    compute_window_weights,
)

# Global precomputed features
_FEATURES_DF = None


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

    return compute_window_weights(_FEATURES_DF, start_date, end_date, current_date)


def main():
    global _FEATURES_DF

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=== Step 1: Baseline Foundation ===")
    logging.info("Signals: MVRV Z-score | MA200 log dev | Halving proximity | Net flow")

    # 1. Load data
    btc_df = load_data()

    # 2. Precompute features
    logging.info("Precomputing features...")
    _FEATURES_DF = precompute_features(btc_df)
    logging.info(f"Features computed: {_FEATURES_DF.shape[0]} rows")

    # 3. Output directory
    output_dir = Path(__file__).parent / "output"

    # 4. Run full backtest & generate artifacts
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="Step1: Baseline (MVRV+MA200+Halving+Flow)",
    )


if __name__ == "__main__":
    main()
