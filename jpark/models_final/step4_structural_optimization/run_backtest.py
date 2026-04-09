"""Run backtest for Step 4: Structural Optimization

Usage:
    python -m jpark.models_final.step4_structural_optimization.run_backtest
"""

import logging
from pathlib import Path

import pandas as pd
from template.prelude_template import load_data
from template.backtest_template import run_full_analysis
from jpark.models_final.step4_structural_optimization.model4 import (
    precompute_features,
    compute_window_weights,
)

_FEATURES_DF = None


def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    global _FEATURES_DF
    if _FEATURES_DF is None:
        raise ValueError("Features not precomputed.")
    start_date = df_window.index.min()
    end_date = df_window.index.max()
    return compute_window_weights(_FEATURES_DF, start_date, end_date, end_date)


def main():
    global _FEATURES_DF
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    logging.info("=== Step 4: Structural Optimization ===")
    logging.info("Grid Search B: MA200 removed (r=0.87 with MVRV), Flow enhanced")
    btc_df = load_data()
    _FEATURES_DF = precompute_features(btc_df)

    output_dir = Path(__file__).parent / "output"
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="Step4: Structural Optimization (4 signals, STR=175)",
    )


if __name__ == "__main__":
    main()
