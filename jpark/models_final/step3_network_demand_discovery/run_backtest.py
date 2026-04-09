"""Run backtest for Step 3: Network Demand Discovery

Usage:
    python -m jpark.models_final.step3_network_demand_discovery.run_backtest
"""

import logging
from pathlib import Path

import pandas as pd
from template.prelude_template import load_data
from template.backtest_template import run_full_analysis
from jpark.models_final.step3_network_demand_discovery.model3 import (
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
    logging.info("=== Step 3: Network Demand Discovery ===")
    logging.info("Testing AdrActCnt (Active Address Count) signal")
    btc_df = load_data()
    _FEATURES_DF = precompute_features(btc_df)
    output_dir = Path(__file__).parent / "output"
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="Step3: Network Demand (AdrActCnt W=0.20)",
    )


if __name__ == "__main__":
    main()
