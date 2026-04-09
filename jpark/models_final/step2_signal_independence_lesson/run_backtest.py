"""Run backtest for Step 2: Signal Independence Lesson

Usage:
    python -m jpark.models_final.step2_signal_independence_lesson.run_backtest
"""

import logging
from pathlib import Path

import pandas as pd
from template.prelude_template import load_data
from template.backtest_template import run_full_analysis
from jpark.models_final.step2_signal_independence_lesson.model2 import (
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
    logging.info("=== Step 2: Signal Independence Lesson ===")
    logging.info("Testing NVT signal addition (expected to fail)")
    btc_df = load_data()
    _FEATURES_DF = precompute_features(btc_df)
    logging.info(f"Features: {list(_FEATURES_DF.columns)}")
    output_dir = Path(__file__).parent / "output"
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="Step2: NVT Lesson (MVRV+MA200+Halving+Flow+NVT)",
    )


if __name__ == "__main__":
    main()
