"""Step 6: Softmax Allocation — run backtest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from template.prelude_template import load_data, backtest_dynamic_dca
from template.backtest_template import run_full_analysis
from jpark.models_final.step6_softmax_allocation.model6 import (
    precompute_features,
    compute_window_weights,
)

# ── Optuna Trial 910 best params (score=98.085%) ─────────────────────────────
BEST_PARAMS = {
    "r_steepness": 13.2944,
    "r_threshold": 1.2398,
    "w_mvrv_bear": 0.02508,
    "w_mvrv_bull": 0.18375,
    "w_halving": 0.05815,
    "w_flow": 0.19255,
    "w_addr": 0.74242,
    "w_ex": 0.04479,
    "base_w": 0.70049,
    "steepness": 457.37,
    "c30": 1.8705,
    "c90": 3.2714,
    "c180": 10.5401,
    "c365": 9.3312,
    "c1461": 7.8503,
}


def main():
    btc_df = load_data()
    features_df = precompute_features(btc_df)

    def compute_weights_wrapper(df_window):
        sd = df_window.index.min()
        ed = df_window.index.max()
        return compute_window_weights(features_df, sd, ed, ed, None, **BEST_PARAMS)

    output_dir = Path(__file__).parent / "output"

    run_full_analysis(
        btc_df,
        features_df,
        compute_weights_wrapper,
        output_dir,
        strategy_label="Step6-Softmax",
    )


if __name__ == "__main__":
    main()
