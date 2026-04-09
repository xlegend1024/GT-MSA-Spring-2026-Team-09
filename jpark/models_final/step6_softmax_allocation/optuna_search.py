"""Step 6: Optuna TPE optimization for softmax allocation model.

Optimizes all 15 parameters of the softmax allocation model.
Key parameter: steepness controls weight concentration in softmax.

Usage:
    python -m jpark.models_final.step6_softmax_allocation.optuna_search
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from template.prelude_template import load_data, backtest_dynamic_dca
from jpark.models_final.step6_softmax_allocation.model6 import (
    precompute_features,
    compute_window_weights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

# ── Data (loaded once) ────────────────────────────────────────────────────────
btc_df = load_data()
features_df = precompute_features(btc_df)

# Output paths
OUTPUT_DIR = Path(__file__).parent / "output" / "optuna"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = OUTPUT_DIR / "step6_tpe.db"


def objective(trial: optuna.Trial) -> float:
    """Optimize all 15 parameters for the softmax allocation model."""
    params = {
        # Regime detection
        "r_steepness": trial.suggest_float("r_steepness", 1.0, 15.0),
        "r_threshold": trial.suggest_float("r_threshold", -0.5, 1.5),
        # MVRV bear/bull split
        "w_mvrv_bear": trial.suggest_float("w_mvrv_bear", 0.0, 0.5),
        "w_mvrv_bull": trial.suggest_float("w_mvrv_bull", 0.0, 0.2),
        # Signal weights
        "w_halving": trial.suggest_float("w_halving", 0.0, 0.5),
        "w_flow": trial.suggest_float("w_flow", 0.0, 1.0),
        "w_addr": trial.suggest_float("w_addr", 0.0, 1.0),
        "w_ex": trial.suggest_float("w_ex", 0.0, 0.5),
        # Blend
        "base_w": trial.suggest_float("base_w", 0.1, 0.9),
        # SOFTMAX steepness (key parameter)
        "steepness": trial.suggest_float("steepness", 100.0, 500.0),
        # Z-score coefficients
        "c30": trial.suggest_float("c30", 0.0, 10.0),
        "c90": trial.suggest_float("c90", 0.0, 10.0),
        "c180": trial.suggest_float("c180", 0.0, 15.0),
        "c365": trial.suggest_float("c365", 0.0, 15.0),
        "c1461": trial.suggest_float("c1461", 0.0, 15.0),
    }

    def wrapper(df_window):
        sd = df_window.index.min()
        ed = df_window.index.max()
        return compute_window_weights(features_df, sd, ed, ed, None, **params)

    try:
        spd_df, exp_avg = backtest_dynamic_dca(
            btc_df,
            wrapper,
            features_df=features_df,
            strategy_label=f"step6-tpe-t{trial.number}",
        )
    except Exception as e:
        logging.warning(f"Trial {trial.number} failed: {e}")
        return 0.0

    win_rate = (spd_df["dynamic_percentile"] > spd_df["uniform_percentile"]).mean() * 100
    score = 0.5 * win_rate + 0.5 * exp_avg

    logging.info(
        f"trial-{trial.number:>3d} | win={win_rate:.2f}% | exp={exp_avg:.2f}% | "
        f"score={score:.2f}% | steep={params['steepness']:.1f}"
    )

    return score


def main():
    N_TRIALS = 1000

    study = optuna.create_study(
        study_name="step6_tpe",
        direction="maximize",
        storage=f"sqlite:///{DB_PATH}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=30),
    )

    # Warm-start with grid-search best (steepness=320, Step 5 best params)
    study.enqueue_trial(
        {
            "r_steepness": 5.477,
            "r_threshold": 0.754,
            "w_mvrv_bear": 0.1754,
            "w_mvrv_bull": 0.0055,
            "w_halving": 0.0538,
            "w_flow": 0.474,
            "w_addr": 0.354,
            "w_ex": 0.192,
            "base_w": 0.4325,
            "steepness": 320.0,
            "c30": 2.228,
            "c90": 1.045,
            "c180": 8.112,
            "c365": 10.052,
            "c1461": 4.001,
        }
    )

    # Second warm-start: slightly different steepness
    study.enqueue_trial(
        {
            "r_steepness": 5.477,
            "r_threshold": 0.754,
            "w_mvrv_bear": 0.1754,
            "w_mvrv_bull": 0.0055,
            "w_halving": 0.0538,
            "w_flow": 0.474,
            "w_addr": 0.354,
            "w_ex": 0.192,
            "base_w": 0.4325,
            "steepness": 280.0,
            "c30": 2.228,
            "c90": 1.045,
            "c180": 8.112,
            "c365": 10.052,
            "c1461": 4.001,
        }
    )

    completed = len(study.trials)
    remaining = max(0, N_TRIALS - completed)
    if remaining > 0:
        logging.info(
            f"Starting optimization: {remaining} trials remaining "
            f"({completed} already completed)"
        )
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)
    else:
        logging.info(f"Already completed {completed} trials.")

    # Report results
    best = study.best_trial
    logging.info(f"\n{'='*60}")
    logging.info(f"BEST: Trial {best.number} | Score = {best.value:.4f}%")
    logging.info(f"Params: {json.dumps(best.params, indent=2)}")

    # Save best params
    result = {"trial": best.number, "score": best.value, "params": best.params}
    out_path = OUTPUT_DIR / "best_params_tpe.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logging.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
