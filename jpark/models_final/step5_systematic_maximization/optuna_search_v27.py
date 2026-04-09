"""
continue the same study with expanded search space + 600 additional trials

Strategy:
  - Load the v25 DB (model4_v25) with load_if_exists=True
  - Expand search space
      (c180/c365/c1461 → 15, r_steepness → 20, steepness → 2000)
  - N_TRIALS = 1000
      (adds 600 trials if the previous 400 are already completed)

Rationale:
  - Even after the v25 best (Trial 348), there may still be better regions
  - c180/c365/c1461 best = 7.82 / 6.13 / 4.44
      → close to upper bounds (10 / 15 / 15), leaving room for further exploration
  - steepness best = 711.9
      → extend the log search upper bound to 2000

"""

import json
import logging
from pathlib import Path

import optuna  # type: ignore
from tqdm import tqdm  # type: ignore

from template.prelude_template import load_data, backtest_dynamic_dca
from jpark.models_final.step5_systematic_maximization.model5 import (
    precompute_features,
    compute_window_weights,
)

N_TRIALS = 1000  # v25 400 completed → 600 additional
OUTPUT_DIR = Path(__file__).parent / "output" / "optuna"
DB_PATH = Path(__file__).parent.parent / "v25" / "output" / "optuna" / "optuna_v25.db"
BEST_JSON = OUTPUT_DIR / "best_params_v27.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _run_backtest(features_df, btc_df, params, trial_num):
    def wrapper(df_window):
        start = df_window.index.min()
        end = df_window.index.max()
        return compute_window_weights(features_df, start, end, end, **params)

    label = f"v25ext-t{trial_num}"
    spd_df, exp_avg = backtest_dynamic_dca(
        btc_df, wrapper, features_df=features_df, strategy_label=label
    )
    win_rate = (spd_df["dynamic_percentile"] > spd_df["uniform_percentile"]).mean() * 100
    return win_rate, exp_avg


def objective(trial: optuna.Trial, features_df, btc_df) -> float:
    params = dict(
        r_steepness=trial.suggest_float("r_steepness", 0.5, 20.0),  # ↑ 10→20
        r_threshold=trial.suggest_float("r_threshold", -2.0, 2.0),
        w_mvrv_bear=trial.suggest_float("w_mvrv_bear", 0.02, 0.50),
        w_mvrv_bull=trial.suggest_float("w_mvrv_bull", 0.00, 0.20),
        w_halving=trial.suggest_float("w_halving", 0.02, 0.40),
        w_flow=trial.suggest_float("w_flow", 0.05, 0.65),
        w_addr=trial.suggest_float("w_addr", 0.05, 0.65),
        w_ex=trial.suggest_float("w_ex", 0.02, 0.65),
        base_w=trial.suggest_float("base_w", 0.15, 0.85),
        steepness=trial.suggest_float("steepness", 5.0, 2000.0, log=True),  # ↑ 1500→2000
        threshold=trial.suggest_float("threshold", 0.25, 0.75),
        c30=trial.suggest_float("c30", 0.0, 5.0),
        c90=trial.suggest_float("c90", 0.0, 5.0),
        c180=trial.suggest_float("c180", 0.0, 15.0),  # ↑ 10→15
        c365=trial.suggest_float("c365", 0.0, 15.0),
        c1461=trial.suggest_float("c1461", 0.0, 15.0),
    )

    try:
        win, exp = _run_backtest(features_df, btc_df, params, trial.number)
    except Exception as e:
        logging.warning(f"Trial {trial.number} failed: {e}")
        return -999.0

    score = 0.5 * win + 0.5 * exp
    result = score - 0.3 * (max(0, 90 - win) + max(0, 90 - exp))

    logging.info(
        f"trial-{trial.number:3d} | win={win:.2f}% | exp={exp:.2f}% | "
        f"score={score:.2f}% | B-soft={result:.4f}"
    )
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading data (v27: Extended v25 search)...")
    btc_df = load_data()
    features_df = precompute_features(btc_df)
    logging.info(f"Features precomputed. DB: {DB_PATH}")

    storage = f"sqlite:///{DB_PATH}"
    study = optuna.create_study(
        study_name="model4_v25",
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=99),
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, N_TRIALS - completed)
    logging.info(f"v25 study loaded | completed={completed} | remaining={remaining}")

    if remaining == 0:
        logging.info("already at N_TRIALS=1000, exiting")
        return

    with tqdm(total=N_TRIALS, initial=completed, desc="Optuna v27 (ext)", ncols=100) as pbar:
        best_score = max((t.value for t in study.trials if t.value is not None), default=-999.0)

        def _cb(study, trial):
            nonlocal best_score
            if trial.value is not None and trial.value > best_score:
                best_score = trial.value
            pbar.set_postfix(
                best_t=study.best_trial.number,
                B_soft=f"{best_score:.3f}",
            )
            pbar.update(1)

        study.optimize(
            lambda t: objective(t, features_df, btc_df),
            n_trials=remaining,
            callbacks=[_cb],
            show_progress_bar=False,
        )

    best = study.best_trial
    logging.info(f"=== BEST (v27 Extended) | trial={best.number} | B-soft={best.value:.4f} ===")

    result_data = {"trial_number": best.number, "b_soft": best.value, "params": best.params}
    BEST_JSON.write_text(json.dumps(result_data, indent=2))
    logging.info(f"Best params saved to {BEST_JSON}")


if __name__ == "__main__":
    main()
