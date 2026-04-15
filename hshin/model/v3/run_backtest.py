"""Run V3 Q-Learning backtest with walk-forward training."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from template.prelude_template import load_data
from template.backtest_template import run_full_analysis
from hshin.model.v3.model_v3 import (
    precompute_features,
    train_qlearner,
    compute_weights_with_agent,
    compute_weights_with_agent_inverse,
    DCAQLearner,
    NUM_STATES,
    NUM_ACTIONS,
    ACTIONS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

VERSION = "v3"
RUN_RESULTS_PATH = Path(__file__).parent.parent / "run_results.json"


def _load_run_results():
    if RUN_RESULTS_PATH.exists():
        with open(RUN_RESULTS_PATH) as f:
            return json.load(f)
    return []


def _next_run_seq(results):
    if not results:
        return 1
    return max(r.get("run_seq", 0) for r in results) + 1


def main():
    btc_df = load_data()
    features_base = precompute_features(btc_df, use_poly=False)
    features_poly = precompute_features(btc_df, use_poly=True)

    # Train on base features (state space uses mvrv/ma/flow only)
    logging.info("=" * 60)
    logging.info("V3: Training Q-Learning Agent")
    logging.info(f"  States: {NUM_STATES}, Actions: {NUM_ACTIONS}")
    logging.info(f"  Action multipliers: {ACTIONS}")
    logging.info("=" * 60)

    agent = train_qlearner(
        features_base,
        train_start="2018-01-01",
        train_end="2024-12-31",
        n_epochs=100,
        alpha=0.15,
        gamma=0.9,
        dyna=200,
    )

    # Print learned Q-table summary
    mvrv_names = ["deep_val", "value", "neutral", "caution", "danger"]
    ma_names = ["bear", "bull"]
    flow_names = ["outflow", "flat", "inflow"]
    state_labels = []
    for mv in range(5):
        for ma in range(2):
            for fl in range(3):
                state_labels.append(f"{mvrv_names[mv]}_{ma_names[ma]}_{flow_names[fl]}")

    print("\n" + "=" * 60)
    print("LEARNED Q-TABLE POLICY SUMMARY")
    print("=" * 60)
    print(f"{'State':<30} {'Best Action':>12} {'Multiplier':>12} {'Q-value':>10}")
    print("-" * 70)
    for s in range(NUM_STATES):
        best_a = int(np.argmax(agent.Q[s, :]))
        best_q = agent.Q[s, best_a]
        if abs(best_q) > 1e-6:
            print(f"{state_labels[s]:<30} {best_a:>12} {ACTIONS[best_a]:>12.2f} {best_q:>10.4f}")
    print("=" * 60)

    # Extract policy for serialization
    policy = {}
    for s in range(NUM_STATES):
        best_a = int(np.argmax(agent.Q[s, :]))
        if abs(agent.Q[s, best_a]) > 1e-6:
            policy[state_labels[s]] = {
                "action": int(best_a),
                "multiplier": float(ACTIONS[best_a]),
                "q_value": float(agent.Q[s, best_a]),
            }

    base_hyperparams = {
        "alpha": 0.15,
        "gamma": 0.9,
        "n_epochs": 100,
        "dyna": 200,
        "num_states": NUM_STATES,
        "num_actions": NUM_ACTIONS,
        "actions": ACTIONS.tolist(),
    }

    # ================================================================
    # Run all 4 variant backtests
    # ================================================================
    variant_configs = [
        ("qlearner", features_base, compute_weights_with_agent,
         {**base_hyperparams}, policy,
         "Q-Learning with Dyna-Q, trained on 2018-2024"),
        ("qlearner+crypto+us_affairs", features_poly, compute_weights_with_agent,
         {**base_hyperparams}, policy,
         "Q-Learning + poly overlays"),
        ("qlearner_inverse", features_base, compute_weights_with_agent_inverse,
         {**base_hyperparams, "inverse": True}, None,
         "Inverse Q-policy: 1/original_multiplier"),
        ("qlearner_inverse+crypto+us_affairs", features_poly, compute_weights_with_agent_inverse,
         {**base_hyperparams, "inverse": True}, None,
         "Inverse Q-policy + poly overlays"),
    ]

    run_results = _load_run_results()

    for variant_name, feat_df, compute_fn, hyperparams, learned_policy, notes in variant_configs:
        logging.info(f"Running V3 {variant_name}...")

        def _make_wrapper(f_df, fn):
            def wrapper(df_window):
                if df_window.empty:
                    return pd.Series(dtype=float)
                return fn(f_df, df_window.index.min(), df_window.index.max(), agent)
            return wrapper

        output_dir = Path(__file__).parent / "output" / variant_name.replace("+", "_")
        run_full_analysis(
            btc_df=btc_df,
            features_df=feat_df,
            compute_weights_fn=_make_wrapper(feat_df, compute_fn),
            output_dir=output_dir,
            strategy_label=f"V3 {variant_name}",
        )

        with open(output_dir / "metrics.json") as f:
            metrics = json.load(f)["summary_metrics"]

        seq = _next_run_seq(run_results)
        run_id = f"v3_r{seq:03d}_{variant_name.replace('+', '_')}"
        record = {
            "run_id": run_id,
            "version": VERSION,
            "variant": variant_name,
            "run_seq": seq,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "score": round(metrics["score"], 2),
            "win_rate": round(metrics["win_rate"], 2),
            "wins": metrics["wins"],
            "losses": metrics["losses"],
            "total_windows": metrics["total_windows"],
            "mean_excess": round(metrics["mean_excess"], 2),
            "median_excess": round(metrics["median_excess"], 2),
            "exp_decay_percentile": round(metrics["exp_decay_percentile"], 2),
            "mean_ratio": round(metrics.get("mean_ratio", 0), 4),
            "median_ratio": round(metrics.get("median_ratio", 0), 4),
            "relative_improvement_pct_mean": round(metrics.get("relative_improvement_pct_mean", 0), 2),
            "relative_improvement_pct_median": round(metrics.get("relative_improvement_pct_median", 0), 2),
            "hyperparams": hyperparams,
            "learned_policy": learned_policy,
            "notes": notes,
        }
        run_results.append(record)
        with open(RUN_RESULTS_PATH, "w") as f:
            json.dump(run_results, f, indent=2)

        logging.info(
            f"  → [{run_id}] Score: {metrics['score']:.2f}% | "
            f"Win Rate: {metrics['win_rate']:.2f}%"
        )

    # Final summary
    print("\n" + "=" * 80)
    print("V3 Q-LEARNER RESULTS")
    print("=" * 80)
    print(f"{'Variant':<45} {'Score':>8} {'Win Rate':>10} {'Mean Excess':>12}")
    print("-" * 80)
    for r in run_results:
        if r["version"] == VERSION:
            print(
                f"{r['variant']:<45} {r['score']:>8.2f} "
                f"{r['win_rate']:>9.2f}% {r['mean_excess']:>11.2f}%"
            )
    print("=" * 80)
    print(f"Results saved to: {RUN_RESULTS_PATH}")


if __name__ == "__main__":
    main()
