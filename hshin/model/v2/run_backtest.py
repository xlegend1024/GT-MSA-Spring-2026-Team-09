"""Run V2 walk-forward optimization + full backtest with best params."""

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
from hshin.model.v2.model_v2 import (
    DEFAULT_PARAMS,
    precompute_features,
    compute_window_weights,
    walk_forward_optimize,
    optimize_params,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

VERSION = "v2"
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

    # Precompute features: base and poly variants
    features_base = precompute_features(btc_df, use_poly=False)
    features_poly = precompute_features(btc_df, use_poly=True)

    # ================================================================
    # Walk-forward optimization: base
    # ================================================================
    logging.info("=" * 60)
    logging.info("Walk-Forward Optimization (base)")
    logging.info("=" * 60)
    wf_results_base = walk_forward_optimize(features_base, btc_df)

    logging.info("Full-period optimization (base, 2018-2024)")
    final_params_base = optimize_params(features_base, btc_df, "2018-01-01", "2024-12-31")

    # ================================================================
    # Walk-forward optimization: poly (crypto+us_affairs)
    # ================================================================
    logging.info("=" * 60)
    logging.info("Walk-Forward Optimization (crypto+us_affairs)")
    logging.info("=" * 60)
    wf_results_poly = walk_forward_optimize(features_poly, btc_df)

    logging.info("Full-period optimization (poly, 2018-2024)")
    final_params_poly = optimize_params(features_poly, btc_df, "2018-01-01", "2024-12-31")

    # ================================================================
    # Run all 4 variant backtests
    # ================================================================
    variant_configs = [
        ("default", features_base, DEFAULT_PARAMS, None,
         "V1 default params with V2 nonlinear boost"),
        ("default+crypto+us_affairs", features_poly, DEFAULT_PARAMS, None,
         "V1 default params with poly overlays"),
        ("optimized", features_base, final_params_base, wf_results_base,
         "Walk-forward optimized (base)"),
        ("optimized+crypto+us_affairs", features_poly, final_params_poly, wf_results_poly,
         "Walk-forward optimized (poly)"),
    ]

    run_results = _load_run_results()

    for variant_name, feat_df, params, wf_data, notes in variant_configs:
        logging.info(f"Running V2 {variant_name}...")

        def _make_wrapper(f_df, p):
            def wrapper(df_window):
                if df_window.empty:
                    return pd.Series(dtype=float)
                return compute_window_weights(
                    f_df, df_window.index.min(), df_window.index.max(),
                    df_window.index.max(), p,
                )
            return wrapper

        output_dir = Path(__file__).parent / "output" / variant_name.replace("+", "_")
        run_full_analysis(
            btc_df=btc_df,
            features_df=feat_df,
            compute_weights_fn=_make_wrapper(feat_df, params),
            output_dir=output_dir,
            strategy_label=f"V2 {variant_name}",
        )

        with open(output_dir / "metrics.json") as f:
            metrics = json.load(f)["summary_metrics"]

        seq = _next_run_seq(run_results)
        run_id = f"v2_r{seq:03d}_{variant_name.replace('+', '_')}"
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
            "hyperparams": params,
            "walk_forward": wf_data,
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
    print("V2 FINAL RESULTS")
    print("=" * 80)
    print(f"{'Variant':<40} {'Score':>8} {'Win Rate':>10} {'Mean Excess':>12}")
    print("-" * 80)
    for r in run_results:
        if r["version"] == VERSION:
            print(
                f"{r['variant']:<40} {r['score']:>8.2f} "
                f"{r['win_rate']:>9.2f}% {r['mean_excess']:>11.2f}%"
            )
    print("=" * 80)
    print(f"Results saved to: {RUN_RESULTS_PATH}")


if __name__ == "__main__":
    main()
