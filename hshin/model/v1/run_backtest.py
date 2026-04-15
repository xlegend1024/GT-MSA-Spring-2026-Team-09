"""Run backtest for all v1 variants and generate results."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from template.prelude_template import load_data
from template.backtest_template import run_full_analysis
from hshin.model.v1.model_v1 import (
    DYNAMIC_STRENGTH,
    W_MVRV,
    W_MA,
    W_FLOW,
    W_HALVING,
    W_CONFIDENCE,
    W_POLY,
    Variant,
    precompute_features,
    compute_window_weights,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

VERSION = "v1"
RUN_RESULTS_PATH = Path(__file__).parent.parent / "run_results.json"


def _load_run_results() -> list[dict]:
    """Load existing run results or return empty list."""
    if RUN_RESULTS_PATH.exists():
        with open(RUN_RESULTS_PATH) as f:
            return json.load(f)
    return []


def _next_run_seq(results: list[dict]) -> int:
    """Get next global run sequence number."""
    if not results:
        return 1
    return max(r.get("run_seq", 0) for r in results) + 1


def _make_run_id(version: str, seq: int, variant: str) -> str:
    """Generate run ID: {version}_r{seq:03d}_{variant_short}."""
    variant_short = variant.replace("base+", "").replace("+", "_") if "+" in variant else variant
    return f"{version}_r{seq:03d}_{variant_short}"


def _save_run_result(results: list[dict], run_record: dict) -> None:
    """Append a run record and save to disk."""
    results.append(run_record)
    RUN_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def run_variant(btc_df: pd.DataFrame, variant: Variant) -> dict:
    """Run backtest for a single variant and return metrics."""
    label = variant.value
    logging.info(f"{'='*60}")
    logging.info(f"Running variant: {label}")
    logging.info(f"{'='*60}")

    features_df = precompute_features(btc_df, variant)

    # Create wrapper that captures variant
    def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
        if df_window.empty:
            return pd.Series(dtype=float)
        start_date = df_window.index.min()
        end_date = df_window.index.max()
        current_date = end_date
        return compute_window_weights(
            features_df, start_date, end_date, current_date, variant
        )

    output_dir = Path(__file__).parent / "output" / label.replace("+", "_")

    run_full_analysis(
        btc_df=btc_df,
        features_df=features_df,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label=f"V1 {label}",
    )

    # Load metrics from generated JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path) as f:
        data = json.load(f)
    return data["summary_metrics"]


def main():
    btc_df = load_data()
    run_results = _load_run_results()

    all_results = {}
    for variant in Variant:
        try:
            metrics = run_variant(btc_df, variant)
            all_results[variant.value] = metrics

            # Build run record
            seq = _next_run_seq(run_results)
            run_id = _make_run_id(VERSION, seq, variant.value)
            run_record = {
                "run_id": run_id,
                "version": VERSION,
                "variant": variant.value,
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
                "mean_ratio": round(metrics["mean_ratio"], 4),
                "median_ratio": round(metrics["median_ratio"], 4),
                "relative_improvement_pct_mean": round(metrics["relative_improvement_pct_mean"], 2),
                "relative_improvement_pct_median": round(metrics["relative_improvement_pct_median"], 2),
                "hyperparams": {
                    "dynamic_strength": DYNAMIC_STRENGTH,
                    "w_mvrv": W_MVRV,
                    "w_ma": W_MA,
                    "w_flow": W_FLOW,
                    "w_halving": W_HALVING,
                    "w_confidence": W_CONFIDENCE,
                    "w_poly": W_POLY if variant != Variant.BASE else 0.0,
                },
                "notes": "",
            }
            _save_run_result(run_results, run_record)

            logging.info(
                f"  → [{run_id}] Score: {metrics['score']:.2f}% | "
                f"Win Rate: {metrics['win_rate']:.2f}% | "
                f"Wins: {metrics['wins']}/{metrics['total_windows']}"
            )
        except Exception as e:
            logging.error(f"Failed variant {variant.value}: {e}")
            all_results[variant.value] = {"error": str(e)}

    # Save combined results per version
    output_path = Path(__file__).parent / "output" / "all_variants_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 90)
    print("V1 MODEL COMPARISON — ALL VARIANTS")
    print("=" * 90)
    print(f"{'Run ID':<30} {'Score':>8} {'Win Rate':>10} {'Wins':>6} {'Losses':>8} {'Mean Excess':>12}")
    print("-" * 90)
    for r in run_results:
        if r["version"] == VERSION:
            print(
                f"{r['run_id']:<30} {r['score']:>8.2f} {r['win_rate']:>9.2f}% "
                f"{r['wins']:>6} {r['losses']:>8} {r['mean_excess']:>11.2f}%"
            )
    print("=" * 90)
    print(f"\nAll results saved to: {RUN_RESULTS_PATH}")


if __name__ == "__main__":
    main()
