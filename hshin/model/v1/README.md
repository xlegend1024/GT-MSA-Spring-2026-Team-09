# V1 Baseline Model — EDA-Driven DCA Strategy

## Overview

V1 implements the EDA Executive Key Takeaways as a rule-based DCA weight computation model. The baseline uses **5 on-chain signals** and is tested with **4 Polymarket overlay variants**.

## Signals (from EDA Executive)

| Signal | Feature | Logic |
|--------|---------|-------|
| MA200 Regime | `price_vs_ma` | Below MA → buy more, above → normal |
| MVRV Valuation | `mvrv_z` + `mvrv_zone` | <1.0 max accumulate, >3.5 reduce |
| Exchange Flow | `flow_signal` | Sustained inflow ≥5d → caution, sustained outflow → accumulate |
| Halving Proximity | `halving_signal` | 2–6 months pre-halving → weighted increase |
| Multi-Signal Confidence | `confidence` | MVRV + MA agreement → amplify position |

### Signal Weights (Base)

| Signal | Weight |
|--------|--------|
| MVRV | 35% |
| MA200 | 20% |
| Exchange Flow | 15% |
| Halving | 10% |
| Confidence | 20% |

When Polymarket is added, base weights are scaled by 0.85 and Polymarket gets 15%.

## Variants

| Variant | Description |
|---------|-------------|
| `base` | On-chain signals only |
| `base+crypto` | + Polymarket Crypto Index |
| `base+trump` | + Polymarket Trump Index |
| `base+us_affairs` | + Polymarket US Affairs Index |
| `base+crypto+us_affairs` | + Crypto + US Affairs |
| `base+crypto+trump` | + Crypto + Trump |
| `base+trump+us_affairs` | + Trump + US Affairs |
| `base+crypto+trump+us_affairs` | + All three Polymarket indexes |

## Results

### Comparison Table (8 Variants)

| Variant | Score | Win Rate | Wins | Losses | Mean Excess | Median Excess |
|---------|-------|----------|------|--------|-------------|---------------|
| **base** | **58.83%** | 67.34% | 1722 | 835 | **+3.84%** | +5.40% |
| base+crypto | 58.73% | 67.93% | 1737 | 820 | +3.36% | +4.88% |
| base+trump | 57.30% | 66.21% | 1693 | 864 | +3.14% | +4.93% |
| base+us_affairs | 58.24% | 67.38% | 1723 | 834 | +3.53% | **+5.56%** |
| base+crypto+us_affairs | 58.82% | **68.28%** | **1746** | **811** | +3.40% | +5.17% |
| base+crypto+trump | 58.19% | 67.46% | 1725 | 832 | +3.25% | +4.97% |
| base+trump+us_affairs | 58.08% | 67.23% | 1719 | 838 | +3.34% | +5.28% |
| base+crypto+trump+us_affairs | 58.38% | 67.70% | 1731 | 826 | +3.33% | +5.08% |

### Reference: Example 1 (MVRV + MA + Polymarket)

| Metric | Example 1 | V1 base | Delta |
|--------|-----------|---------|-------|
| Score | 59.54% | 58.83% | -0.71% |
| Win Rate | 60.31% | 67.34% | **+7.03%** |
| Mean Excess | +5.70% | +3.84% | -1.86% |

### Key Observations

1. **Base model has the highest Score (58.83%) and Mean Excess (+3.84%)** — adding Polymarket never improves these
2. **base+crypto+us_affairs has the highest Win Rate (68.28%)** but lower mean excess — more consistent, smaller gains
3. **Trump index consistently hurts performance** — all Trump-containing variants score ≤ 58.38%
4. **Combining all three Polymarket indexes (58.38%)** doesn't beat any subset — no synergy effect
5. **V1 is more conservative than Example 1** — higher win consistency (67% vs 60%), lower per-window excess
6. **Confirms EDA conclusion**: Polymarket indexes do not add meaningful signal for BTC accumulation timing

### V1 vs Example 1 Trade-off

- Example 1 uses **DYNAMIC_STRENGTH = 5.0** with aggressive MVRV asymmetric boost → higher excess per window
- V1 uses **DYNAMIC_STRENGTH = 4.0** with 5 balanced signals → higher win consistency
- V1's win rate advantage suggests the multi-signal approach is more robust

## Design Decisions

1. **All features lagged 1 day** — `.shift(1)` applied to all signal columns
2. **`allocate_sequential_stable()`** reused from template — weight sum = 1.0 guaranteed
3. **exp(combined × strength)** transformation — same pattern as template/example_1
4. **Polymarket signals as z-scored daily changes** — not raw levels (prevents scale issues)

## File Structure

```
hshin/model/v1/
├── __init__.py
├── model_v1.py          # Model logic (features + weights)
├── run_backtest.py      # Backtest runner for all variants
├── README.md            # This file
└── output/
    ├── all_variants_summary.json
    ├── base/             # Charts + metrics.json
    ├── base_crypto/
    ├── base_trump/
    ├── base_us_affairs/
    ├── base_crypto_trump/
    ├── base_trump_us_affairs/
    └── base_crypto_trump_us_affairs/
```

## Next Steps (V2)

- Feature engineering: additional on-chain metrics (NVT, hash rate momentum)
- Hyperparameter optimization (signal weights, DYNAMIC_STRENGTH)
- Walk-forward validation for overfitting check
- Regime-adaptive weight allocation
