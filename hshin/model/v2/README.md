# V2 — Walk-Forward Optimized DCA Strategy

## Approach

Extends V1 base signals with **scipy.optimize SLSQP** for signal weight optimization and **nonlinear MVRV boost** at valuation extremes. Walk-forward validation prevents overfitting.

### Look-Ahead Bias Fix (2026-04-04)

Prior to this run, `mvrv_raw` was used **without lag** in `compute_dynamic_multiplier`, meaning the model could see today's MVRV to decide today's weight — a look-ahead bias. This was fixed by introducing `mvrv_raw_lagged` (`shift(1).fillna(1.5)`) so all signals are strictly lagged by 1 day. The results below reflect the corrected, leak-free model.

## Optimization Variables (7)

| Variable | Range | V1 Default | V2 Optimized |
|----------|-------|------------|--------------|
| W_MVRV | [0.05, 0.60] | 0.35 | **0.480** |
| W_MA | [0.05, 0.50] | 0.20 | 0.050 |
| W_FLOW | [0.05, 0.40] | 0.15 | **0.400** |
| W_HALVING | [0.02, 0.30] | 0.10 | 0.020 |
| W_CONFIDENCE | [0.05, 0.50] | 0.20 | 0.050 |
| DYNAMIC_STRENGTH | [2.0, 7.0] | 4.0 | **7.0** |
| MVRV_ACCUM | [0.7, 1.3] | 1.0 | **1.3** |

## Walk-Forward Design

```
Train: 2018 → 2021  →  Test: 2022
Train: 2018 → 2022  →  Test: 2023
Train: 2018 → 2023  →  Test: 2024
Train: 2018 → 2024  →  Test: 2025
```

## Results

| Variant | Score | Win Rate | Mean Excess | Median Excess |
|---------|-------|----------|-------------|---------------|
| V2 default (V1-equivalent) | 58.87% | 67.42% | +3.86% | +5.44% |
| **V2 optimized** | **61.37%** | 66.64% | **+6.52%** | **+7.68%** |

**Detailed metrics (V2 optimized):**
- Total windows: 2,557 | Wins: 1,704 | Losses: 853
- Exp-decay percentile: 56.10%
- Relative improvement: mean +18.29%, median +19.29%
- Dynamic/Uniform ratio: mean 1.18, median 1.19

### vs Other Models

| Model | Score | Win Rate | Mean Excess |
|-------|-------|----------|-------------|
| V1 base | 58.83% | 67.34% | +3.84% |
| Example 1 (MVRV+MA+Poly) | 59.54% | 60.31% | +5.70% |
| **V2 optimized** | **61.37%** | 66.64% | **+6.52%** |

## Key Insights

1. **MVRV is the dominant signal (48%)** — optimizer converged to high MVRV weight, though less extreme than pre-fix (was 59%)
2. **Exchange Flow is the #2 signal (40%)** — significantly more important than V1 assumed (15%); increased from pre-fix 29%
3. **MA, Halving, Confidence hit minimum bounds** — optimizer pushed them to floor values
4. **Aggressive DYNAMIC_STRENGTH (7.0)** — "when you're right, bet big" pays off in DCA
5. **Higher MVRV_ACCUM threshold (1.3)** — accumulation zone extends further than traditional 1.0
6. **Trade-off: higher excess, lower win rate** — V2 wins bigger but slightly less often vs V1
7. **Look-ahead fix had minimal impact on overall score** — 61.40% → 61.37%, confirming the strategy's edge was real, not from data leakage
