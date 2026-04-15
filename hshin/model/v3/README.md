# V3 — Q-Learning DCA Strategy

## Approach

Tabular Q-Learning with Dyna-Q to learn optimal DCA multipliers from discretized market states. Tests whether RL can discover better timing than rule-based approaches.

## Design

| Component | Value |
|-----------|-------|
| States | 30 (MVRV 5 zones × MA 2 regimes × Flow 3 zones) |
| Actions | 5 multipliers: [0.25x, 0.5x, 1.0x, 1.5x, 2.5x] |
| Reward | Sats-per-dollar improvement vs uniform + price-direction penalty/bonus |
| Training | 100 epochs on 2018–2024, Dyna-Q 200 replays/step |

## Results (2026-04-04)

| Variant | Score | Win Rate | Mean Excess | Median Excess |
|---------|-------|----------|-------------|---------------|
| Q-Learner | 41.77% | 42.00% | -1.03% | -0.67% |
| Q-Learner Inverse | 41.62% | 46.85% | +1.33% | -0.27% |

**Detailed metrics (Q-Learner):**
- Total windows: 2,557 | Wins: 1,074 | Losses: 1,483
- Exp-decay percentile: 41.55%
- Relative improvement: mean -2.47%, median -2.09%

**Detailed metrics (Q-Learner Inverse):**
- Total windows: 2,557 | Wins: 1,198 | Losses: 1,359
- Exp-decay percentile: 36.38%
- Relative improvement: mean +3.51%, median -0.79%

> **Note:** Neither variant meets the ≥50% win rate submission threshold.

### vs All Models

| Model | Score | Win Rate | Mean Excess |
|-------|-------|----------|-------------|
| V2 optimized | **61.37%** | 66.64% | **+6.52%** |
| V1 base | 58.83% | 67.34% | +3.84% |
| Example 1 | 59.54% | 60.31% | +5.70% |
| V3 inverse | 41.62% | 46.85% | +1.33% |
| V3 original | 41.77% | 42.00% | -1.03% |

## Learned Q-Table Policy (Notable States)

| State | Learned Multiplier | Interpretation |
|-------|-------------------|----------------|
| neutral_bear_* | 2.5x | Over-buys during downtrends → harmful |
| caution_bull_flat | 2.5x | Over-buys near tops → harmful |
| caution_bear_* | 0.25x | Avoids buying in caution bear → mixed |
| danger_bull_* | 0.25x | Correctly avoids overvaluation |
| deep_val_bear_* | 1.0x | Neutral in deep value → should buy more |
| value_bear_inflow | 2.5x | Aggressive on exchange inflow in value zone |

## Key Insights

1. **RL fails to beat uniform DCA** — Score 41.77% (down from previous 28.66% due to stochastic training), consistent with data-scarcity hypothesis
2. **Inverse policy also underperforms** — Score 41.62% with 46.85% win rate, no longer beating uniform DCA
3. **Both variants far below V1/V2** — rule-based (58.8%) and optimized (61.4%) outperform RL by a large margin
4. **Root cause: ~2,900 training days for 30×5 Q-table** — insufficient exploration per state-action pair
5. **Non-stationary environment** — 2018 bear market patterns ≠ 2024 bull market patterns; Q-table can't adapt
6. **Stochastic instability** — results vary across runs due to random initialization and exploration, further limiting reliability

## Conclusion

> **Domain knowledge encoded as rules (V1) + mathematical optimization (V2) > data-driven RL (V3) for BTC DCA.**
>
> This is not a failure of RL in general, but a finding that BTC accumulation has too little data and too much regime change for tabular RL to learn useful policies. The Q-Learner consistently learns *some* structure, but the signal-to-noise ratio is too low to produce a robust, deployable strategy.
