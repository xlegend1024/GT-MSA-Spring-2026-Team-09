# Modeling Journey

The project first built an interpretable baseline model and then refined it through experiments that added or removed candidate signals as their value became clearer. In the final stage, the main improvement came from optimizing the allocation mechanism itself. 

## Cross-Step Comparison Table

| Main idea | What changed | Score | Win rate | Exp-decay percentile | Main lesson |
| --- | --- | ---: | ---: | ---: | --- |
| 1. Baseline foundation | Built the first EDA-driven on-chain rule set | 54.36% | 68.13% | 40.59% | Simple, interpretable on-chain signals already beat uniform DCA |
| 2. Signal independence lesson | Added NVT and tested sentiment-style overlays without meaningful gain | 53.66% | 65.86% | 39.58% | More signals do not help when they overlap or dilute the base model |
| 3. Network demand discovery | Added active-address demand information | 56.98% | 72.82% | - | Demand activity added incremental value beyond valuation and flow |
| 4. Structural optimization | Removed MA200 redundancy and built a stronger flow composite | 72.48% | 74.15% | - | Simplification and deduplication beat raw feature accumulation |
| 5. Systematic maximization | Added regime-aware weighting and multi-timescale optimization | 89.17% | 89.21% | 89.13% | Careful architecture and search can push a good signal set close to the ceiling |
| 6. Softmax allocation correction | Replaced sequential allocation with direct softmax normalization | 98.08% | 96.25% | 99.92% | Allocation mechanics, not feature count, unlocked the final breakthrough |

![Scatter plot of model score versus win rate across the six modeling steps](assets/figures/modeling-score-vs-winrate-scatter.png)

*Figure M-F1. The six-step progression follows a clear path in score and win rate space. The final jump is not the result of random drift across experiments; it reflects a sequence of corrections that steadily improved both consistency and overall strategy quality, with the largest final gain arriving after allocator correction.*

## Step 1. Establishing The Baseline

- Built the first interpretable on-chain baseline using valuation, regime, cycle, and exchange-flow information.
- Reached a 54.36% score, a 68.13% win rate, and a 40.59% exp-decay percentile.
- Showed that the project had a credible baseline worth refining rather than a weak benchmark to overfit against.

## Step 2. Learning That More Signals Are Not Better

- Added NVT and tested sentiment-style overlays to see whether more signals would improve the baseline.
- Performance fell to a 53.66% score and a 65.86% win rate, and the Polymarket overlays also failed to improve the base model in a meaningful overall way.
- Showed that plausible signals are not enough and that only independent information deserves retention.

## Step 3. Finding An Independent Demand Channel

- Added active-address based demand information as a new economic channel rather than another overlapping signal.
- Improved to a 56.98% score and a 72.82% win rate, making demand the first clearly additive post-baseline feature family.
- Showed that improvement was still possible, but only when a new input captured something valuation and flow were not already providing.

## Step 4. Improving The Architecture Rather Than Expanding It

- Strengthened flow through a better composite and removed MA200 as a direct input after it proved too redundant with valuation structure.
- The score jumped to 72.48% and the win rate rose to 74.15%.
- Showed that simplification and redundancy control could create more edge than raw feature expansion.

## Step 5. Pushing A Clean Signal Set Toward Its Limit

- Kept the cleaned architecture but added regime-aware weighting, multi-timescale structure, and broader search.
- Performance rose sharply to an 89.17% score, an 89.21% win rate, and an 89.13% exp-decay percentile.
- Showed that optimization mattered most after the signal set had already been simplified and cleaned up.

## Step 6. Realizing That The Allocator Was The Bottleneck

- Changed the allocator rather than the signal set by replacing sequential allocation with direct softmax normalization over conviction.
- With the same strong inputs expressed through a better allocation rule, the model jumped to a 98.08% score, a 96.25% win rate, and a 99.92% exp-decay percentile.
- Showed the central result of the paper: the final bottleneck was not feature discovery but capital assignment.

## Final Results

This project shows that a dynamic, interpretable, long-only Bitcoin accumulation rule can improve meaningfully on uniform DCA under the same rolling-budget constraints. The final model reached a 98.08% score, a 96.25% win rate, and a 99.92% exp-decay percentile in rolling one-year evaluations.
