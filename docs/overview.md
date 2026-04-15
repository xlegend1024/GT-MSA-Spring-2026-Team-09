# Overview

## Problem

We ask a practical capital allocation question: can a data-driven Bitcoin accumulation strategy deploy a fixed budget more efficiently than uniform DCA while remaining systematic, interpretable, and long-only? The goal is not to predict every price move. The goal is to improve accumulation timing so that more capital is concentrated in higher-conviction buying conditions without abandoning the discipline that makes DCA useful in the first place.

## Baseline

Uniform DCA is the benchmark because it is realistic, transparent, and already strong. It spreads purchases evenly through time, avoids discretionary overreaction, and gives us a serious baseline that many investors would actually consider using. The first baseline-foundation model already demonstrated why this comparison was meaningful: even a simple interpretable on-chain rule reached a 54.36% score with a 68.13% win rate, which was enough to show that we could improve on uniform DCA without abandoning transparency. Any later dynamic model therefore had to justify its extra complexity by improving on a rule that was not only simple and defensible, but already measurably competitive.

## Final Result

Across the six-step modeling progression, the final strategy reached a 98.08% model score, a 96.25% win rate, and a 99.92% exp-decay percentile in rolling one-year evaluations against uniform DCA. The progression moved from an interpretable on-chain baseline, through feature selection and structural simplification, into regime-aware optimization, and finally to an allocation-rule correction that unlocked the largest remaining gain.

The main takeaway is that the strongest improvement did not come from endlessly adding more signals. It came from learning which signals were actually independent, removing redundant structure, and then correcting the allocation mechanism itself. By the end of the analysis, the limiting factor was no longer feature discovery. It was how the model translated conviction into weights within a fixed rolling budget.

| Step | Score | Win rate | Exp-decay percentile | Main lesson |
| --- | ---: | ---: | ---: | --- |
| 1. Baseline foundation | 54.36% | 68.13% | 40.59% | A simple, interpretable on-chain rule set already beats uniform DCA |
| 2. Signal independence lesson | 53.66% | 65.86% | 39.58% | More signals do not help when they overlap or fail to add real decision value |
| 3. Network demand discovery | 56.98% | 72.82% | - | Active-address demand contributed a genuinely additive signal channel |
| 4. Structural optimization | 72.48% | 74.15% | - | Removing redundancy and improving compositing beat naive feature accumulation |
| 5. Systematic maximization | 89.17% | 89.21% | 89.13% | Regime-aware optimization and multi-timescale structure unlocked major gains |
| 6. Softmax allocation correction | 98.08% | 96.25% | 99.92% | The final breakthrough came from fixing allocation mechanics, not adding new data |

## Conclusion

The results show that a dynamic, interpretable, long-only Bitcoin accumulation rule can improve meaningfully on uniform DCA under the same rolling-budget constraints.

The main conclusion is that the final edge did not come from adding more signals. It came from keeping the signals that proved independent, removing redundant structure, and correcting the allocator itself.
